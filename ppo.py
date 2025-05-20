from collections import Counter

import gym_super_mario_bros
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from wrappers import *

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


def make_env():
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    return env


def get_reward(r):
    r = np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + 0.001 * r
    return r


class ActorCritic(nn.Module):
    def __init__(self, n_frame, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_frame, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(20736, 512)
        self.policy_head = nn.Linear(512, act_dim)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x):
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x.permute(2, 0, 1)
        x = self.net(x)
        x = x.reshape(-1, 20736)
        x = torch.relu(self.linear(x))

        return self.policy_head(x), self.value_head(x).squeeze(-1)

    def act(self, obs):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, value


def compute_gae_batch(rewards, values, dones, gamma=0.99, lam=0.95):
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(N, device=device)

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


def rollout_with_bootstrap(envs, model, rollout_steps, init_obs):
    obs = init_obs
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    obs_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], []

    for _ in range(rollout_steps):
        obs_buf.append(obs)

        with torch.no_grad():
            action, logp, value = model.act(obs)

        val_buf.append(value)
        logp_buf.append(logp)
        act_buf.append(action)

        actions = action.cpu().numpy()
        next_obs, reward, done, infos = envs.step(actions)

        reward = [get_reward(r) for r in reward]
        # done = np.logical_or(terminated)

        rew_buf.append(torch.tensor(reward, dtype=torch.float32).to(device))
        done_buf.append(torch.tensor(done, dtype=torch.float32).to(device))

        for i, d in enumerate(done):
            if d:
                print(f"Env {i} done. Resetting. (info: {infos[i]})")
                next_obs[i] = envs.envs[i].reset()

        obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        max_stage = max([i["stage"] for i in infos])

    with torch.no_grad():
        _, last_value = model.forward(obs)

    obs_buf = torch.stack(obs_buf)
    act_buf = torch.stack(act_buf)
    rew_buf = torch.stack(rew_buf)
    done_buf = torch.stack(done_buf)
    val_buf = torch.stack(val_buf)
    val_buf = torch.cat([val_buf, last_value.unsqueeze(0)], dim=0)
    logp_buf = torch.stack(logp_buf)

    adv_buf, ret_buf = compute_gae_batch(rew_buf, val_buf, done_buf)
    adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)

    return {
        "obs": obs_buf,  # [T, N, obs_dim]
        "actions": act_buf,
        "logprobs": logp_buf,
        "advantages": adv_buf,
        "returns": ret_buf,
        "max_stage": max_stage,
        "last_obs": obs,
    }


def evaluate_policy(env, model, episodes=5, render=False):
    """
    Function to evaluate the learned policy

    Args:
    env: gym.Env single environment (not vector!)

    model: ActorCritic model

    episodes: number of episodes to evaluate

    render: whether to visualize (if True, display on screen)

    Returns:
    avg_return: average total reward
    """
    model.eval()
    total_returns = []
    actions = []
    stages = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        if render:
            env.render()
        while not done:
            obs_tensor = (
                torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                logits, _ = model(obs_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.probs.argmax(dim=-1).item()  # greedy action
                actions.append(action)

            obs, reward, done, info = env.step(action)
            stages.append(info["stage"])
            total_reward += reward

        total_returns.append(total_reward)
        info["action_count"] = Counter(actions)
    model.train()
    return np.mean(total_returns), info, max(stages)


def train_ppo():
    num_env = 8
    envs = gym.vector.SyncVectorEnv([lambda: make_env() for _ in range(num_env)])
    obs_dim = envs.single_observation_space.shape[-1]
    act_dim = envs.single_action_space.n
    print(f"{obs_dim=} {act_dim=}")
    model = ActorCritic(obs_dim, act_dim).to(device)
    model.load_state_dict(torch.load("mario_1_1.pt"))
    optimizer = optim.Adam(model.parameters(), lr=2.5e-4)

    rollout_steps = 128
    epochs = 4
    minibatch_size = 64
    clip_eps = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    eval_env = make_env()
    eval_env.reset()

    init_obs = envs.reset()
    update = 0
    while True:
        update += 1
        batch = rollout_with_bootstrap(envs, model, rollout_steps, init_obs)
        init_obs = batch["last_obs"]

        T, N = rollout_steps, envs.num_envs
        total_size = T * N

        obs = batch["obs"].reshape(total_size, *envs.single_observation_space.shape)
        act = batch["actions"].reshape(total_size)
        logp_old = batch["logprobs"].reshape(total_size)
        adv = batch["advantages"].reshape(total_size)
        ret = batch["returns"].reshape(total_size)

        for _ in range(epochs):
            idx = torch.randperm(total_size)
            for start in range(0, total_size, minibatch_size):
                i = idx[start : start + minibatch_size]
                logits, value = model(obs[i])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[i])
                ratio = torch.exp(logp - logp_old[i])
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]
                policy_loss = -torch.min(ratio * adv[i], clipped).mean()
                value_loss = (ret[i] - value).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # logging
        avg_return = batch["returns"].mean().item()
        max_stage = batch["max_stage"]
        print(f"Update {update}: avg return = {avg_return:.2f} {max_stage=}")

        # eval and save
        if update % 10 == 0:
            avg_score, info, eval_max_stage = evaluate_policy(
                eval_env, model, episodes=1, render=False
            )
            print(f"[Eval] Update {update}: avg return = {avg_score:.2f} info: {info}")
            if eval_max_stage > 1:
                torch.save(model.state_dict(), "mario_1_1_clear.pt")
                break
        if update > 0 and update % 50 == 0:
            torch.save(model.state_dict(), "mario_1_1_ppo.pt")


if __name__ == "__main__":
    train_ppo()
