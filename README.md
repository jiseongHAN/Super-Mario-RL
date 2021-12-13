# :mushroom: Super-Mario-RL 

This is a private project to make Super Mario Agent.

It consists of training an agent to clear Super Mario Bros with deep reinforcement learning methods.

Here are my super mario agents with dueling network. ( trained 7,000 epoch )

<p float="center">
  <img src="/mario1.gif" width="350" />
  <img src="/mario14.gif" width="350" /> 
</p>

# Get started

## Cloning git

```
git clone https://github.com/jiseongHAN/Super-Mario-RL.git
cd Super-Mario-RL
```

## Install Requirements
```
pip install -r requirements.txt
```

## Or Install Manually
* Install [openAI gym](http://gym.openai.com/)
```
pip install 'gym'
```
* Install [Pytorch](https://pytorch.org/)
```
pip install torch torchvision
```
* Install [nes-py](https://pypi.org/project/nes-py/)
```
pip install nes-py
```
* Install [gym-super-mario-bros](https://pypi.org/project/gym-super-mario-bros/)
```
pip install gym-super-mario-bros
```

# Running

## Train

* Train with dueling dqn.
```
python duel_dqn.py
```

### Result
* score.p : save total score every 50 episode
* *.pth : save weight of q, q_target every 50 training


## Evaluate
* (Now, pre-trained agent has been corrupted😢)
* Test and render trained agent.
* To test our agent, we need 'q_target.pth' that generated at the training step.
```
python eval.py
```
* Or you can use your own agent.
```
python eval.py your_own_agent.pth
```

## Reference
[Wang, Ziyu, et al. "Dueling network architectures for deep reinforcement learning." International conference on machine learning. PMLR, 2016.](https://arxiv.org/pdf/1511.06581.pdf)
