# Platform Project InstaDeep
Project PAMDP - Technical test for InstaDeep

Requirements
------------------------------------------

### Python (mandatory)

[Python](https://www.python.org/) 3.8+ (3.10 issue with Pytorch)

### Requirements (mandatory)

[PyTorch](https://pytorch.org/get-started/locally/)

Environment
------------------------------------------

- [GYM](https://gym.openai.com/docs/): `pip install gym`
- [Platform](https://github.com/cycraig/gym-platform): `pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform`

Algorithms
------------------------------------------

* [Random](https://github.com/TanguyLevent/Platform_Project_InstaDeep/blob/main/agents/algorithms/random.py) Random Agent
* [DQN](https://github.com/TanguyLevent/Platform_Project_InstaDeep/blob/main/agents/algorithms/dqn.py) DQN with random continuous action
* [PDQN](https://github.com/TanguyLevent/Platform_Project_InstaDeep/blob/main/agents/algorithms/pdqn.py) 
* [Hybrid Soft Actor Critic](https://github.com/TanguyLevent/Platform_Project_InstaDeep/blob/main/agents/algorithms/hybrid_sac.py)

Each algorithm can be trained and tested (test not yet finished).

```shell
python ./main.py --algorithm NAME train  # Train algorithm NAME
python ./main.py --algorithm NAME test  # Test algorithm NAME
```
