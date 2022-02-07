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

### Libraries

- [GYM](https://gym.openai.com/docs/): `pip install gym`
- [Platform](https://github.com/cycraig/gym-platform): `pip install -e git+https://github.com/cycraig/gym-platform#egg=gym_platform`

Algorithms
------------------------------------------

Each algorithm can be trained and tested (test not yet finished).

```shell
python ./main.py --algorithm NAME train  # Train algorithm NAME
python ./main.py --algorithm NAME test  # Test algorithm NAME
```
