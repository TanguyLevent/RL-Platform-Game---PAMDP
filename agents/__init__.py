from agents.algorithms.random import RandomAgent
from agents.algorithms.dqn import DQN
from agents.algorithms.hybrid_sac import HSAC
from agents.algorithms.pdqn import PDQN

# Available algorithms
ALGORITHMS = dict(

    random=RandomAgent(name="Random"),
    dqn=DQN(name="DDQN"),
    hsac=HSAC(name="Hybrid Soft Actor Critic"),
    pdqn=PDQN(name="PDDQN")

)
