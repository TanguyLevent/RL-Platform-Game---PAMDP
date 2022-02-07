import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class Policy(nn.Module):

    def __init__(self, state_size, action_size):
        super(Policy, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.mean = nn.Linear(128, action_size)
        self.logstd = nn.Linear(128, action_size)
        self.pi_d = nn.Linear(128, action_size)

        self.LOG_STD_MAX = 0.0
        self.LOG_STD_MIN = -3.0

    def forward(self, x):

        x = F.relu(self.fc1(x))
        mean = T.tanh(self.mean(x))
        log_std = T.tanh(self.logstd(x))
        pi_d = self.pi_d(x)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From Denis Yarats

        return mean, log_std, pi_d

    def sample_normal(self, x):

        mean, log_std, pi_d = self.forward(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action_c = T.tanh(x_t)
        log_prob_c = normal.log_prob(x_t)
        log_prob_c -= T.log(1.0 - action_c.pow(2) + 1e-8)

        dist = Categorical(logits=pi_d)
        action_d = dist.sample()
        prob_d = dist.probs
        log_prob_d = T.log(prob_d + 1e-8)

        return action_c, action_d, log_prob_c, log_prob_d, prob_d

class SoftQNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(SoftQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_size + action_size, 128)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x, a):

        x = T.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
