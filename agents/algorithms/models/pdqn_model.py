import torch.nn as nn
import torch.nn.functional as F
import torch as T

class ParamNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super(ParamNetwork,self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 24)
        self.p = nn.Linear(24, action_size)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.p(x)

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):

        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 24)
        self.q = nn.Linear(24, action_size)


    def forward(self, x, a):
        x = T.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.q(x)
