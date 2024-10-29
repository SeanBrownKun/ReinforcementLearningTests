import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 2))
        self.value = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 1))

    def forward(self, x):
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

