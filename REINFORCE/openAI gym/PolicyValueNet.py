import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self):
        super(PolicyValueNet, self).__init__()
        self.fc1 = nn.Linear(4, 8)
        self.tanh = nn.Tanh()
        self.policy = nn.Linear(8, 2)
        self.value = nn.Linear(8, 1)

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        policy = nn.functional.softmax(self.policy(x), dim=-1)
        value = self.value(x)
        return policy, value

