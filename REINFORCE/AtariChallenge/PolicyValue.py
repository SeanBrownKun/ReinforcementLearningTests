import torch
import torch.nn as nn
import torch.optim.adam as Adam


class PolicyValue(nn.Module):
    def __init__(self):
        super(PolicyValue, self).__init__()
        self.cv1 = nn.Conv2d(1, 4, 3, 2, 1)
        self.cv2 = nn.Conv2d(4, 8, 3, 2, 1)
        self.cv3 = nn.Conv2d(8, 16, 3, 2, 1)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.cv4 = nn.Conv2d(16, 32, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        # policy
        self.pfc1 = nn.Linear(32*30, 128)
        self.prelu = nn.LeakyReLU()
        self.pfc2 = nn.Linear(128, 18)

        # value
        self.vfc1 = nn.Linear(32*30, 32)
        self.vrelu = nn.LeakyReLU()
        self.vfc2 = nn.Linear(32, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.maxpool2(self.cv4(self.maxpool1(self.cv3(self.cv2(self.cv1(x)))))).flatten()
        x = x.view(batch_size, -1)
        policy = nn.functional.softmax(self.pfc2(self.prelu(self.pfc1(x))), dim=-1)
        value = self.vfc2(self.vrelu(self.vfc1(x)))
        return policy, value


