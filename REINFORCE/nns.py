import torch
import torch.nn as nn


def clip(x, lower, upper):
    return max(lower, min(upper, x))


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 4)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0002)
        self.epsilon = 0.2

    def forward(self, x):
        x = nn.functional.tanh(self.fc1(x))
        x = nn.functional.softmax(self.fc2(x), dim=-1)
        return x

    def train_model(self, log, value, old_policy):
        """# REINFORCE
        self.train()
        log_length = len(log)
        gamma = 0.99
        learning_rate = 0.01
        for idx, entry in enumerate(log):
            old_s, action, new_s, score = entry
            action_prob = self(old_s)[action]

            G = gamma ** (log_length - idx) * score - value(old_s).squeeze(0)
            total_gradient = learning_rate * -torch.log(action_prob) * G

            self.optim.zero_grad()
            total_gradient.backward()
            self.optim.step()"""

        # PPO
        self.train()
        log_length = len(log)
        gamma = 0.99
        for idx, entry in enumerate(log):
            old_s, action, new_s, score = entry
            action_prob = self(old_s)[action]
            action_prob_old = old_policy(old_s)[action]
            r_theta = action_prob / action_prob_old

            A = gamma ** (log_length - idx) * score - value(old_s).squeeze(0)
            L_clip_theta = -min(r_theta * A, clip(r_theta, 1 - self.epsilon, 1 + self.epsilon) * A)  # minus was very important!

            self.optim.zero_grad()
            L_clip_theta.backward()
            self.optim.step()


class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        x = nn.functional.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_model(self, log):
        self.train()
        criterion = torch.nn.MSELoss()
        for entry in log:
            old_s, action, new_s, score = entry
            self.optim.zero_grad()
            y_pred = self(old_s)
            y_actual = score
            loss = criterion(y_pred, y_actual)
            loss.backward()
            self.optim.step()



