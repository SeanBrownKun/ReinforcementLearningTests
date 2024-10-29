import gym
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import sleep
from random import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, BatchSampler, TensorDataset

from PolicyValueNet import PolicyValueNet
from Autoencoder import Autoencoder


class CarRacing:

    def __init__(self):
        self.env = gym.make("CarRacing-v2", render_mode="rgb_array")  # , obs_type="grayscale")
        self.autoencoder = Autoencoder()
        self.pv_new = PolicyValueNet()
        self.pv_old = PolicyValueNet()
        self.pv_old.load_state_dict(self.pv_new.state_dict())

        self.crit = nn.MSELoss()
        self.optim = torch.optim.Adam(self.pv_new.parameters(), lr=0.0001, eps=1e-5)

        self.autoen_optim = torch.optim.Adam(self.autoencoder.parameters(), lr=0.0001)
        self.autoen_loss = nn.MSELoss()

    def train_autoencoder(self):
        self.autoencoder.train()
        for game in range(1000):
            state, _ = self.env.reset()
            self.env.render()
            total_loss = 0.0
            actions = np.random.uniform(-0.3, 0.2, 5000)
            for i in range(5000):
                random_action = [actions[i], .2, 0]
                obs, reward, terminated, _, _ = self.env.step(random_action)
                obs_tensor = torch.tensor([obs], dtype=torch.float32).permute(0, 3, 1, 2) / 255
                autoen_out = self.autoencoder(obs_tensor)
                autoen_out_img = autoen_out.permute(0, 2, 3, 1).detach().numpy() * 255
                autoen_out_img = autoen_out_img[0]

                obs = obs.astype(np.uint8)
                autoen_out_img = autoen_out_img.astype(np.uint8)

                imgs = cv2.hconcat([obs, autoen_out_img])
                cv2.imshow("original", imgs)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if terminated:
                    break
                if i == 999:
                    print(f"total loss: {total_loss}\n")
                    print(obs.mean())
                    print(autoen_out_img.mean())
                    print("---")
                    print(obs_tensor.mean())
                    print(autoen_out.mean())
                    print()

                self.autoen_optim.zero_grad()
                loss = self.autoen_loss(autoen_out, obs_tensor)
                loss.backward()
                self.autoen_optim.step()
                total_loss += loss.item()


if __name__ == '__main__':
    cr = CarRacing()
    cr.train_autoencoder()


