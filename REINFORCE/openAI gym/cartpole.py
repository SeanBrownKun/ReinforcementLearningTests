import gym
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

import torch
import torch.nn as nn

from PolicyValueNet import PolicyValueNet

env = gym.make("CartPole-v1", render_mode="rgb_array")

# hyperparams
lr = 0.0001
gamma = 0.95
entropy_coeff = 0.01

pvn = PolicyValueNet()
pvn.eval()
pvn_old = PolicyValueNet()
pvn_old.load_state_dict(pvn.state_dict())
pvn_old.eval()
value_crit = nn.MSELoss()
optim = torch.optim.Adam(pvn.parameters(), lr=lr, eps=1e-5)

game_scores = []
trends = 0
reward_shaper = [0.8, 0.6, 0.4, 0.2, 0]

torch.autograd.set_detect_anomaly(True)

for game in range(2000):
    pvn.eval()
    pvn_old.load_state_dict(pvn.state_dict())

    # game start
    state, _ = env.reset()
    env.render()  # render -> show

    run_reward = 0

    log = []
    rewards = np.zeros((1000,))

    # play
    for i in range(1000):
        optim.zero_grad()

        state = torch.from_numpy(state).to(dtype=torch.float32)
        policy, value = pvn(state)
        action = torch.distributions.Categorical(policy).sample().item()
        obs, reward, terminated, _, _ = env.step(action)
        reward = torch.tensor(reward).to(dtype=torch.float32).requires_grad_(False)

        if terminated:
            game_scores.append(run_reward)
            if len(game_scores) > 3:
                trend = (sum(game_scores[1:]) - sum(game_scores[:-1])) / len(game_scores)
                if game % 10 == 9:
                    print(f"--- game {game} ---")
                    print(f"trend:  {'+' if trend >= 0 else ''}{trend}")
                    print(f"average: {sum(game_scores)/len(game_scores)}")

            if len(game_scores) > 50:
                game_scores.pop(0)
            break

        rewards[1:] = rewards[:-1]
        rewards[0] = np.max(rewards) + reward * gamma ** i
        log.append([state, action])
        state = obs
        run_reward += reward

    rewards = torch.tensor(rewards, dtype=torch.float32)
    rewards = (rewards - rewards.mean()) - rewards.std()
    # train
    pvn.train()
    for epoch in range(3):
        for idx, (state, action) in enumerate(log):
            policy, value = pvn(state)
            # value = value[0]
            avg_reward = rewards.mean()
            entropy = -torch.sum(policy * torch.log(policy + 1e-10))
            policy_old, _ = pvn_old(state)
            r_theta = policy[action] / policy_old[action]
            r_theta_clamped = torch.clamp(r_theta, 0.8, 1.2)

            # policy loss
            A = rewards[idx]  # - avg_reward  # - value
            L_clip = -lr * torch.min(r_theta * A, r_theta_clamped * A)

            # value loss
            # value_loss = value_crit(value, rewards[idx])

            # loss with entropy
            loss = L_clip  # - entropy_coeff*entropy  # + value_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

