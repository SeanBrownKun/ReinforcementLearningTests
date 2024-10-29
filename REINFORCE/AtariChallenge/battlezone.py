import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import numpy as np
from PolicyValue import PolicyValue
from random import shuffle
from math import floor, ceil
from time import perf_counter


class Battlezone:

    def __init__(self):
        self.env = gym.make("ALE/BattleZone-v5", render_mode="rgb_array", mode=2, obs_type="grayscale")
        self.pv_new = PolicyValue()
        self.pv_old = PolicyValue()
        self.pv_old.load_state_dict(self.pv_new.state_dict())
        self.crit = nn.MSELoss()
        self.optim = torch.optim.Adam(self.pv_new.parameters(), lr=0.0001)

        # parameters
        """reward_stretch = 13
        self.reward_convolution_filter = np.linspace(0.0, 2.0, num=reward_stretch)
        self.reward_convolution_filter = np.square(self.reward_convolution_filter)
        half = ceil(len(self.reward_convolution_filter) / 2)
        self.reward_convolution_filter[half:] = 0
        self.reward_convolution_filter = self.reward_convolution_filter[1:-1]"""
        self.previous = None
        gamma = 0.8
        self.gamma = np.array([gamma**i for i in range(3000)])

    def run_n_episodes(self, n_episodes, length_episodes, epoch=0):
        state_log = []
        reward_log = []
        action_log = []
        for episode in range(n_episodes):
            # reset episode
            rewards = []
            state, _ = self.env.reset()
            self.env.render()
            # play game
            for i in range(length_episodes):
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                state_log.append(state)
                policy, _ = self.pv_old(state)
                action = torch.distributions.Categorical(policy).sample().item()
                obs, reward, terminated, truncated, info = self.env.step(action)
                rewards.append(reward)
                action_log.append(action)
                state = obs
                if terminated or truncated:
                    break

            # penalize losing
            # rewards[-1] = -1000
            rewards = np.array(rewards)
            rewards /= 1000
            Gts = [
                np.sum(rewards[i:] * self.gamma[:len(rewards)-i])
                for i in range(len(rewards))
            ]

            """convolved_rewards = np.convolve(rewards, self.reward_convolution_filter, mode="same")
            convolved_rewards /= 1000"""
            reward_log.append(Gts)
        average_score = sum([np.max(x) for x in reward_log]) / len(reward_log)
        reward_log = np.concatenate(reward_log).tolist()
        if epoch:
            print(f"average score of epoch {epoch}: {average_score}")
        else:
            print(f"average_score of this game set: {average_score}")
        return list(zip(state_log, reward_log, action_log))

    def infer_a_game(self, model, deterministic=True):
        self.env = gym.make("ALE/BattleZone-v5", render_mode="human", mode=2, obs_type="grayscale")
        self.pv_old.load_state_dict(torch.load(model))
        state, _ = self.env.reset()
        self.env.render()
        # play game
        for i in range(1000):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            policy, _ = self.pv_old(state)
            if deterministic:
                action = torch.argmax(policy)
            else:
                action = torch.distributions.Categorical(policy).sample().item()
            obs, reward, terminated, truncated, info = self.env.step(action)
            state = obs
            if terminated or truncated:
                break

    def train_model(self, state_reward_pairs, n_epochs=5, batch_size=4):
        self.pv_new.train()
        """state1 = state_reward_pairs[0][0]
        state1policy = self.pv_old(state1)
        print(state1policy)
        if self.previous != None:
            print(f"the two states are equal: {state1.sum() == self.previous.sum()}")
        else:
            self.previous = state1"""

        shuffle(state_reward_pairs)
        states, rewards, actions = zip(*state_reward_pairs)

        n_batches = floor(len(states) / batch_size)
        left_over = len(states) % batch_size
        if left_over > 0:
            states = states[:-left_over]
            rewards = rewards[:-left_over]
            actions = actions[:-left_over]

        states = torch.stack(states)
        states = states.reshape((n_batches, batch_size, 1, 210, 160))
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = rewards.reshape((n_batches, batch_size, 1))
        actions = torch.tensor(actions, dtype=torch.long)
        actions = actions.reshape((n_batches, batch_size, 1))

        max_cutoff = 0
        time_for_get_probs = 0
        for epoch in range(n_epochs):
            for state, reward, action in zip(states, rewards, actions):
                self.optim.zero_grad()
                a = perf_counter()
                new_policy, value = self.pv_new(state)
                old_policy, _ = self.pv_old(state)
                b = perf_counter()
                time_for_get_probs += (b-a)

                new_action_prob = torch.gather(new_policy, dim=-1, index=action)
                old_action_prob = torch.gather(old_policy, dim=-1, index=action)
                r_theta = new_action_prob / old_action_prob
                r_theta_clamped = torch.clamp(r_theta, 0.8, 1.2)
                cut_off = r_theta - r_theta_clamped
                cut_off *= cut_off
                cut_off = cut_off.sum().item()
                if cut_off > max_cutoff:
                    max_cutoff = cut_off
                    # print(max_cutoff)

                A = reward - value
                entropy = -torch.sum(new_policy * torch.log(new_policy + 1e-10)) * 0.1
                L_clip = -torch.min(r_theta * A, r_theta_clamped * A).sum()
                V_loss = self.crit(value, reward)
                Loss = L_clip + V_loss - entropy

                Loss.backward()
                self.optim.step()

        # print(f"time for neural nets: {time_for_get_probs / n_epochs}")
        self.pv_old.load_state_dict(self.pv_new.state_dict())


if __name__ == '__main__':
    game = Battlezone()
    episode_length = 1000
    infer = False
    if infer:
        game.infer_a_game("test1", deterministic=False)
        input()

    for i in range(1000):
        state_reward_pairs = game.run_n_episodes(8, episode_length, epoch=i)
        game.train_model(state_reward_pairs, n_epochs=4, batch_size=8)
        torch.save(game.pv_old.state_dict(), "test1")

