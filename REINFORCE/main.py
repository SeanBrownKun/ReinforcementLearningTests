from nns import Policy, Value
from game import Game
from random import randint
import torch
import torch.nn as nn


policy = Policy()
old_policy = Policy()
old_policy.load_state_dict(policy.state_dict())
value = Value()
game = Game()


highscore = 0
for epoch in range(100000):
    game.reset()
    policy.eval()
    old_policy.eval()
    value.eval()
    if epoch % 600 == 599:
        for turns in range(50):
            game.show_game()
            current_state = game.get_state()
            print(f"value: {value(current_state).item()}")

            # choose action
            action = policy(current_state)
            print(action)
            action = torch.distributions.Categorical(action).sample().item()

            game.perform_action(action)
            input()
    else:
        for turns in range(30):
            current_state = game.get_state()
            action = policy(current_state)
            action = torch.distributions.Categorical(action).sample().item()
            game.perform_action(action)

    if game.score > highscore:
        highscore = game.score
        print(f"new highscore: {highscore}")
    game.prepare_log()
    for mini_epoch in range(5):
        value.train_model(game.log)
        policy.train_model(game.log, value, old_policy)
    old_policy.load_state_dict(policy.state_dict())







