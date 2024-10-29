from random import randint
import torch


class Game:

    def __init__(self, board_size=5):
        self.board_size = board_size
        self.goal = [randint(0, board_size-1), randint(0, board_size-1)]
        self.player = [randint(0, board_size-1), randint(0, board_size-1)]
        self.log = []
        self.old_log = {}
        self.score = 0

    def reset(self):
        self.goal = [randint(0, self.board_size - 1), randint(0, self.board_size - 1)]
        self.player = [randint(0, self.board_size - 1), randint(0, self.board_size - 1)]
        self.log = []
        self.score = 0

    def get_state(self):
        return torch.tensor([self.goal[0]-self.player[0], self.goal[1]-self.player[1]], dtype=torch.float32)

    def perform_action(self, action):
        old_state = self.get_state()
        if action == 0:
            self.player[1] += 1
        elif action == 1:
            self.player[0] += 1
        elif action == 2:
            self.player[1] -= 1
        elif action == 3:
            self.player[0] -= 1
        new_state = self.get_state()

        if self.goal[0] == self.player[0] and self.goal[1] == self.player[1]:
            self.score += 1
            self.goal = [randint(0, self.board_size - 1), randint(0, self.board_size - 1)]

        self.log.append([old_state, action, new_state, 0])

    def prepare_log(self):
        for entry in range(len(self.log)):
            self.log[entry][-1] = torch.tensor(self.score, dtype=torch.float32)

    def show_game(self):
        for i in range(2*self.board_size):
            for j in range(2*self.board_size):
                if self.goal[1] == i and self.goal[0] == j:
                    print(" o ", end="")
                elif self.player[1] == i and self.player[0] == j:
                    print(" v ", end="")
                else:
                    print("   ", end="")
            print()
