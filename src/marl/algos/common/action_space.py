import numpy as np

from typing import List


class ActionSpace:
    def __init__(self, valid_actions: List[int]):
        self.actions = valid_actions
        self.n_actions = len(valid_actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self) -> int:
        return np.random.randint(self.low, self.high + 1, 2)
