import math
from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train(self):
        """
        Trains on a batch of data.
        """
        raise NotImplementedError

    # TODO: Revisit
    @staticmethod
    def get_distance_traveled(dist_travel, prev_observation, observation):
        dist_travel += math.sqrt(
            (prev_observation[0] - observation[0]) ** 2
            + (prev_observation[1] - observation[1]) ** 2
        )
        return dist_travel

    # @abstractmethod
    # def gradient_update(self):
    #     """
    #     Performs the gradient update step (backpropagation + optimizer).
    #     """
    #     raise NotImplementedError
