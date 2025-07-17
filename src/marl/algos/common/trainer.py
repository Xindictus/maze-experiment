from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train(self, batch):
        # TODO: revisit to include typing
        """
        Trains on a batch of data.
        """
        raise NotImplementedError

    # @abstractmethod
    # def gradient_update(self):
    #     """
    #     Performs the gradient update step (backpropagation + optimizer).
    #     """
    #     raise NotImplementedError
