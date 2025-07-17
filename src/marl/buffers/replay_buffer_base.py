import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class ReplayBufferBase(ABC):
    def __init__(self, mem_size: int):
        self.mem_size = mem_size
        self.storage: List[Dict[str, Any]] = []

    @abstractmethod
    def add(self, transition: Dict[str, Any]) -> None:
        # Adds a single transition state to the buffer.
        raise NotImplementedError

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Samples a batch of full episodes.
        """
        if batch_size > len(self):
            raise ValueError(
                f"Batch size [{batch_size}] larger "
                + f"than buffer size [{len(self)}]"
            )
        indices = self._sample_indices(batch_size=batch_size)
        return self._encode_sample(indices)

    def _sample_indices(self, batch_size: int) -> List[int]:
        return random.choices(range(len(self)), k=batch_size)

    def __len__(self) -> int:
        # Returns the current size of the buffer.
        return len(self.storage)
