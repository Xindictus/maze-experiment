from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ReplayBufferBase(ABC):
    def __init__(self, mem_size: int):
        self.mem_size = mem_size
        self.storage: List[Dict[str, Any]] = []

    @abstractmethod
    def add(self, transition: Dict[str, Any]) -> None:
        # Adds a single transition state to the buffer.
        raise NotImplementedError

    @abstractmethod
    def sample(self, batch_size: int) -> Dict[str, Any]:
        # Samples a batch of transitions.
        raise NotImplementedError

    def __len__(self) -> int:
        # Returns the current size of the buffer.
        return len(self.storage)
