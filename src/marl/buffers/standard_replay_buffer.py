import random
from typing import Any, Dict, List

import numpy as np

from src.marl.buffers.replay_buffer_base import ReplayBufferBase


class StandardReplayBuffer(ReplayBufferBase):
    def __init__(self, mem_size: int):
        super().__init__(mem_size=mem_size)
        # Circular buffer index
        self.next_idx = 0

    def add(self, transition: Dict[str, Any]) -> None:
        if len(self) < self.mem_size:
            self.storage.append(transition)
        else:
            self.storage[self.next_idx] = transition

        self.next_idx = (self.next_idx + 1) % self.mem_size

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size > len(self):
            raise ValueError(
                f"Batch size [{batch_size}] larger "
                + f"than buffer size [{len(self)}]"
            )
        indices = self._sample_indices(batch_size=batch_size)
        return self._encode_sample(indices)

    def _sample_indices(self, batch_size: int) -> List[int]:
        return random.choices(range(len(self)), k=batch_size)

    def _encode_sample(self, indices: List[int]) -> Dict[str, np.ndarray]:
        actions, dones, next_obses, obses, rewards = [], [], [], [], []

        for idx in indices:
            transition = self.storage[idx]
            actions.append(transition["action"])
            dones.append(transition["done"])
            next_obses.append(transition["next_obs"])
            obses.append(transition["obs"])
            rewards.append(transition["reward"])

        return {
            "obs": np.array(obses),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "next_obs": np.array(next_obses),
            "dones": np.array(dones),
        }
