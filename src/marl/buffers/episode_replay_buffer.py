from typing import Any, Dict, List

import numpy as np

from src.marl.buffers.replay_buffer_base import ReplayBufferBase
from src.utils.logger import Logger


class EpisodeReplayBuffer(ReplayBufferBase):
    """
    Stores full episodes (not individual transitions).

    Each episode is a dictionary with the full trajectory:
    - obs: (T+1, n_agents, obs_dim)
    - actions: (T, n_agents, 1)
    - rewards: (T, 1)
    - dones: (T, 1)
    - avail_actions: (T+1, n_agents, n_actions)
    - state: (T+1, state_dim)
    - mask: (T, 1)
    """

    def __init__(self, mem_size: int):
        super().__init__(mem_size=mem_size)
        self.next_idx = 0

    def add(self, episode: Dict[str, Any]) -> None:
        """
        Adds one full episode to the buffer.
        """
        if len(self) < self.mem_size:
            self.storage.append(episode)
        else:
            self.storage[self.next_idx] = episode

        self.next_idx = (self.next_idx + 1) % self.mem_size

    def add_many(self, episodes: List[Dict]) -> None:
        """
        Bulk add episodes.
        """
        for ep in episodes:
            self.add(ep)

    def list(self) -> List[Dict[str, Any]]:
        return self.storage

    def _encode_sample(self, indices: List[int]) -> Dict[str, np.ndarray]:
        Logger().debug(f"Indices: {indices}")

        batch = {}

        for key in self.storage[0].keys():
            batch[key] = [self.storage[i][key] for i in indices]

        return {k: np.array(v) for k, v in batch.items()}
        # return {k: T.stack(v) for k, v in batch.items()}
