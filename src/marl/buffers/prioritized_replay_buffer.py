from typing import Any, Dict

from src.marl.buffers.replay_buffer_base import ReplayBufferBase


class PrioritizedReplayBuffer(ReplayBufferBase):
    def __init__(self, mem_size: int) -> None:
        super().__init__(mem_size=mem_size)
        # TODO: Add priorities

    def add(self, transition: Dict[str, Any]) -> None:
        # TODO
        err = "PrioritizedReplayBuffer.add() not implemented."
        raise NotImplementedError(err)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        # TODO
        err = "PrioritizedReplayBuffer.sample() not implemented."
        raise NotImplementedError(err)
