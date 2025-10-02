from unittest.mock import MagicMock

import numpy as np
import torch as T

from src.config.qmix_base import QmixBaseConfig
from src.marl.algos.qmix import MAC, QmixTrainer
from src.marl.buffers.episode_replay_buffer import EpisodeReplayBuffer
from src.marl.mixing.qmix import QMixer


def create_dummy_qmix_trainer() -> QmixTrainer:
    # Default config
    config = QmixBaseConfig()

    # Init buffer
    buffer = EpisodeReplayBuffer(mem_size=100)

    # Dummy mixer and MAC
    mixer = QMixer(config, "MAIN")
    target_mixer = QMixer(config, "TARGET")

    mac = MAC(config=config)
    target_mac = MAC(config=config)

    trainer = QmixTrainer(
        buffer=buffer,
        buffer_type="episode",
        mac=mac,
        mixer=mixer,
        target_mac=target_mac,
        target_mixer=target_mixer,
        config=config,
    )

    return trainer


def convert_batch_to_torch(batch: dict, device: str) -> dict:
    batch_torch = {}

    for key, value in batch.items():
        if key == "actions":
            tensor = T.tensor(value, dtype=T.long)
        else:
            tensor = T.tensor(value, dtype=T.float32)
        batch_torch[key] = tensor.to(device)

    return batch_torch


def generate_fake_qmix_batch(
    batch_size=4,
    episode_len=5,
    n_agents=2,
    input_dim=4,
    n_actions=3,
    state_dim=8,
) -> dict:
    # Generates a fake batch to test the QMIX trainer.
    T = episode_len
    B = batch_size
    N = n_agents

    return {
        "obs": np.random.randn(B, T + 1, N, input_dim).astype(np.float32),
        "actions": np.random.randint(
            low=0, high=n_actions, size=(B, T, N, 1), dtype=np.int64
        ),
        "rewards": np.random.randn(B, T, 1).astype(np.float32),
        "dones": np.random.choice([0, 1], size=(B, T, 1)).astype(np.float32),
        "avail_actions": (
            np.random.choice([0, 1], size=(B, T + 1, N, n_actions)).astype(
                np.float32
            )
        ),
        "state": np.random.randn(B, T + 1, state_dim).astype(np.float32),
        "mask": np.ones((B, T, 1), dtype=np.float32),
    }


def test_qmix_trainer_fake_batch():
    config = QmixBaseConfig()
    fake_batch = generate_fake_qmix_batch()
    for k, v in fake_batch.items():
        print(f"{k}: {v.shape}")
    fake_batch_torch = convert_batch_to_torch(fake_batch, config.device)

    for k, v in fake_batch_torch.items():
        print(
            f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}"
        )

    # Initialize dummy components
    trainer = create_dummy_qmix_trainer()
    trainer.buffer.sample = MagicMock(return_value=fake_batch_torch)

    trainer.train()
