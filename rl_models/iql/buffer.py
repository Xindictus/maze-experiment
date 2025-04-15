import os
import random
from collections import deque, namedtuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, args, buffer_size, batch_size, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.args = args
        self.device = device
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=[
                "state",
                "action",
                "reward",
                "next_state",
                "done",
                "transition_info",
            ],
        )

        if self.args.leb:
            self.merge_buffers(self.args.buffer_path)
        if self.args.dqfd:
            self.load_demostration(self.args.demo_path)

    def add(self, state, action, reward, next_state, done, tr_info):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, tr_info)
        self.memory.append(e)

    def sample(self, block_nb):
        if self.args.dqfd:
            splits = [1, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0, 0, 0, 0]

            if int(self.batch_size * (1 - splits[block_nb])) > 0:
                self_data = random.sample(
                    self.memory,
                    k=int(self.batch_size * (1 - splits[block_nb])),
                )

                states = (
                    torch.from_numpy(
                        np.stack([e.state for e in self_data if e is not None])
                    )
                    .float()
                    .to(self.device)
                )
                actions = (
                    torch.from_numpy(
                        np.vstack(
                            [e.action for e in self_data if e is not None]
                        )
                    )
                    .float()
                    .to(self.device)
                )
                rewards = (
                    torch.from_numpy(
                        np.vstack(
                            [e.reward for e in self_data if e is not None]
                        )
                    )
                    .float()
                    .to(self.device)
                )
                next_states = (
                    torch.from_numpy(
                        np.stack(
                            [e.next_state for e in self_data if e is not None]
                        )
                    )
                    .float()
                    .to(self.device)
                )
                dones = (
                    torch.from_numpy(
                        np.vstack(
                            [e.done for e in self_data if e is not None]
                        ).astype(np.uint8)
                    )
                    .float()
                    .to(self.device)
                )

            if self.batch_size * splits[block_nb] > 0:
                dem_batch = random.sample(
                    self.dem_memory, k=int(self.batch_size * splits[block_nb])
                )

                if int(self.batch_size * (1 - splits[block_nb])) > 0:
                    states = torch.cat(
                        (
                            states,
                            torch.from_numpy(
                                np.stack(
                                    [
                                        e.state
                                        for e in dem_batch
                                        if e is not None
                                    ]
                                )
                            )
                            .float()
                            .to(self.device),
                        ),
                        0,
                    )
                    actions = torch.cat(
                        (
                            actions,
                            torch.from_numpy(
                                np.vstack(
                                    [
                                        e.action
                                        for e in dem_batch
                                        if e is not None
                                    ]
                                )
                            )
                            .float()
                            .to(self.device),
                        ),
                        0,
                    )
                    rewards = torch.cat(
                        (
                            rewards,
                            torch.from_numpy(
                                np.vstack(
                                    [
                                        e.reward
                                        for e in dem_batch
                                        if e is not None
                                    ]
                                )
                            )
                            .float()
                            .to(self.device),
                        ),
                        0,
                    )
                    next_states = torch.cat(
                        (
                            next_states,
                            torch.from_numpy(
                                np.stack(
                                    [
                                        e.next_state
                                        for e in dem_batch
                                        if e is not None
                                    ]
                                )
                            )
                            .float()
                            .to(self.device),
                        ),
                        0,
                    )
                    dones = torch.cat(
                        (
                            dones,
                            torch.from_numpy(
                                np.vstack(
                                    [
                                        e.done
                                        for e in dem_batch
                                        if e is not None
                                    ]
                                ).astype(np.uint8)
                            )
                            .float()
                            .to(self.device),
                        ),
                        0,
                    )

                else:
                    states = (
                        torch.from_numpy(
                            np.stack(
                                [e.state for e in dem_batch if e is not None]
                            )
                        )
                        .float()
                        .to(self.device)
                    )
                    actions = (
                        torch.from_numpy(
                            np.vstack(
                                [e.action for e in dem_batch if e is not None]
                            )
                        )
                        .float()
                        .to(self.device)
                    )
                    rewards = (
                        torch.from_numpy(
                            np.vstack(
                                [e.reward for e in dem_batch if e is not None]
                            )
                        )
                        .float()
                        .to(self.device)
                    )
                    next_states = (
                        torch.from_numpy(
                            np.stack(
                                [
                                    e.next_state
                                    for e in dem_batch
                                    if e is not None
                                ]
                            )
                        )
                        .float()
                        .to(self.device)
                    )
                    dones = (
                        torch.from_numpy(
                            np.vstack(
                                [e.done for e in dem_batch if e is not None]
                            ).astype(np.uint8)
                        )
                        .float()
                        .to(self.device)
                    )
        else:
            """Randomly sample a batch of experiences from memory."""
            experiences = random.sample(self.memory, k=self.batch_size)

            states = (
                torch.from_numpy(
                    np.stack([e.state for e in experiences if e is not None])
                )
                .float()
                .to(self.device)
            )
            actions = (
                torch.from_numpy(
                    np.vstack([e.action for e in experiences if e is not None])
                )
                .float()
                .to(self.device)
            )
            rewards = (
                torch.from_numpy(
                    np.vstack([e.reward for e in experiences if e is not None])
                )
                .float()
                .to(self.device)
            )
            next_states = (
                torch.from_numpy(
                    np.stack(
                        [e.next_state for e in experiences if e is not None]
                    )
                )
                .float()
                .to(self.device)
            )
            dones = (
                torch.from_numpy(
                    np.vstack(
                        [e.done for e in experiences if e is not None]
                    ).astype(np.uint8)
                )
                .float()
                .to(self.device)
            )

        return (states, actions, rewards, next_states, dones)

        # Load Buffer

    def load_buffer(self, path):
        self.storage = np.load(path, allow_pickle=True).tolist()

    def merge_buffers(self, path):
        files = os.listdir(path)
        buffers = []
        for file in files:
            buffers.append(
                np.load(os.path.join(path, file), allow_pickle=True).tolist()
            )
        temp_storage = []
        for buffer in buffers:
            temp_storage += buffer
        print("Merged Buffers size:", len(temp_storage))
        self.storage = deque(maxlen=len(temp_storage))
        for data in temp_storage:
            state, action, reward, next_state, done, tr_info = data
            e = self.experience(
                state, action, reward, next_state, done, tr_info
            )
            self.memory.append(e)

    def load_demostration(self, path):
        dem_data = np.load(path, allow_pickle=True).tolist()
        self.dem_memory = deque(maxlen=len(dem_data))
        for data in dem_data:
            state, action, reward, next_state, done, tr_info = data
            e = self.experience(
                state, action, reward, next_state, done, tr_info
            )
            self.dem_memory.append(e)

        print("Demonstration Buffer size:", len(self.dem_memory))

        # Save Buffer

    def save_buffer(self, path, name):
        path = os.path.join(path, name)
        self.memory = np.array(self.memory, dtype=object)
        np.save(path, self.memory)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
