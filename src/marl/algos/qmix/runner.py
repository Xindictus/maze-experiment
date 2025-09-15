import time
from collections import deque
from typing import Dict, List

import numpy as np
from joblib import dump
from tqdm import tqdm

from src.config.full_config import FullConfig
from src.game.experiment import Experiment
from src.game.game_controller import GameController
from src.marl.algos.qmix import MAC, QmixTrainer
from src.marl.buffers.episode_replay_buffer import EpisodeReplayBuffer
from src.utils.logger import Logger


class EpsilonDecay:
    def __init__(self, eps0: float, eps_min: float, X: int, p: float):
        # TODO: PYDOC
        self.eps0 = eps0
        self.eps_min = eps_min
        self.X = X
        self.p = p
        # current step
        self.t = 0
        self.epsilon = eps0

    def step(self) -> float:
        # Advance one round and return the new epsilon.
        if self.t >= self.X:
            self.epsilon = self.eps_min
            return self.epsilon

        # compute factor
        num = (
            self.eps_min
            + (self.eps0 - self.eps_min)
            * (1 - (self.t + 1) / self.X) ** self.p
        )
        den = (
            self.eps_min
            + (self.eps0 - self.eps_min) * (1 - self.t / self.X) ** self.p
        )
        d_t = num / den

        # update epsilon
        self.epsilon *= d_t
        self.t += 1
        return self.epsilon


class QmixRunner:
    def __init__(
        self,
        config: FullConfig,
        game_controller: GameController,
        mac: MAC,
        trainer: QmixTrainer,
        replay_buffer: EpisodeReplayBuffer,
    ):
        self.config = config
        self.out_dir = config.out_dir
        self.maze = game_controller
        self.mac = mac
        self.trainer = trainer
        self.replay_buffer = replay_buffer

        self.mode = config.experiment.mode
        self.goal = config.game.goal
        self.games_per_block = config.experiment.games_per_block
        self.max_blocks = config.experiment.max_blocks
        self.max_game_duration = config.experiment.max_duration
        self.action_duration = config.experiment.action_duration
        self.popup_window_time = config.gui.popup_window_time
        self.log_interval = config.experiment.log_interval

        self.path_to_save = f"results/{self.mode}/QMIX"
        self.best_game_score = 0
        self.last_score = 0
        self.duration_pause_total = 0
        self.current_block = 0
        self.epsilon = self.config.qmix.epsilon

        # TODO: Dirty
        self.epsilons = []
        self.losses = []
        self.rewards = []
        # self.decay_rate = pow(
        #     0.01 / self.epsilon,
        #     1 / (self.max_blocks * self.games_per_block)
        # )
        # self.decay = EpsilonDecay(
        #     eps0=self.epsilon,
        #     eps_min=0.01,
        #     X=(self.max_blocks * self.games_per_block),
        #     p=0.5,
        # )

    def run(self):
        for block in range(self.max_blocks):
            Logger().info(f"Test Block: {block}")
            self.run_block(block, mode="test")

            Logger().info(f"Train Block: {block}")
            self.run_block(block, mode="train")

            Logger().info(f"Save checkpoint: {block}")
            self.save_chkp()

        Logger().info(f"Final Test Block: {block}")
        self.run_block(block, mode="test")

        Logger().info("QMIX Training Complete")
        self.maze.finished()

        # TODO: Dirty
        dump(
            self.epsilons,
            f"{self.out_dir}/epsilons.joblib",
            compress=("gzip", 5),
        )
        dump(
            self.rewards,
            f"{self.out_dir}/rewards.joblib",
            compress=("gzip", 5),
        )
        dump(
            self.losses,
            f"{self.out_dir}/losses.joblib",
            compress=("gzip", 5),
        )

    def run_block(self, block_number: int, mode: str):
        max_rounds = int(self.games_per_block)

        for round in range(max_rounds):
            is_paused = True
            while is_paused:
                Logger().info("Game Reseting")
                Logger().info(f"Starting block {block_number}, round {round}")
                prev_raw_obs, setting_up_duration, is_paused = self.maze.reset(
                    mode
                )

            experiment = Experiment(self.config.qmix)
            prev_normalized_obs = experiment._normalize_global_state(
                prev_raw_obs
            )
            experiment.global_observation = prev_normalized_obs

            Logger().debug(prev_normalized_obs.shape)
            Logger().debug(prev_normalized_obs)
            Logger().debug(experiment.global_observation)

            local_obs = [
                experiment.get_local_obs(agent_id)
                for agent_id in range(self.config.qmix.n_agents)
            ]
            global_state = experiment.global_observation

            step_counter = 0
            episode_reward = 0
            timed_out = False
            redundant_end_duration = 0
            self.duration_pause_total = 0
            episode = deque(maxlen=self.config.qmix.batch_episode_size)

            t_start = time.perf_counter()

            while True:
                t1 = time.perf_counter()

                step_counter += 1

                actions = self.mac.select_actions(
                    experiment,
                    epsilon=self.epsilon,
                    mode=mode,
                )

                """
                The game understands actions only in terms of angle
                direction. Hence, we convert agent actions to the game
                specific actions (-1, 0, 1).
                """
                env_actions = experiment.get_env_actions(actions=actions)

                Logger().debug(f"actions: {actions}")

                timed_out = (
                    time.perf_counter() - t_start - redundant_end_duration
                    >= self.max_game_duration
                )

                display_text = (
                    f"Block {block_number}, Round {round}, Step {step_counter}"
                )

                (
                    next_raw_obs,
                    reward,
                    done,
                    fps,
                    pause_duration,
                    action_pair,
                    internet_delay,
                    dist_travelled,
                ) = self.maze.step(
                    action_agent=env_actions,
                    timed_out=timed_out,
                    action_duration=self.action_duration,
                    prev_obs=prev_raw_obs,
                    mode=mode,
                    text=display_text,
                    t1=t1,
                )

                t5 = time.perf_counter()
                elapsed_ms = (t5 - t1) * 1000
                elapsed_ms2 = (t5 - t_start) * 1000
                Logger().debug(f"Elapsed (t1): {elapsed_ms:.2f} ms")
                Logger().debug(f"Elapsed (tStart): {elapsed_ms2:.2f} ms")

                redundant_end_duration += pause_duration
                self.duration_pause_total += pause_duration

                # Normalize global state
                normalized_obs_next = experiment._normalize_global_state(
                    next_raw_obs
                )
                experiment.global_observation = normalized_obs_next

                Logger().debug(f"Normalized OBS (Next): {normalized_obs_next}")

                next_local_obs = [
                    experiment.get_local_obs(agent_id)
                    for agent_id in range(self.config.qmix.n_agents)
                ]
                next_global_state = experiment.global_observation

                transition = {
                    "obs": local_obs,
                    "state": global_state,
                    "actions": actions,
                    "reward": reward,
                    "next_obs": next_local_obs,
                    "next_state": next_global_state,
                    "done": done,
                }

                Logger().debug(f"Transition: {transition}")

                # Append to our buffer episode only when it's train blocks
                # TODO: success only buffer?
                if mode == "train":
                    episode.append(transition)
                    log_msg = f"[Round {round}] Episode size: {len(episode)}"
                    Logger().debug(log_msg)

                if len(episode) == self.config.qmix.batch_episode_size:
                    self.replay_buffer.add(
                        episode=self.pack_episode(episode=list(episode))
                    )

                Logger().debug(f"Replay buffer: {self.replay_buffer.list()}")

                episode_reward += reward

                if done:
                    if not timed_out:
                        Logger().info("Goal reached")
                    else:
                        Logger().info("Timeout")

                    end = time.perf_counter()
                    Logger().info(f"Round duration: {(end - t_start):0.2f}")

                    time.sleep(self.popup_window_time)
                    break

                # update obs
                local_obs = next_local_obs
                global_state = next_global_state

            self.last_score = episode_reward
            # TODO: Doesn't work
            self.best_game_score = max(self.best_game_score, episode_reward)

            Logger().info(
                f"[{mode.upper()}] Block {block_number} | Round {round} | "
                f"Reward: {episode_reward:.2f} | Steps: {step_counter} | "
                f"Best: {self.best_game_score:.2f} | Epsilon: {self.epsilon}"
            )

            # TODO: Dirty
            self.epsilons.append(self.epsilon)
            self.rewards.append(episode_reward)

            if mode == "train":
                # TODO: Packs leftover transitions into an episode
                self.replay_buffer.add(
                    self.pack_episode(episode=list(episode))
                )

                Logger().info(f"Buffer size: {len(self.replay_buffer)}")

                # TODO: Dirty
                rb_losses = []

                if len(self.replay_buffer) >= self.config.qmix.batch_size:
                    Logger().info("Training...")

                    pbar = tqdm(
                        range(self.config.experiment.epochs),
                        desc="Epochs",
                        dynamic_ncols=True,
                    )

                    for e in pbar:
                        loss = self.trainer.train()

                        if e % 10 == 0:
                            # TODO: Dirty
                            rb_losses.append(loss)
                            pbar.set_postfix(loss=f"{loss:.4f}")

                self.losses.append(
                    {
                        "block": block_number,
                        "round": round,
                        "losses": rb_losses,
                    }
                )
                # self.epsilon = self.decay.step()
                self.epsilon = self.config.qmix.epsilon * (
                    0.85 ** ((block_number + 1) * (round + 1))
                )

    def pack_episode(self, episode: List[Dict]) -> Dict:
        t = len(episode)
        N = len(episode[0]["obs"])
        obs_dim = episode[0]["obs"][0].shape[0]
        state_dim = episode[0]["state"].shape[0]

        # Allocate storage
        obs = np.zeros((t + 1, N, obs_dim), dtype=np.float32)
        state = np.zeros((t + 1, state_dim), dtype=np.float32)
        actions = np.zeros((t, N, 1), dtype=np.int64)
        rewards = np.zeros((t, 1), dtype=np.float32)
        dones = np.zeros((t, 1), dtype=np.float32)
        mask = np.ones((t, 1), dtype=np.float32)

        # Static avail_actions: [T + 1, N, n_actions] filled with 1s
        # TODO: static avail_actions. [T + 1, N, n_actions]
        # TODO: needs to be dynamic based on agent initialization.
        # todo: for now hardcoding it
        avail_actions = np.ones(
            (t + 1, N, self.config.qmix.n_actions), dtype=np.float32
        )

        for t_step in range(t):
            transition = episode[t_step]

            # [N, obs_dim]
            obs[t_step] = np.stack(transition["obs"])
            # [state_dim]
            state[t_step] = transition["state"]
            actions[t_step] = np.array(
                transition["actions"], dtype=np.int64
            ).reshape(N, 1)
            rewards[t_step] = transition["reward"]
            dones[t_step] = float(transition["done"])

        # Handle final obs and state
        obs[t] = np.stack(episode[-1]["obs"])
        state[t] = episode[-1]["state"]

        return {
            # [T + 1, N, obs_dim]
            "obs": obs,
            # [T + 1, state_dim]
            "state": state,
            # [T, N, 1]
            "actions": actions,
            # [T, 1]
            "rewards": rewards,
            # [T, 1]
            "dones": dones,
            # [T, 1]
            "mask": mask,
            # [T + 1, N, n_actions]
            "avail_actions": avail_actions,
        }

    def save_chkp(self) -> None:
        pass


def print_dict_shapes(d):
    for k, v in d.items():
        print(f"{k}: {v.shape}")
