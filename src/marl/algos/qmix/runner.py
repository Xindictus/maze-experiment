import math
import time
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
        self.total_steps = math.ceil(self.max_game_duration / 0.2)
        self.action_duration = config.experiment.action_duration
        self.popup_window_time = config.gui.popup_window_time
        self.log_interval = config.experiment.log_interval

        self.path_to_save = f"results/{self.mode}/QMIX"
        self.best_game_score = 0
        self.last_score = 0
        self.duration_pause_total = 0
        self.current_block = 0
        self.epsilon = self.config.qmix.epsilon

        # TODO: Dirty - Refactor
        self.epsilons = []
        self.losses = []
        self.rewards = []

    def run(self):
        for block in range(self.max_blocks):
            Logger().info(f"Train Block: {block}")
            self.run_block(block, mode="train")

            Logger().info(f"Test Block: {block}")
            self.run_block(block, mode="test")

            Logger().info(f"Save checkpoint: {block}")
            self.save_chkp()

        Logger().info("QMIX Training Complete")
        self.maze.finished()

        # TODO: Dirty - Refactor
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

            Logger().debug(
                f"prev_normalized_obs (shape): {prev_normalized_obs.shape}"
            )
            Logger().debug(f"prev_normalized_obs: {prev_normalized_obs}")
            Logger().debug(
                f"experiment.global_observation: {experiment.global_observation}"
            )

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

            # TODO: Set configurable max steps

            # pre-allocate
            states = [None] * self.total_steps

            t_start = time.perf_counter()

            while True:
                step_counter += 1

                action_timer_start = time.perf_counter()

                actions = self.mac.select_actions(
                    experiment,
                    epsilon=self.epsilon,
                    mode=mode,
                )

                self.action_duration = time.perf_counter() - action_timer_start

                """
                The game understands actions only in terms of angle
                direction. Hence, we convert agent actions to the game
                specific actions (-1, 0, 1).
                """
                env_actions = experiment.get_env_actions(actions=actions)

                if mode == "test":
                    Logger().debug(f"actions: {actions}")

                # TODO: Original
                # timed_out = (
                #     time.perf_counter()
                #     - t_start
                #     - redundant_end_duration
                #     # - self.action_duration
                #     >= self.max_game_duration
                # ) or step_counter >= 200

                timed_out = step_counter >= self.total_steps

                display_text = (
                    f"Block {block_number} | "
                    + f"Round {round} | "
                    + f"Step {step_counter} | "
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
                )

                redundant_end_duration += internet_delay
                self.duration_pause_total += internet_delay

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
                if mode == "train":
                    states[step_counter - 1] = transition

                    # episode.append(transition)
                    # log_msg = f"[Round {round}] Episode size: {len(episode)}"
                    # Logger().debug(log_msg)

                # if len(episode) == self.config.qmix.batch_episode_size:
                #     self.replay_buffer.add(
                #         episode=self._pack_episode(episode=list(episode))
                #     )

                # Logger().debug(f"Replay buffer: {self.replay_buffer.list()}")

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

                # Logger().info(
                #     f"Action duration {(self.action_duration * 1000):.2f}ms | "
                #     + f"Redundant duration {(redundant_end_duration * 1000):.2f}ms | "
                #     + f"Maze time {(maze_time * 1000):.2f}ms | "
                #     + f"Transition timer {(transition_timer * 1000):.2f}ms | "
                # )

            self.last_score = episode_reward
            # TODO: Doesn't work
            self.best_game_score = max(self.best_game_score, episode_reward)

            Logger().info(
                f"[{mode.upper()}] Block {block_number} | Round {round} | "
                f"Reward: {episode_reward:.2f} | Steps: {step_counter} | "
                f"Best: {self.best_game_score:.2f} | Epsilon: {self.epsilon}"
            )

            # TODO: Dirty - Refactor
            self.epsilons.append(self.epsilon)
            self.rewards.append(episode_reward)

            if mode == "train":
                windows = self._sliding_windows(
                    states, self.config.qmix.batch_episode_size
                )

                if windows:
                    packed = [
                        self._pack_episode(episode=win) for win in windows
                    ]
                    self.replay_buffer.add_many(packed)

                # TODO: Packs leftover transitions into an episode
                # self.replay_buffer.add(
                #     self._pack_episode(episode=list(episode))
                # )

                Logger().info(f"Buffer size: {len(self.replay_buffer)}")

                # TODO: Dirty - Refactor
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
                            # TODO: Dirty - Refactor
                            rb_losses.append(loss)
                            pbar.set_postfix(loss=f"{loss:.4f}")

                self.losses.append(
                    {
                        "block": block_number,
                        "round": round,
                        "losses": rb_losses,
                    }
                )

                self.epsilon = self.config.qmix.epsilon * (
                    0.85 ** (block_number * max_rounds + (round + 1))
                )

    def _sliding_windows(self, transitions, W: int):
        # trim None
        try:
            L = transitions.index(None)
        except ValueError:
            L = len(transitions)

        if L < W:
            return []

        # creates shallow sublists
        return [transitions[i : i + W] for i in range(L - W + 1)]

    def _pack_episode(self, episode: List[Dict]) -> Dict:
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
        obs[t] = np.stack(episode[-1]["next_obs"])
        state[t] = episode[-1]["next_state"]

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
