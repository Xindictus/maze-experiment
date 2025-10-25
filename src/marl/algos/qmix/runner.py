import math
import time
from typing import Any, Dict, List

import numpy as np
from joblib import dump
from tqdm import tqdm

from src.config.full_config import FullConfig
from src.game.experiment import Experiment
from src.game.game_controller import GameController
from src.marl.algos.common.epsilon_decay import EpsilonDecayRate
from src.marl.algos.qmix import MAC, QmixTrainer
from src.marl.buffers import ReplayBufferBase
from src.utils.logger import Logger
from src.utils.sigint import sigint_controller


class QmixRunner:
    def __init__(
        self,
        config: FullConfig,
        game_controller: GameController,
        mac: MAC,
        trainer: QmixTrainer,
        replay_buffer: ReplayBufferBase,
    ):
        self.config = config
        self.out_dir = config.out_dir
        self.maze = game_controller
        self.mac = mac
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.replay_buffer_type = config.experiment.buffer_type

        self.goal = config.game.goal
        self.games_per_block = config.experiment.games_per_block
        self.max_blocks = config.experiment.max_blocks
        self.max_game_duration = config.experiment.max_duration
        self.total_steps = math.ceil(self.max_game_duration / 0.2)
        self.action_duration = config.experiment.action_duration
        self.popup_window_time = config.gui.popup_window_time
        self.epsilon = self.config.qmix.max_epsilon
        self.eps_dr = EpsilonDecayRate(
            eps_max=self.config.qmix.max_epsilon,
            eps_min=self.config.qmix.min_epsilon,
            T=(self.max_blocks * self.games_per_block),
            method=self.config.qmix.epsilon_decay_method,
        )

        self.best_game_score = -9_999
        self.last_score = 0
        self.duration_pause_total = 0
        self.current_block = 0

        # TODO: Dirty - Refactor
        self.to_dump = ["epsilons", "losses", "rewards", "wins"]
        self.epsilons = []
        self.losses = []
        self.rewards = []
        self.wins = {}

    def _run_loop(self) -> None:
        for block in range(self.max_blocks):
            if block not in self.wins:
                self.wins[block] = {"train": [], "test": []}

            Logger().info(f"Train Block: {block}")
            self.run_block(block, mode="train")

            # TODO: Adjust with CLI arg
            # Logger().info(f"Test Block: {block}")
            # self.run_block(block, mode="test")

            if sigint_controller.is_requested():
                Logger().warning("Shutdown requested: aborting block!")
                return

            Logger().info(f"Save checkpoint: {block}")
            self.save_chkp()

        total_test_wins = sum(
            sum(self.wins[v]["test"]) for _, v in enumerate(self.wins)
        )
        total_train_wins = sum(
            sum(self.wins[v]["train"]) for _, v in enumerate(self.wins)
        )

        Logger().info(
            f"[Block #{block}]: QMIX training complete | "
            f"Total train wins: {total_train_wins} | "
            f"Total test wins: {total_test_wins}"
        )

        self.maze.finished()

        self.save_results()

    def run(self) -> None:
        try:
            self._run_loop()
        except KeyboardInterrupt:
            if sigint_controller.is_requested():
                Logger().warning("Shutdown requested, exiting...")
        finally:
            if sigint_controller.is_requested():
                self.save_results()

    def run_block(self, block_number: int, mode: str):
        max_rounds = int(self.games_per_block)

        experiment = Experiment(self.config.qmix)

        for round in range(max_rounds):
            if sigint_controller.is_requested():
                Logger().warning("Shutdown requested: aborting round!")
                return

            is_paused = True

            # Initialize hidden states for all agents
            # at the start of each round - GRU only
            self.mac.init_hidden()
            self.maze.reset_reward_engine()

            while is_paused:
                Logger().info("Game Reseting")
                Logger().info(f"Starting block {block_number}, round {round}")
                (
                    prev_raw_obs,
                    init_ball_pos_r,
                    setting_up_duration,
                    is_paused,
                ) = self.maze.reset(mode)

            prev_normalized_obs = experiment._normalize_global_state(
                prev_raw_obs
            )
            experiment.global_observation = (
                prev_normalized_obs,
                init_ball_pos_r,
            )

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
                    observations=local_obs,
                    # experiment,
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
                    init_ball_pos,
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
                experiment.global_observation = (
                    normalized_obs_next,
                    init_ball_pos,
                )

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
                    if self._is_episode_buffer():
                        states[step_counter - 1] = transition
                    elif self._is_standard_buffer():
                        self.replay_buffer.add(
                            transition=self._pack_transition(transition)
                        )

                episode_reward += reward
                goal_reached = done and not timed_out

                # TODO: Improve
                if done:
                    if not timed_out:
                        Logger().info("Goal reached")

                        if mode == "train":
                            self.wins[block_number][mode].append(1)
                    else:
                        Logger().info("Timeout")

                        if mode == "train":
                            self.wins[block_number][mode].append(0)

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
                #     + f"Maze time {(maze_time * 1000):.2f}ms"
                # )

            self.last_score = episode_reward
            self.best_game_score = max(self.best_game_score, episode_reward)

            total_mode_wins = sum(
                sum(self.wins[block].get(mode, [])) for block in self.wins
            )

            Logger().info(
                f"[{mode.upper()}] Block {block_number} | Round {round} | "
                f"Reward: {episode_reward:.2f} | Steps: {step_counter} | "
                f"Goal reached: {goal_reached} | Epsilon: {self.epsilon:.4f}"
            )
            Logger().info(
                f"Total Wins for [{mode.upper()}]: {total_mode_wins} | "
                f"Best: {self.best_game_score:.2f}"
            )

            # TODO: Dirty - Refactor
            self.epsilons.append(self.epsilon)
            self.rewards.append(episode_reward)

            if mode == "train":
                if self._is_episode_buffer():
                    windows = self._sliding_windows(
                        states, self.config.qmix.batch_episode_size
                    )

                    if windows:
                        packed = [
                            self._pack_episode(episode=win) for win in windows
                        ]
                        self.replay_buffer.add_many(packed)

                Logger().info(f"Buffer size: {len(self.replay_buffer)}")

                # TODO: Dirty - Refactor
                rb_losses = []

                if len(self.replay_buffer) >= self.config.qmix.batch_size:
                    Logger().info("Training...")

                    pbar = tqdm(
                        range(self.config.experiment.update_cycles),
                        desc="Update Cycles",
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

                self.epsilon = self.eps_dr.decay(
                    self._global_round(block_number, max_rounds, round)
                )

    def _global_round(self, block_number, max_rounds, round_idx):
        return block_number * max_rounds + (round_idx + 1)

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

    def _is_episode_buffer(self):
        return self.config.experiment.buffer_type == "episode"

    def _is_standard_buffer(self):
        return self.config.experiment.buffer_type == "standard"

    def _pack_transition(self, transition: Dict[str, Any]) -> Dict:
        return self._pack_episode([transition])

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

    def save_results(self) -> None:
        for name in self.to_dump:
            val = getattr(self, name)

            dump(
                val,
                f"{self.out_dir}/{name}.joblib",
                compress=("gzip", 5),
            )
