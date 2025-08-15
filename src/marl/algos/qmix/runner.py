import time
from typing import Dict, List

import numpy as np

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

    def run(self):
        for block in range(self.max_blocks):
            Logger().info(f"Test Block: {block}")
            self.run_block(block, mode="test")

            Logger().info(f"Train Block: {block}")
            self.run_block(block, mode="train")

            Logger().info(f"Save checkpoint: {block}")
            self.save_chkp()

        Logger().info("QMIX Training Complete")
        self.maze.finished()

    def run_block(self, block_number: int, mode: str):
        # max_rounds = int(self.games_per_block / 2)
        max_rounds = int(self.games_per_block)
        # max_rounds = 1

        for round in range(max_rounds):
            is_paused = True
            while is_paused:
                Logger().info("Game Reseting")
                raw_obs, setting_up_duration, is_paused = self.maze.reset(mode)

            experiment = Experiment(self.config.qmix)
            normalized_obs = experiment._normalize_global_state(raw_obs)
            experiment.global_observation = normalized_obs

            Logger().debug(normalized_obs.shape)
            Logger().debug(normalized_obs)
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
            episode = []

            start_game_time = time.time()

            while True:
                step_counter += 1

                actions = self.mac.select_actions(
                    experiment, epsilon=self.config.qmix.epsilon
                )

                """
                The game understands actions only in terms of angle
                direction. Hence, we convert agent actions to the game
                specific actions (-1, 0, 1).
                """
                env_actions = experiment.get_env_actions(actions=actions)

                Logger().debug(f"actions: {actions}")

                timed_out = (
                    time.time() - start_game_time - redundant_end_duration
                    >= self.max_game_duration
                )

                display_text = (
                    f"Block {block_number}, Round {round}, Step {step_counter}"
                )

                (
                    raw_obs_next,
                    reward,
                    done,
                    fps,
                    pause_duration,
                    action_pair,
                    internet_delay,
                ) = self.maze.step(
                    action_agent=env_actions,
                    timed_out=timed_out,
                    action_duration=self.action_duration,
                    mode=mode,
                    text=display_text,
                )

                redundant_end_duration += pause_duration
                self.duration_pause_total += pause_duration

                Logger().debug(raw_obs_next.shape)
                experiment.global_observation = raw_obs_next
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
                # TODO: if mode == "train":
                episode.append(transition)
                log_msg = f"[Round {round}] Episode size: {len(episode)}"
                Logger().debug(log_msg)

                if len(episode) == self.config.qmix.batch_episode_size:
                    self.replay_buffer.add(
                        episode=self.pack_episode(episode=episode)
                    )
                    episode = []

                episode_reward += reward

                if done:
                    if not timed_out:
                        Logger().info("Goal reached")
                    else:
                        Logger().info("Timeout")
                    time.sleep(self.popup_window_time)
                    break

                # update obs
                local_obs = next_local_obs
                global_state = next_global_state

            self.last_score = episode_reward
            self.best_game_score = max(self.best_game_score, episode_reward)

            Logger().info(
                f"[{mode.upper()}] Block {block_number} | Round {round} | "
                f"Reward: {episode_reward:.2f} | Steps: {step_counter} | "
                f"Best: {self.best_game_score:.2f}"
            )

            if mode == "train":
                # Packs leftover transitions into an episode
                # self.replay_buffer.add(self.pack_episode(episode=episode))

                Logger().debug(f"Buffer size: {len(self.replay_buffer)}")

                if len(self.replay_buffer) >= self.config.qmix.batch_size:
                    Logger().info("Training...")
                    for _ in range(self.config.experiment.epochs):
                        self.trainer.train()

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
        avail_actions = np.ones((t + 1, N, 3), dtype=np.float32)

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
