import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests as requests

from src.config import FullConfig
from src.marl.algos.common.action_space import ActionSpace
from src.marl.rewards.reward_engine import RewardEngine
from src.utils.logger import Logger


class GameController:
    reset_endpoints: Dict[str, str] = {"train": "/reset", "test": "/testreset"}

    def __init__(self, config: FullConfig):
        Logger().info("Initializing Maze3D GameController...")

        self.action_space = ActionSpace(list(range(3)))
        self.config = config
        self.done = False
        self.fps = 60
        self.internet_delay: List[float] = []

        # Initialize reusable session
        self.session = requests.Session()

        # Initialize connection
        self.set_host()
        self.send_config()
        self.agent_ready()

        self.observation, _, _ = self.reset("test")
        self.observation_shape = (len(self.observation),)

    def send_config(self) -> None:
        json_config = self.config.game.model_dump(mode="json")

        json_config["action_duration"] = self.config.experiment.action_duration
        json_config["max_duration"] = self.config.experiment.max_duration
        json_config["popup_window_time"] = self.config.gui.popup_window_time
        json_config["start_up_screen_display_duration"] = (
            self.config.gui.start_up_screen_display_duration
        )

        endpoint = f"{self.config.network.maze_rl}/config"

        try:
            self.session.post(endpoint, json=json_config).json()
        except Exception as e:
            Logger().exception(f"Failed to send config: {e}")

    def set_host(self) -> None:
        endpoint: str = f"{self.config.network.ip_distributor}/set_server_host"
        payload = {"server_host": self.config.network.maze_server}

        while True:
            try:
                Logger().warning("Trying to connect to game host...")
                self.session.post(endpoint, json=payload)
                Logger().info("Connected to host successfully!")
                break
            except Exception as e:
                Logger().exception(f"IP host is offline: {e}")
                time.sleep(3)

        Logger().info("Connected to host successfully!")

    def agent_ready(self) -> None:
        endpoint: str = f"{self.config.network.ip_distributor}/agent_ready"

        while True:
            try:
                res = self.session.get(endpoint).json()
                if res.get("command") == "player_ready":
                    return
            except Exception as e:
                Logger().error(f"/agent_ready not returned: {e}")
                time.sleep(3)

    def send(
        self, namespace: str, method: str = "GET", data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        while True:
            try:
                url = self.config.network.maze_rl + namespace

                if method == "GET":
                    res = self.session.get(url).json()
                else:
                    res = self.session.post(url, json=data).json()

                if res.get("command") == "player_ready":
                    continue
                return res
            except Exception as e:
                Logger().error(f"/send failed: {e}")
                Logger().warning("Retrying after agent ready...")
                self.agent_ready()
                time.sleep(0.1)

    def reset(self, type: str) -> Tuple[np.ndarray, float, bool]:
        start_time = time.time()
        res = self.send(self.reset_endpoints[type])
        set_up_time = time.time() - start_time

        return (np.asarray(res["observation"]), set_up_time, res["pause"])

    def training(self, cycle: int, total_cycles: int) -> None:
        self.send(
            "/training",
            method="POST",
            data={"cycle": cycle, "total_cycles": total_cycles},
        )

    def finished(self) -> None:
        Logger().info("Finished experiment")
        self.send("/finished")

    def step(
        self,
        action_agent: List[int],
        timed_out: Optional[float],
        action_duration: float,
        mode: str,
        text: str,
    ) -> Tuple[np.ndarray, float, bool, float, float, List[int], float]:
        """
        Performs the action of the agent to the environment for
        action_duration time. Simultaneously, receives input from the user
        via the keyboard arrows.

        :param action_agent: the action of the agent
                - gives -1 for down
                - 0 for nothing and 1 for up
        :param timed_out: used
        :param action_duration: the duration of the agent's action on the game
        :param mode: training or test
        :return: a transition [
                    observation,
                    reward,
                    done,
                    timeout,
                    train_fps,
                    duration_pause,
                    action_list
                ]
        """
        payload = {
            "action_agent": int(action_agent[0]),
            "action_duration": action_duration,
            "display_text": text,
            "mode": mode,
            "second_agent_action": int(action_agent[1]),
            "timed_out": timed_out,
        }

        start_time = time.time()
        Logger().debug(f"-- UNITY STEP -- {payload}")
        res = self.send("/step_two_agents", method="POST", data=payload)
        delay = time.time() - start_time

        self.internet_delay.append(delay)
        self.observation = np.array(res["observation"])
        # true if goal_reached OR timeout
        self.done = res["done"]

        agent_action = res["agent_action"]
        duration_pause = res["duration_pause"]
        fps = res["fps"]
        human_action = res["human_action"]
        internet_pause = delay - duration_pause - action_duration
        reward = RewardEngine.compute_reward(self.done, timed_out)

        return (
            self.observation,
            reward,
            self.done,
            fps,
            duration_pause,
            [agent_action, human_action],
            internet_pause,
        )
