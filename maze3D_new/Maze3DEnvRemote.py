import random
import time
import traceback

import numpy as np
import requests as requests

from game.utils import get_config
from utils.logger import Logger

logger = Logger().get_logger()

# TODO: While exp running ~> agent interacts 5 times per second (~200ms)


def reward_function_timeout_penalty(
    goal_reached: bool, timed_out: bool
) -> int:
    """_summary_

    - for every timestep -1
    - timed out -50
    - reached goal +100

    Args:
        goal_reached (bool): _description_
        timed_out (bool): _description_

    Returns:
        int: reward
    """
    if goal_reached and not timed_out:
        return 10

    if timed_out:
        return -1

    return -1


class ActionSpace:
    def __init__(self):
        self.actions = list(range(3))
        self.shape = 2
        self.actions_number = len(self.actions)
        self.high = self.actions[-1]
        self.low = self.actions[0]

    def sample(self):
        return np.random.randint(self.low, self.high + 1, 2)


class Maze3D:
    def __init__(self, config=None):
        print("Init Maze3D")
        self.config = config
        self.network_config = get_config("", "network_config.yaml")
        self.ip_host = self.network_config["ip_distributor"]
        self.outer_host = self.network_config["maze_server"]
        self.host = self.network_config["maze_rl"]

        self.action_space = ActionSpace()
        self.fps = 60
        self.done = False
        self.set_host()
        self.send_config()
        self.agent_ready()
        self.observation, _, _ = self.reset("test")
        self.observation_shape = (len(self.observation),)
        self.internet_delay = []

    def send_config(self):
        config = {}

        while True:
            # try:
            # print("Sending config",self.config)
            mode = self.config["Experiment"]["mode"]
            config["discrete_input"] = self.config["game"]["discrete_input"]
            config["max_duration"] = self.config["Experiment"][mode][
                "max_duration"
            ]
            config["action_duration"] = self.config["Experiment"][mode][
                "action_duration"
            ]
            config["human_speed"] = self.config["game"]["human_speed"]
            config["agent_speed"] = self.config["game"]["agent_speed"]
            config["discrete_angle_change"] = self.config["game"][
                "discrete_angle_change"
            ]
            config["human_assist"] = self.config["game"]["human_assist"]
            config["human_only"] = self.config["game"]["human_only"]
            config["start_up_screen_display_duration"] = self.config["GUI"][
                "start_up_screen_display_duration"
            ]
            config["popup_window_time"] = self.config["GUI"][
                "popup_window_time"
            ]
            print(config)
            requests.post(self.host + "/config", json=config).json()
            return
        # except Exception as e:
        #     print("/agent_ready not returned", e)
        #     time.sleep(1)

    def set_host(self):
        while True:
            print("Im trying this at least")
            try:
                requests.post(
                    self.ip_host + "/set_server_host",
                    json={"server_host": self.outer_host},
                ).json()
                break
            except Exception as e:
                print("ip host offline", e)
                time.sleep(1)
        print("I succseded here")

    def agent_ready(self):
        while True:
            try:
                res = requests.get(self.host + "/agent_ready").json()
                if "command" in res and res["command"] == "player_ready":
                    break
            except Exception:
                # print("/agent_ready not returned", e)
                time.sleep(1)

    def send(self, namespace, method="GET", data=None):
        while True:
            try:
                if method == "GET":
                    res = requests.get(self.host + namespace).json()
                else:
                    res = requests.post(
                        self.host + namespace, json=data
                    ).json()

                if "command" in res and res["command"] == "player_ready":
                    continue
                return res
            except Exception:
                # in here when wrong request is given
                # traceback.print_exc()
                self.agent_ready()
                time.sleep(0.1)

    def reset(self, type):
        # print("reset")
        start_time = time.time()
        if type == "train":
            res = self.send("/reset")
        elif type == "test":
            res = self.send("/testreset")
        set_up_time = time.time() - start_time
        # print("reset time:", set_up_time)
        # return np.array(res['observation']), res['setting_up_duration']
        return np.asarray(res["observation"]), set_up_time, res["pause"]

    def training(self, cycle, total_cycles):
        self.send(
            "/training", "POST", {"cycle": cycle, "total_cycles": total_cycles}
        )

    def finished(self):
        print("finished")
        self.send("/finished", "GET")

    def step(self, action_agent, timed_out, action_duration, mode, text):
        """
        Performs the action of the agent to the environment for action_duration time.
        Simultaneously, receives input from the user via the keyboard arrows.

        :param action_agent: the action of the agent. gives -1 for down, 0 for nothing and 1 for up
        :param timed_out: used
        :param action_duration: the duration of the agent's action on the game
        :param mode: training or test
        :return: a transition [observation, reward, done, timeout, train_fps, duration_pause, action_list]
        """
        # print("step", timed_out)
        # if timed_out:
        #     print("timeout", timed_out, int(time.time()))
        if mode == "one_agent" or mode == "human":
            payload = {
                "action_agent": action_agent,
                "second_agent_action": -2,
                "action_duration": action_duration,
                "timed_out": timed_out,
                "mode": mode,
                "display_text": text,
            }
            start_time = time.time()
            res = self.send("/step", method="POST", data=payload)
        elif mode == "two_agents":
            # print("action_agent", type(action_agent[0]), action_agent[1])
            payload = {
                "action_agent": int(action_agent[0]),
                "second_agent_action": int(action_agent[1]),
                "action_duration": action_duration,
                "timed_out": timed_out,
                "mode": mode,
                "display_text": text,
            }
            start_time = time.time()
            res = self.send("/step_two_agents", method="POST", data=payload)

            # print('But it never comes back')

        delay = time.time() - start_time
        self.internet_delay.append(delay)
        self.observation = np.array(res["observation"])
        self.done = res["done"]  # true if goal_reached OR timeout
        fps = res["fps"]
        human_action = res["human_action"]
        agent_action = res["agent_action"]
        duration_pause = res["duration_pause"]
        internet_pause = delay - duration_pause - action_duration
        reward = reward_function_timeout_penalty(self.done, timed_out)

        return (
            self.observation,
            reward,
            self.done,
            fps,
            duration_pause,
            [agent_action, human_action],
            internet_pause,
        )


if __name__ == "__main__":
    """Dummy execution"""
    while True:
        try:
            maze = Maze3D()
            while True:
                maze.step(random.randint(-1, 1), None, None, 200)
        except Exception:
            traceback.print_exc()
