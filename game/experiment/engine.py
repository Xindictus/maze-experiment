import os

from rl_models.networks_discrete import ReplayBuffer

from .agent import get_agent_action, save_experience
from .game_loop import run_game_block
from .trainer import offline_grad_updates_session
from .utils import normalize_state, save_pickle


class ExperimentEngine:
    def __init__(
        self,
        args,
        environment,
        agent=None,
        config=None,
        participant_name=None,
        second_agent=None,
    ):
        self.args = args
        self.config = config
        self.env = environment
        self.agent = agent
        self.second_agent = second_agent
        self.participant_name = participant_name

        self.mode = config["Experiment"]["mode"]
        self.goal = config["game"]["goal"]
        self.second_human = config["game"].get("second_human", False)
        self.games_per_block = config["Experiment"][self.mode][
            "games_per_block"
        ]
        self.max_blocks = config["Experiment"][self.mode]["max_blocks"]
        self.max_game_duration = config["Experiment"][self.mode][
            "max_duration"
        ]
        self.action_duration = config["Experiment"][self.mode][
            "action_duration"
        ]
        self.popup_window_time = config["GUI"]["popup_window_time"]
        self.log_interval = config["Experiment"][self.mode]["log_interval"]
        self.is_discrete = config["SAC"].get("discrete", True)

        self.path_to_save = f"results/{self.mode}/{participant_name}"
        self.best_game_score = 0
        self.last_score = 0
        self.duration_pause_total = 0
        self.current_block = 0

        if self.mode != "human":
            if config["SAC"].get("load_checkpoint"):
                self.agent.load_models()
            if config["SAC"].get("load_second_agent"):
                self.second_agent.load_models()
            if args.ppr:
                self.second_agent.load_models()
                self.ppr_table = [0.7, 0.55, 0.4, 0.25, 0.1, 0.05, 0.01]
            else:
                self.ppr_table = None

    def mz_experiment(self):
        train_data = self._run_blocks("train")
        test_data = (
            self._run_blocks("test")
            if self.config["Experiment"][self.mode]["test_block"]
            else {}
        )

        if self.config["Experiment"][self.mode].get("extra_test_block"):
            extra_test = self._run_blocks(
                "test", start=self.max_blocks, num_blocks=1
            )
            test_data.update(extra_test)

        save_pickle(self.participant_name, self.mode, "train", train_data)
        save_pickle(self.participant_name, self.mode, "test", test_data)
        self._save_buffer(self.agent.memory)

    def human_play(self):
        self.human_replay_buffer = ReplayBuffer(self.args)
        data = self._run_blocks("train", mode_override="human", human=True)
        save_pickle(self.participant_name, self.mode, "train", data)
        self._save_buffer(self.human_replay_buffer)

    def _run_blocks(
        self, mode, start=0, num_blocks=None, mode_override=None, human=False
    ):
        all_metrics = {}
        num = num_blocks if num_blocks is not None else self.max_blocks

        for i_block in range(start, start + num):
            block_name = f"block_{i_block}"
            print(f"{mode.capitalize()} Block: {i_block}")

            def log_fn(i, total):
                return f"{i + 1} / {total}"
                # f" -- Last Score: {self.last_score}"
                # f" -- Best Score: {self.best_game_score}"

            all_metrics[block_name] = run_game_block(
                env=self.env,
                agent_callback=self if not human else self._dummy_agent(),
                step_callback=self if not human else self._dummy_agent(),
                block_name=block_name,
                block_number=i_block,
                max_games=self.games_per_block,
                max_game_duration=self.max_game_duration,
                action_duration=self.action_duration,
                popup_window_time=self.popup_window_time,
                games_per_block=self.games_per_block,
                mode=mode_override or self.mode,
                log_action=log_fn,
                can_learn=True,
            )
        return all_metrics

    def get_action(self, norm_obs, mode):
        return get_agent_action(
            norm_obs,
            agent=self.agent,
            second_agent=self.second_agent,
            second_human=self.second_human,
            is_discrete=self.is_discrete,
            enemy_status="agent",
            game_mode=mode,
            block=self.current_block,
            use_ppr=self.args.ppr,
            ppr_table=self.ppr_table,
        )

    def normalize_state(self, observation):
        return normalize_state(observation)

    def after_game(self, score):
        self.last_score = score
        self.best_game_score = max(score, self.best_game_score)

    def should_learn(self, mode, block_number, i_game):
        return (
            mode == "train"
            and hasattr(self.agent, "can_learn")
            and self.agent.can_learn(block_number + 1)
        )

    def learn(self, i_game, block_number, metric_dict):
        print("Start Offline Gradient Updates Session")
        offline_grad_updates_session(
            agent=self.agent,
            second_agent=self.second_agent,
            config=self.config,
            mode=self.mode,
            block_number=block_number,
        )
        metric_dict[f"update_history_{i_game}"] = self.agent.collect_data()

    def save_experience(
        self,
        norm_prev_obs,
        norm_obs,
        action,
        reward,
        done,
        block_number,
        i_game,
        step_counter,
    ):
        interaction = [
            norm_prev_obs,
            [action],
            reward,
            norm_obs,
            done,
            [block_number, i_game, step_counter],
        ]
        save_experience(
            interaction,
            self.agent,
            self.second_agent,
            self.config,
            self.mode,
            self.second_human,
        )

    def _save_buffer(self, memory):
        buffer_dir = os.path.join(
            "results", self.mode, "buffer", self.participant_name
        )
        os.makedirs(buffer_dir, exist_ok=True)
        fname = "buffer.npy"
        c = 1
        while os.path.isfile(os.path.join(buffer_dir, fname)):
            fname = f"buffer_{c}.npy"
            c += 1
        memory.save_buffer(buffer_dir, fname)

    def _dummy_agent(self):
        class Dummy:
            def get_action(self, *_):
                return None, None

            def normalize_state(self, obs):
                return obs

            def save_experience(self, *_):
                pass

            def after_game(self, *_):
                pass

            def should_learn(self, *_):
                return False

            def learn(self, *_):
                pass

        return Dummy()
