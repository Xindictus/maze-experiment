# Virtual environment
import argparse
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from prettytable import PrettyTable

# Experiment
from game.experiment import Experiment
from game.utils import get_config
from maze3D_new.Maze3DEnvRemote import Maze3D as Maze3D_v2

# RL modules
from rl_models.utils import get_sac_agent
from utils.logger import Logger

# from maze3D_new.assets import *
# from maze3D_new.utils import save_logs_and_plot


logger = Logger().get_logger()


"""
The code of this work is based on the following github repos:
https://github.com/kengz/SLM-Lab
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--participant", type=str, default="test")
    parser.add_argument("--seed", type=int, default=4213)
    parser.add_argument("--scale-obs", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=3500)
    parser.add_argument("--actor-lr", type=float, default=0.0003)
    parser.add_argument("--critic-lr", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--auto-alpha", action="store_true", default=False)
    parser.add_argument("--alpha-lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=[32, 32])
    parser.add_argument(
        "--hidden-sizes", type=int, nargs="*", default=[32, 32]
    )
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--num-actions", type=int, default=3)
    parser.add_argument("--agent-type", type=str, default="basesac")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--avg-q", action="store_true", default=False)
    parser.add_argument("--clip-q", action="store_true", default=False)
    parser.add_argument("--clip-q-epsilon", type=float, default=0.5)
    parser.add_argument(
        "--entropy-penalty", action="store_true", default=False
    )

    parser.add_argument("--entropy-penalty-beta", type=float, default=0.5)

    parser.add_argument(
        "--buffer-path-1",
        type=str,
        default="game/Saved_Buffers/Buffer_Cris.npy",
    )
    parser.add_argument(
        "--buffer-path-2",
        type=str,
        default="game/Saved_Buffers/Buffer_Koutris.npy",
    )
    parser.add_argument("--buffer-path-3", type=str, default=None)

    parser.add_argument(
        "--Load-Expert-Buffers", action="store_true", default=False
    )

    parser.add_argument("--load-buffer", action="store_true", default=False)

    return parser.parse_args()


def print_setting(agent, x):
    (
        ID,
        actor_lr,
        critic_lr,
        alpha_lr,
        hidden_size,
        tau,
        gamma,
        batch_size,
        target_entropy,
        log_alpha,
        freeze_status,
    ) = agent.return_settings()
    x.field_names = [
        "Agent ID",
        "Actor LR",
        "Critic LR",
        "Alpha LR",
        "Hidden Size",
        "Tau",
        "Gamma",
        "Batch Size",
        "Target Entropy",
        "Log Alpha",
        "Freeze Status",
    ]
    x.add_row(
        [
            ID,
            actor_lr,
            critic_lr,
            alpha_lr,
            hidden_size,
            tau,
            gamma,
            batch_size,
            target_entropy,
            log_alpha,
            freeze_status,
        ]
    )
    return x


def check_save_dir(
    checkpoint_dir: str = None, participant_nm: str = None
) -> str:
    # Create checkpoint directory if not exists
    base_dir = Path(checkpoint_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create unique subdirectory for participant
    candidate = base_dir / participant_nm
    counter = 0

    while candidate.exists():
        if counter > 99:
            logger.error(
                "Too many checkpoint directories for participant {nm}".format(
                    nm=participant_nm
                )
            )
            sys.exit(1)

        candidate = base_dir / f"{participant_nm}{counter:02d}"
        counter += 1

    candidate.mkdir()

    # Preserve trailing slash if needed
    return f"{str(candidate)}/"


def check_save_dir2(checkpoint_dir: str = None, participant_name: str = None):
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    if not os.path.isdir(os.path.join(checkpoint_dir, participant_name)):
        os.mkdir(os.path.join(checkpoint_dir, participant_name))
        checkpoint_dir = os.path.join(checkpoint_dir, participant_name) + "/"
    else:
        c = 1
        while os.path.isdir(
            os.path.join(checkpoint_dir, participant_name + str(c))
        ):
            c += 1
        os.mkdir(os.path.join(checkpoint_dir, participant_name + str(c)))
        checkpoint_dir = (
            os.path.join(checkpoint_dir, participant_name + str(c)) + "/"
        )

    config["SAC"]["chkpt"] = checkpoint_dir
    return config


def main(argv):
    args = get_args()
    # get configuration
    print("IM trying to get this config")
    print(args.config)

    config = get_config(args.config)

    print("Config loaded", config)

    # creating environment
    maze = Maze3D_v2(config_file=args.config)
    loop = config["Experiment"]["mode"]
    if loop != "human":
        config = check_save_dir(config, args.participant)
        print_array = PrettyTable()
        if args.agent_type == "basesac":
            agent = get_sac_agent(
                args, config, maze, p_name=args.participant, ID="First"
            )
            agent.save_models("initial")
        print_array = print_setting(agent, print_array)

        if loop == "no_tl_two_agents":
            if args.agent_type == "basesac":
                second_agent = get_sac_agent(
                    args, config, maze, p_name=args.participant, ID="Second"
                )
                second_agent.save_models("initial")
            print_array = print_setting(second_agent, print_array)
        else:
            second_agent = None

        print("Agent created")
        print(print_array)
        # create the experiment
    else:
        agent = None
        second_agent = None
    experiment = Experiment(
        maze,
        agent,
        config=config,
        participant_name=args.participant,
        second_agent=second_agent,
    )

    start_experiment = time.time()

    # Run a Pre-Training with Expert Buffers
    print("Load Expert Buffers:", args.load_expert_buffers)
    if args.load_expert_buffers:
        experiment.test_buffer(2500)
    elif args.load_buffer:
        experiment.test_buffer(2500)

    if loop == "no_tl":
        experiment.mz_experiment(args.participant)
    elif loop == "no_tl_only_agent":
        experiment.mz_only_agent(args.participant)
    elif loop == "no_tl_two_agents":
        experiment.mz_two_agents(args.participant)
    elif loop == "eval":
        experiment.mz_eval(args.participant)
    elif loop == "human":
        experiment.human_play(args.participant)
    else:
        print("Unknown training mode")
        exit(1)
    # experiment.env.finished()
    end_experiment = time.time()
    experiment_duration = timedelta(
        seconds=end_experiment
        - start_experiment
        - experiment.duration_pause_total
    )

    print("Total Experiment time: {}".format(experiment_duration))

    return


if __name__ == "__main__":
    main(sys.argv[1:])
    exit(0)
