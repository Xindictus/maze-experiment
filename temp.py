import sys
import time
from datetime import timedelta
from pathlib import Path

from prettytable import PrettyTable

from game.experiment.engine import ExperimentEngine
from maze3D_new.Maze3DEnvRemote import Maze3D
from rl_models.utils import get_sac_agent
from utils.logger import Logger

logger = Logger().get_logger()


"""
The code of this work is based on the following github repos:
https://github.com/kengz/SLM-Lab
https://github.com/EveLIn3/Discrete_SAC_LunarLander/blob/master/sac_discrete.py
"""


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


def run_experiment(args, config):
    # Initialize environment
    maze = Maze3D(config)
    loop = config["Experiment"]["mode"]

    if loop != "human":
        # TODO
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
    experiment = ExperimentEngine(
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
