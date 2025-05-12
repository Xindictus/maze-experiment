import argparse

from game.utils import get_config
from temp import run_experiment


def parse_args():
    """
    Parses command-line arguments for training a SAC agent in the Maze3D
    environment.

    Returns:
        argparse.Namespace: Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a SAC agent in the Maze3D environment"
    )

    # Core training options
    parser.add_argument(
        "--alg",
        type=str,
        default="",
        required=True,
        help="The algorithm to be used for the training",
    )
    parser.add_argument(
        "--alg-config",
        type=str,
        default="",
        required=True,
        help="YAML config based on algorithm selected",
    )
    parser.add_argument(
        "--participant",
        type=str,
        default="test",
        help="Participant identifier for logs and results",
    )

    # SAC hyperparameters
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Initial entropy coefficient"
    )
    parser.add_argument(
        "--alpha-lr",
        type=float,
        default=0.001,
        help="Learning rate for entropy coefficient",
    )
    parser.add_argument(
        "--auto-alpha",
        action="store_true",
        help="Enable automatic entropy tuning",
    )

    # Agent setup
    parser.add_argument(
        "--agent-type",
        type=str,
        default="basesac",
        help="Type of agent to use",
    )
    parser.add_argument(
        "--num-actions", type=int, default=3, help="Number of discrete actions"
    )

    # Buffer and pretraining
    parser.add_argument(
        "--buffer-size", type=int, default=3500, help="Replay buffer size"
    )
    parser.add_argument(
        "--buffer-path-1",
        type=str,
        default="game/saved_buffers/buffer_cris.npy",
        help="Expert buffer 1 path",
    )
    parser.add_argument(
        "--buffer-path-2",
        type=str,
        default="game/saved_buffers/buffer_koutris.npy",
        help="Expert buffer 2 path",
    )
    parser.add_argument(
        "--buffer-path-3",
        type=str,
        default=None,
        help="Optional third buffer path",
    )
    parser.add_argument(
        "--load-buffer", action="store_true", help="Load buffer from file"
    )
    parser.add_argument(
        "--load-expert-buffers",
        action="store_true",
        help="Load expert demonstration buffers",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args.alg, args.alg_config)
    run_experiment(args, config)


if __name__ == "__main__":
    main()
