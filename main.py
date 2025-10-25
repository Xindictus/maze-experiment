import json
import random
import signal
import time
import traceback
from typing import Annotated, List, Literal

from cyclopts import App, Parameter

from src.config.full_config import FullConfig, build_config
from src.config.loader import flatten_overrides
from src.game.game_controller import GameController
from src.marl.algos.qmix import MAC, QmixTrainer
from src.marl.algos.qmix.runner import QmixRunner
from src.marl.buffers import (
    EpisodeReplayBuffer,
    StandardReplayBuffer,
)
from src.marl.mixing.qmix import QMixer
from src.marl.rewards.reward_engines import get_reward_engine
from src.utils.logger import LOG_LEVELS, Logger
from src.utils.sigint import sigint_controller

app = App()


def _signal_handler(signum, frame):
    Logger().warning("SIGINT received, shutting down...")
    sigint_controller.request()


@app.default
def run(
    log: Literal[
        "critical",
        "fatal",
        "error",
        "warning",
        "warn",
        "info",
        "debug",
        "notset",
    ] = "info",
    game: Annotated[
        str, Parameter(name=["--game"], help="Game variant")
    ] = "default",
    gui: Annotated[
        str, Parameter(name=["--gui"], help="Gui variant")
    ] = "default",
    experiment: Annotated[
        str, Parameter(name=["--experiment"], help="Experiment variant")
    ] = "default",
    sac: Annotated[
        str, Parameter(name=["--sac"], help="SAC variant")
    ] = "default",
    qmix: Annotated[
        str, Parameter(name=["--qmix"], help="QMIX variant")
    ] = "default",
    overrides: Annotated[
        List[str],
        Parameter(
            name=["--overrides", "-o"],
            help="Field overrides like sac.alpha=0.02",
            consume_multiple=True,
        ),
    ] = [],
) -> int:
    log_lvl = LOG_LEVELS[log]

    # Parse overrides from list[str] ~> nested dict
    override_dict = flatten_overrides(overrides)
    Logger(log_lvl).debug(f"[OVERRIDES]: {override_dict}")

    config = build_config(
        game=game,
        gui=gui,
        experiment=experiment,
        qmix=qmix,
        sac=sac,
        overrides=override_dict,
    )

    full_config: FullConfig = json.dumps(
        config.model_dump(mode="json"), indent=2
    )
    Logger().info(f"[FULL-CONFIG]: {full_config}")

    reward_engine = get_reward_engine(
        name=config.experiment.reward_engine,
        config=config.experiment,
    )

    maze = GameController(config=config, reward_engine=reward_engine)

    mem_size = config.experiment.buffer_memory_size
    buffer_type = config.experiment.buffer_type

    # Init buffer
    if buffer_type == "episode":
        buffer = EpisodeReplayBuffer(mem_size=mem_size)
    elif buffer_type == "standard":
        buffer = StandardReplayBuffer(mem_size=mem_size)
    else:
        raise ValueError("Unknown buffer type")

    # Init mixers
    mixer = QMixer(
        config=config.qmix,
        name="MAIN",
        buffer_type=config.experiment.buffer_type,
    )
    target_mixer = QMixer(
        config=config.qmix,
        name="TARGET",
        buffer_type=config.experiment.buffer_type,
    )
    target_mixer.load_state_dict(mixer.state_dict())

    mac = MAC(config=config.qmix)
    target_mac = MAC(config=config.qmix)

    # Make sure agent networks are init
    # the same way in target MAC
    target_mac.load_state(other=mac)

    trainer = QmixTrainer(
        buffer=buffer,
        buffer_type=buffer_type,
        mac=mac,
        mixer=mixer,
        target_mac=target_mac,
        target_mixer=target_mixer,
        config=config.qmix,
    )

    runner = QmixRunner(
        config=config,
        game_controller=maze,
        mac=mac,
        trainer=trainer,
        replay_buffer=buffer,
    )

    signal.signal(signal.SIGINT, _signal_handler)

    runner.run()

    return 0


def my_test(config):
    """Dummy execution"""
    while True:
        try:
            maze = GameController(config)

            while True:
                maze.step(
                    [random.randint(-1, 1), random.randint(-1, 1)],
                    False,
                    0.2,
                    "test",
                    "",
                )

        except Exception:
            traceback.print_exc()
            time.sleep(3)


if __name__ == "__main__":
    app()
