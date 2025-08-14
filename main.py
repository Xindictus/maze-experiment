import json
import random
import time
import traceback
from typing import Annotated, List, Literal

from cyclopts import App, Parameter

from src.config.full_config import FullConfig, build_config
from src.config.loader import flatten_overrides
from src.game.game_controller import GameController
from src.marl.algos.qmix import MAC, QmixTrainer
from src.marl.algos.qmix.runner import QmixRunner
from src.marl.buffers.episode_replay_buffer import EpisodeReplayBuffer

# from src.marl.buffers.standard_replay_buffer import StandardReplayBuffer
from src.marl.mixing.qmix import QMixer
from src.utils.logger import LOG_LEVELS, Logger

app = App()


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
):
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

    maze = GameController(config)

    # Init buffer
    # buffer = StandardReplayBuffer(mem_size=100)
    buffer = EpisodeReplayBuffer(mem_size=config.experiment.buffer_memory_size)

    # Dummy mixer and MAC
    mixer = QMixer(config.qmix, "MAIN")
    target_mixer = QMixer(config.qmix, "TARGET")

    mac = MAC(config=config.qmix)
    target_mac = MAC(config=config.qmix)

    trainer = QmixTrainer(
        buffer=buffer,
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

    runner.run()
    # my_test(config)


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
