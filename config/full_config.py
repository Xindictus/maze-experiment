from pydantic import BaseModel

from config.experiment_base import ExperimentBaseConfig
from config.game_base import GameBaseConfig
from config.loader import discover_variants
from config.sac_base import SACBaseConfig
from config.gui_base import GUIBaseConfig
from utils.logger import Logger

logger = Logger().get_logger()


class FullConfig(BaseModel):
    game: GameBaseConfig
    gui: GUIBaseConfig
    experiment: ExperimentBaseConfig
    sac: SACBaseConfig


def build_config(
    game: str = "default",
    gui: str = "default",
    experiment: str = "default",
    sac: str = "default",
    overrides: dict | None = None,
) -> FullConfig:
    """
    Create and return a FullConfig instance by name + overrides.
    """
    # Dynamically discovered variants from single-variant modules
    GAME_VARIANTS = discover_variants("config.game_variants", GameBaseConfig)
    GUI_VARIANTS = discover_variants("config.gui_variants", GUIBaseConfig)
    EXPERIMENT_VARIANTS = discover_variants(
        "config.experiment_variants", ExperimentBaseConfig
    )
    SAC_VARIANTS = discover_variants("config.sac_variants", SACBaseConfig)

    logger.debug(f"[GAME-VARIANTS]: {GAME_VARIANTS}")
    logger.debug(f"[GUI-VARIANTS]: {GUI_VARIANTS}")
    logger.debug(f"[EXPERIMENT-VARIANTS]: {EXPERIMENT_VARIANTS}")
    logger.debug(f"[SAC-VARIANTS]: {SAC_VARIANTS}")

    game_config = GAME_VARIANTS[game]()
    gui_config = GUI_VARIANTS[gui]()
    exp_config = EXPERIMENT_VARIANTS[experiment]()
    sac_config = SAC_VARIANTS[sac]()

    config = FullConfig(
        game=game_config,
        gui=gui_config,
        experiment=exp_config,
        sac=sac_config,
    )

    return config.model_copy(update=overrides or {})
