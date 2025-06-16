from pydantic import BaseModel

from src.config.experiment_base import ExperimentBaseConfig
from src.config.game_base import GameBaseConfig
from src.config.loader import discover_variants
from src.config.sac_base import SACBaseConfig
from src.config.gui_base import GUIBaseConfig
from src.utils.logger import Logger

logger = Logger().get_logger()

# Dynamically discovered variants from single-variant modules
GAME_VARIANTS = discover_variants("src.config.game_variants", GameBaseConfig)
GUI_VARIANTS = discover_variants("src.config.gui_variants", GUIBaseConfig)
EXPERIMENT_VARIANTS = discover_variants(
    "src.config.experiment_variants", ExperimentBaseConfig
)
SAC_VARIANTS = discover_variants("src.config.sac_variants", SACBaseConfig)


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
    overrides: dict | None = {},
) -> FullConfig:
    """
    Create and return a FullConfig instance by name + overrides.
    """
    logger.debug(f"[GAME-VARIANTS]: {GAME_VARIANTS}")
    logger.debug(f"[GUI-VARIANTS]: {GUI_VARIANTS}")
    logger.debug(f"[EXPERIMENT-VARIANTS]: {EXPERIMENT_VARIANTS}")
    logger.debug(f"[SAC-VARIANTS]: {SAC_VARIANTS}")

    # Overrides
    overrides = overrides or {}
    game_over = overrides.get("game", {})
    gui_over = overrides.get("gui", {})
    exp_over = overrides.get("experiment", {})
    sac_over = overrides.get("sac", {})

    # Get appropriate config dataclasses
    game_cls = GAME_VARIANTS[game]
    gui_cls = GUI_VARIANTS[gui]
    experiment_cls = EXPERIMENT_VARIANTS[experiment]
    sac_cls = SAC_VARIANTS[sac]

    # Get config dumps
    game_base = game_cls().model_dump()
    gui_base = gui_cls().model_dump()
    experiment_base = experiment_cls().model_dump()
    sac_base = sac_cls().model_dump()

    # Merge base configs with overrides
    merged_game = {**game_base, **game_over}
    merged_gui = {**gui_base, **gui_over}
    merged_experiment = {**experiment_base, **exp_over}
    merged_sac = {**sac_base, **sac_over}

    game_config = game_cls.model_validate(merged_game)
    gui_config = gui_cls.model_validate(merged_gui)
    exp_config = experiment_cls.model_validate(merged_experiment)
    sac_config = sac_cls.model_validate(merged_sac)

    config = FullConfig(
        game=game_config,
        gui=gui_config,
        experiment=exp_config,
        sac=sac_config,
    )

    return config
