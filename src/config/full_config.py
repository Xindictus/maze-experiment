import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from src.config.experiment_base import ExperimentBaseConfig
from src.config.game_base import GameBaseConfig
from src.config.gui_base import GUIBaseConfig
from src.config.loader import discover_variants
from src.config.network_config import NetworkConfig
from src.config.qmix_base import QmixBaseConfig
from src.config.sac_base import SACBaseConfig
from src.utils.logger import Logger

# Dynamically discovered variants from single-variant modules
GAME_VARIANTS = discover_variants("src.config.game_variants", GameBaseConfig)
GUI_VARIANTS = discover_variants("src.config.gui_variants", GUIBaseConfig)
EXPERIMENT_VARIANTS = discover_variants(
    "src.config.experiment_variants", ExperimentBaseConfig
)
SAC_VARIANTS = discover_variants("src.config.sac_variants", SACBaseConfig)


class FullConfig(BaseModel):
    experiment: ExperimentBaseConfig
    game: GameBaseConfig
    gui: GUIBaseConfig
    network: NetworkConfig
    qmix: QmixBaseConfig
    sac: SACBaseConfig
    date: str
    out_dir: str


def save_json_config(cfg: FullConfig, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(
            cfg.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def build_config(
    game: str = "default",
    gui: str = "default",
    experiment: str = "default",
    qmix: str = "default",
    sac: str = "default",
    overrides: dict | None = {},
) -> FullConfig:
    """
    Create and return a FullConfig instance by name + overrides.
    """
    Logger().debug(f"[GAME-VARIANTS]: {GAME_VARIANTS}")
    Logger().debug(f"[GUI-VARIANTS]: {GUI_VARIANTS}")
    Logger().debug(f"[EXPERIMENT-VARIANTS]: {EXPERIMENT_VARIANTS}")
    Logger().debug(f"[SAC-VARIANTS]: {SAC_VARIANTS}")

    # Overrides
    overrides = overrides or {}
    exp_over = overrides.get("experiment", {})
    game_over = overrides.get("game", {})
    gui_over = overrides.get("gui", {})
    net_over = overrides.get("network", {})
    sac_over = overrides.get("sac", {})
    qmix_over = overrides.get("qmix", {})

    # Get appropriate config dataclasses
    experiment_cls = EXPERIMENT_VARIANTS[experiment]
    game_cls = GAME_VARIANTS[game]
    gui_cls = GUI_VARIANTS[gui]
    sac_cls = SAC_VARIANTS[sac]

    # Get config dumps
    experiment_base = experiment_cls().model_dump()
    game_base = game_cls().model_dump()
    gui_base = gui_cls().model_dump()
    sac_base = sac_cls().model_dump()

    # Merge base configs with overrides
    merged_experiment = {**experiment_base, **exp_over}
    merged_game = {**game_base, **game_over}
    merged_gui = {**gui_base, **gui_over}
    merged_sac = {**sac_base, **sac_over}

    exp_config = experiment_cls.model_validate(merged_experiment)
    game_config = game_cls.model_validate(merged_game)
    gui_config = gui_cls.model_validate(merged_gui)
    net_config = NetworkConfig(**net_over)
    sac_config = sac_cls.model_validate(merged_sac)
    qmix_config = QmixBaseConfig(**qmix_over)

    today: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir: str = f"out/{today}"
    config_out: str = f"{out_dir}/config.json"

    # Creates the folder to save all experiment outputs
    Path(out_dir).mkdir(exist_ok=True)

    config = FullConfig(
        experiment=exp_config,
        game=game_config,
        gui=gui_config,
        network=net_config,
        sac=sac_config,
        qmix=qmix_config,
        date=today,
        out_dir=out_dir,
    )

    # Save current experiment config
    save_json_config(config, config_out)

    return config
