import json
from typing import Annotated, List

from cyclopts import App, Parameter

from src.config.full_config import build_config
from src.config.loader import flatten_overrides
from src.utils.logger import Logger

logger = Logger().get_logger()
app = App()


@app.default
def run(
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
    # Parse overrides from list[str] ~> nested dict
    override_dict = flatten_overrides(overrides)
    logger.debug(f"[OVERRIDES]: {override_dict}")

    config = build_config(
        game=game,
        gui=gui,
        experiment=experiment,
        sac=sac,
        overrides=override_dict,
    )

    full_config = json.dumps(config.model_dump(mode="json"), indent=2)
    logger.info(f"[FULL-CONFIG]: {full_config}")


if __name__ == "__main__":
    app()
