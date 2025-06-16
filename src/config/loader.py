import importlib
import inspect
import re
import typer

from typing import Type

from src.utils.logger import Logger

logger = Logger().get_logger()


def discover_variants(module_name: str, base_class: Type) -> dict[str, Type]:
    """
    Dynamically discover subclasses of a base class within a given module.
    Returns a map from lower_snake_case class name to the subclass type.
    """
    module = importlib.import_module(module_name)
    variants = {}

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, base_class) and obj is not base_class:
            key = (
                re.sub(r"(?<!^)([A-Z])", r"-\1", name)
                .removesuffix("-Config")
                .lower()
                .replace(base_class.__name__.lower(), "")
                or "base"
            )
            variants[key] = obj

    return variants


def flatten_overrides(pairs: list[str]) -> dict:
    """
    Turns list like ["sac.alpha=0.02", "experiment.mode=eval"] into nested dict
    """
    result = {}

    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"Invalid override format: '{pair}'")
        key, val = pair.split("=", 1)
        keys = key.split(".")
        d = result

        for part in keys[:-1]:
            d = d.setdefault(part, {})

        try:
            # Try to coerce value
            val = eval(val)
        except Exception:
            pass
        d[keys[-1]] = val

    return result
