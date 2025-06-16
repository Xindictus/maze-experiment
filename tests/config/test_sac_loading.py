import pytest

from deepdiff import DeepDiff
from pydantic import ValidationError

from src.config.full_config import build_config, SAC_VARIANTS
from src.config.loader import flatten_overrides
from src.config.sac_base import SACBaseConfig


def test_build_default_config():
    config = build_config()
    print(config)

    assert hasattr(config, "game")
    assert config.game is not None

    assert hasattr(config, "gui")
    assert config.gui is not None

    assert hasattr(config, "experiment")
    assert config.experiment is not None

    assert hasattr(config, "sac")
    assert config.sac is not None


def test_sac_variant_agent_agent():
    config = build_config(sac="agent-agent")
    print(config)

    assert hasattr(config, "sac")
    assert config.sac is not None

    assert hasattr(config.sac, "freeze_agent")
    assert config.sac.freeze_agent is True


def test_sac_variant_agent_only():
    config = build_config(sac="agent-only")
    print(config)

    assert hasattr(config, "sac")
    assert config.sac is not None

    assert hasattr(config.sac, "batch_size")
    assert config.sac.batch_size == 64


def test_overrides_creation():
    overrides = ["sac.alpha=0.02"]

    overrides_dict = flatten_overrides(overrides)
    expected_dict = {
        "sac": {"alpha": 0.02},
    }

    diff = DeepDiff(overrides_dict, expected_dict, ignore_order=True)
    assert not diff, diff


def test_multiple_overrides_creation():
    overrides = ["sac.alpha=0.02", "gui.foo=foo"]

    overrides_dict = flatten_overrides(overrides)
    expected_dict = {"sac": {"alpha": 0.02}, "gui": {"foo": "foo"}}

    diff = DeepDiff(overrides_dict, expected_dict, ignore_order=True)
    assert not diff, diff


def test_sac_override():
    overrides = ["sac.alpha=0.02", "sac.batch_size=8"]
    override_dict = flatten_overrides(overrides)
    print(override_dict)

    config = build_config(overrides=override_dict)
    print(config)

    assert hasattr(config, "sac")
    assert config.sac is not None

    print(config.sac)
    assert isinstance(config.sac, SACBaseConfig)
    assert hasattr(config.sac, "alpha")
    assert hasattr(config.sac, "target_entropy_ratio")
    assert config.sac.alpha == 0.02
    assert config.sac.batch_size == 8


def test_sac_invalid_override_rejected():
    overrides = ["sac.alpha=-1"]
    override_dict = flatten_overrides(overrides)

    with pytest.raises(ValidationError):
        build_config(overrides=override_dict)


@pytest.mark.parametrize("sac_key", SAC_VARIANTS.keys())
def test_all_sac_variants_load(sac_key):
    config = build_config(sac=sac_key)
    assert config.sac is not None
