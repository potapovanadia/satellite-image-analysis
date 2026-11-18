"""Тести завантаження конфігурації та перевизначення параметрів."""
import pytest

from satimg.config import ConfigError, apply_overrides, load_config


def test_load_project_config():
    cfg = load_config("config.yaml")
    assert cfg["seed"] == 42
    assert "pipeline1" in cfg and "pipeline2" in cfg


def test_override_parses_types():
    cfg = {"a": {"b": 1, "c": [1, 1]}}
    new = apply_overrides(cfg, ["a.b=7", "a.c=[8, 8]"])
    assert new["a"]["b"] == 7
    assert new["a"]["c"] == [8, 8]
    assert cfg["a"]["b"] == 1  # оригінал не змінюється


def test_override_unknown_key():
    with pytest.raises(ConfigError):
        apply_overrides({"a": {"b": 1}}, ["a.x=2"])


def test_override_bad_format():
    with pytest.raises(ConfigError):
        apply_overrides({"a": 1}, ["a"])


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
