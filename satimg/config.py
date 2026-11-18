"""Завантаження конфігурації з YAML та перевизначення параметрів з командного рядка."""
import copy
from pathlib import Path

import yaml


class ConfigError(Exception):
    """Помилка у конфігураційному файлі або в перевизначенні параметра."""


def load_config(path):
    """Читає YAML-файл конфігурації та повертає словник."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Конфігураційний файл не знайдено: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError("Конфігурація має бути словником YAML")
    return data


def apply_overrides(config, overrides):
    """Застосовує перевизначення виду 'pipeline1.kuwahara.window=7'.

    Значення розбирається як YAML, тому '7' стане int, '0.5' — float,
    '[8, 8]' — списком. Повертає нову копію конфігурації.
    """
    result = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Очікується формат ключ=значення, отримано: {item}")
        key, raw_value = item.split("=", 1)
        parts = key.strip().split(".")
        node = result
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(f"Невідомий параметр: {key}")
            node = node[part]
        if parts[-1] not in node:
            raise ConfigError(f"Невідомий параметр: {key}")
        node[parts[-1]] = yaml.safe_load(raw_value)
    return result
