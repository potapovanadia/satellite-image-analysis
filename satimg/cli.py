"""Інтерфейс командного рядка.

Приклади:
    python -m satimg config
    python -m satimg config --set pipeline1.kuwahara.window=7
"""
import argparse

import yaml

from .config import ConfigError, apply_overrides, load_config


def build_parser():
    """Створює парсер аргументів з усіма підкомандами."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="шлях до YAML-конфігурації")
    common.add_argument("--set", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕННЯ",
                        help="перевизначити параметр, напр. pipeline1.kuwahara.window=7")
    parser = argparse.ArgumentParser(prog="satimg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config", parents=[common],
                   help="показати конфігурацію з урахуванням --set")
    return parser


def main(argv=None):
    """Розбирає аргументи та виконує обрану команду. Повертає код виходу."""
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_overrides(load_config(args.config), args.set)
    except ConfigError as exc:
        print(f"Помилка конфігурації: {exc}")
        return 2
    if args.command == "config":
        print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return 0
