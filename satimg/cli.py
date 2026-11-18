"""Інтерфейс командного рядка.

Приклади:
    python -m satimg config
    python -m satimg config --set pipeline1.kuwahara.window=7
    python -m satimg dataset
"""
import argparse

import yaml

from .config import ConfigError, apply_overrides, load_config
from .dataset import build_manifest


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
    p_data = sub.add_parser("dataset", parents=[common],
                            help="перевірити вибірку і створити data/manifest.csv")
    p_data.add_argument("--input", help="папка з вибіркою (за замовчуванням з конфігурації)")
    p_data.add_argument("--manifest", default="data/manifest.csv", help="куди записати опис")
    return parser


def cmd_dataset(args, cfg):
    """Команда dataset: перевірка всіх файлів вибірки та опис у CSV."""
    root = args.input or cfg["data"]["input_dir"]
    ok, per_class, bad = build_manifest(root, args.manifest)
    for cls, count in sorted(per_class.items()):
        print(f"  {cls}: {count}")
    print(f"Коректних зображень: {ok}, пошкоджених: {bad}. Опис: {args.manifest}")
    return 0 if ok else 1


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
    if args.command == "dataset":
        return cmd_dataset(args, cfg)
    return 0
