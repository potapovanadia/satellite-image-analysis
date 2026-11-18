"""Інтерфейс командного рядка проєкту.

Приклади:
    python -m satimg config                         # показати конфігурацію
    python -m satimg dataset                        # перевірити вибірку
    python -m satimg run --pipeline p1              # пайплайн №1 на всій вибірці
    python -m satimg run --pipeline all --limit 5   # обидва пайплайни, 5 знімків
    python -m satimg method canny --image data/dataset/freeway/freeway00.tif
    python -m satimg verify                         # еталонні сценарії (C1)
    python -m satimg analyze                        # таблиці та графіки порівняння
    python -m satimg sweep --pipeline p1 --param pipeline1.kuwahara.window --values 3 5 7
Будь-який параметр змінюється без правки коду:
    python -m satimg run --pipeline p1 --set pipeline1.kuwahara.window=5
"""
import argparse
from pathlib import Path

import yaml

from .config import ConfigError, apply_overrides, load_config
from .dataset import build_manifest
from .io_utils import ImageLoadError, load_image, save_image
from .noise import apply_noise
from .pipelines import METHODS, PIPELINES

COMMANDS = {}


def command(name):
    """Декоратор: реєструє функцію як обробник підкоманди."""
    def wrap(func):
        COMMANDS[name] = func
        return func
    return wrap


def build_parser():
    """Створює парсер аргументів з усіма підкомандами."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="шлях до YAML-конфігурації")
    common.add_argument("--set", action="append", default=[], metavar="КЛЮЧ=ЗНАЧЕННЯ",
                        help="перевизначити параметр, напр. pipeline1.kuwahara.window=7")
    parser = argparse.ArgumentParser(prog="satimg", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", parents=[common], help="показати конфігурацію з урахуванням --set")

    p_data = sub.add_parser("dataset", parents=[common],
                            help="перевірити вибірку і створити data/manifest.csv")
    p_data.add_argument("--input", help="папка з вибіркою (за замовчуванням з конфігурації)")
    p_data.add_argument("--manifest", default="data/manifest.csv", help="куди записати опис")

    p_run = sub.add_parser("run", parents=[common], help="запустити пайплайн(и) на вибірці")
    p_run.add_argument("--pipeline", choices=["p1", "p2", "all"], default="all")
    p_run.add_argument("--input", help="папка з вибіркою")
    p_run.add_argument("--output", help="папка для результатів")
    p_run.add_argument("--noise", nargs="+", metavar="НАЗВА",
                       help="лише вказані умови шуму (назви з конфігурації)")
    p_run.add_argument("--limit", type=int, help="обробити лише перші N зображень")
    p_run.add_argument("--per-class", type=int, help="лише перші N зображень кожного класу")

    p_method = sub.add_parser("method", parents=[common],
                              help="застосувати один метод до одного зображення")
    p_method.add_argument("name", choices=sorted(METHODS))
    p_method.add_argument("--image", required=True, help="вхідне зображення")
    p_method.add_argument("--output", help="куди зберегти результат (PNG)")
    p_method.add_argument("--noise", help="назва умови шуму з конфігурації")

    p_verify = sub.add_parser("verify", parents=[common],
                              help="перевірка коректності за еталонними сценаріями (C1)")
    p_verify.add_argument("--output", help="папка для результатів")

    p_an = sub.add_parser("analyze", parents=[common],
                          help="зведені таблиці та графіки за результатами run")
    p_an.add_argument("--results", help="папка з metrics_*.csv")

    p_sw = sub.add_parser("sweep", parents=[common],
                          help="підбір параметра: порівняти кілька значень")
    p_sw.add_argument("--param", required=True, help="напр. pipeline1.kuwahara.window")
    p_sw.add_argument("--values", nargs="+", required=True, help="значення для перевірки")
    p_sw.add_argument("--pipeline", choices=["p1", "p2"], required=True)
    p_sw.add_argument("--noise", nargs="+", default=["none", "gaussian_20"],
                      help="умови шуму (за замовчуванням none і gaussian_20)")
    p_sw.add_argument("--per-class", type=int, default=4, help="знімків кожного класу")
    p_sw.add_argument("--output", help="папка для результатів")
    return parser


@command("config")
def cmd_config(args, cfg):  # pylint: disable=unused-argument
    """Друкує конфігурацію з урахуванням перевизначень."""
    print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    return 0


@command("dataset")
def cmd_dataset(args, cfg):
    """Перевіряє всі файли вибірки та записує опис у CSV."""
    root = args.input or cfg["data"]["input_dir"]
    ok, per_class, bad = build_manifest(root, args.manifest)
    for cls, count in sorted(per_class.items()):
        print(f"  {cls}: {count}")
    print(f"Коректних зображень: {ok}, пошкоджених: {bad}. Опис: {args.manifest}")
    return 0 if ok else 1


@command("run")
def cmd_run(args, cfg):
    """Запускає обрані пайплайни на всій вибірці з усіма умовами шуму."""
    from .runner import run_experiment  # pylint: disable=import-outside-toplevel
    names = list(PIPELINES) if args.pipeline == "all" else [args.pipeline]
    known = {n["name"] for n in cfg["noise"]}
    for name in args.noise or []:
        if name not in known:
            print(f"Невідома умова шуму: {name}. Доступні: {', '.join(sorted(known))}")
            return 2
    rows = run_experiment(cfg, names, args.input, args.output, args.noise, args.limit,
                          per_class=args.per_class)
    out = Path(args.output or cfg["output"]["dir"])
    for name, table in rows.items():
        print(f"{name}: {len(table)} запусків, метрики -> {out / f'metrics_{name}.csv'}")
    return 0 if any(rows.values()) else 1


@command("method")
def cmd_method(args, cfg):
    """Застосовує один метод до одного зображення і зберігає результат."""
    try:
        img = load_image(args.image)
    except ImageLoadError as exc:
        print(f"Помилка: {exc}")
        return 1
    if args.noise:
        spec = next((n for n in cfg["noise"] if n["name"] == args.noise), None)
        if spec is None:
            print(f"Невідома умова шуму: {args.noise}")
            return 2
        img = apply_noise(img, spec, int(cfg["seed"]))
    result = METHODS[args.name](img, cfg, int(cfg["seed"]))
    out = Path(args.output or Path(cfg["output"]["dir"]) / "methods" /
               f"{Path(args.image).stem}_{args.name}.png")
    save_image(result, out)
    print(f"Збережено: {out}")
    return 0


@command("verify")
def cmd_verify(args, cfg):
    """Виконує еталонні сценарії перевірки коректності."""
    from .verify import run_verification  # pylint: disable=import-outside-toplevel
    _, share = run_verification(cfg, args.output)
    return 0 if share > 0 else 1


@command("analyze")
def cmd_analyze(args, cfg):
    """Будує зведені таблиці та графіки з метрик попереднього запуску."""
    from .analysis import run_analysis  # pylint: disable=import-outside-toplevel
    return 0 if run_analysis(args.results or cfg["output"]["dir"]) else 1


@command("sweep")
def cmd_sweep(args, cfg):
    """Порівнює кілька значень одного параметра на частині вибірки."""
    from .analysis import run_sweep  # pylint: disable=import-outside-toplevel
    try:
        apply_overrides(cfg, [f"{args.param}={args.values[0]}"])
    except ConfigError as exc:
        print(f"Помилка: {exc}")
        return 2
    run_sweep(cfg, args.param, args.values, args.pipeline, args.noise, args.per_class,
              args.output or cfg["output"]["dir"])
    return 0


def main(argv=None):
    """Розбирає аргументи та виконує обрану команду. Повертає код виходу."""
    args = build_parser().parse_args(argv)
    try:
        cfg = apply_overrides(load_config(args.config), args.set)
    except ConfigError as exc:
        print(f"Помилка конфігурації: {exc}")
        return 2
    return COMMANDS[args.command](args, cfg)
