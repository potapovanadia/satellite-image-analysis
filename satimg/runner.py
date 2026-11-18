"""Запуск пайплайнів на вибірці: шум -> обробка -> метрики -> збереження."""
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from . import hough, metrics
from .core import to_gray
from .io_utils import ImageLoadError, list_images, load_image, save_image
from .noise import apply_noise
from .pipelines import PIPELINES


def _class_of(path, root):
    """Клас зображення = назва першої підпапки відносно кореня вибірки."""
    rel = Path(path).relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "unknown"


def _clean(value):
    """Перетворює значення метрики на придатне для CSV/JSON (inf/nan -> рядок)."""
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return _clean(value.item())
    return value


def p1_metrics(clean_img, noisy_img, stages, ref_stages):
    """Метрики пайплайна №1 для одного запуску.

    s1_* — якість фільтрації Кувахари відносно ЧИСТОГО оригіналу;
    seg_* — стабільність сегментації: збіг з сегментацією чистого знімка.
    """
    filtered = stages[0].data
    labels, ref_labels = stages[3].data, ref_stages[3].data
    in_psnr = metrics.psnr(clean_img, noisy_img)
    s1_psnr = metrics.psnr(clean_img, filtered)
    return {
        "input_psnr": in_psnr,
        "input_ssim": metrics.ssim(clean_img, noisy_img),
        "s1_mse": metrics.mse(clean_img, filtered),
        "s1_psnr": s1_psnr,
        "s1_ssim": metrics.ssim(clean_img, filtered),
        "s1_psnr_gain": s1_psnr - in_psnr if math.isfinite(in_psnr) else float("nan"),
        "seg_miou_vs_clean": metrics.mean_iou(labels, ref_labels),
        "seg_acc_vs_clean": metrics.pixel_accuracy(labels, ref_labels),
        "n_segments": int(len(np.unique(labels))),
    }


def p2_metrics(clean_img, noisy_img, stages, ref_stages):
    """Метрики пайплайна №2 для одного запуску.

    s1_* — CLAHE зашумленого знімка проти CLAHE чистого (стабільність);
    clahe_contrast_gain — у скільки разів CLAHE збільшив контраст (std);
    edges_f1 / morph_f1 — збіг контурів із контурами чистого знімка;
    lines_recall — частка прямих чистого знімка, знайдених і на зашумленому.
    """
    gray_noisy = to_gray(noisy_img)
    enhanced, ref_enhanced = stages[0].data, ref_stages[0].data
    lines, ref_lines = stages[3].data, ref_stages[3].data
    matched, _ = hough.match_lines(lines, ref_lines, angle_tol=2.0, rho_tol=3.0)
    in_std = metrics.contrast_std(gray_noisy)
    return {
        "input_psnr": metrics.psnr(clean_img, noisy_img),
        "input_ssim": metrics.ssim(clean_img, noisy_img),
        "s1_mse": metrics.mse(ref_enhanced, enhanced),
        "s1_psnr": metrics.psnr(ref_enhanced, enhanced),
        "s1_ssim": metrics.ssim(ref_enhanced, enhanced),
        "clahe_contrast_gain": metrics.contrast_std(enhanced) / in_std if in_std else float("nan"),
        "clahe_psnr_vs_input": metrics.psnr(gray_noisy, enhanced),
        "edge_density": float(np.mean(stages[1].data)),
        "edges_f1_vs_clean": metrics.edge_f1(stages[1].data, ref_stages[1].data),
        "morph_f1_vs_clean": metrics.edge_f1(stages[2].data, ref_stages[2].data),
        "n_lines": len(lines),
        "n_lines_clean": len(ref_lines),
        "lines_recall_vs_clean": matched / len(ref_lines) if ref_lines else float("nan"),
    }


METRIC_FUNCS = {"p1": p1_metrics, "p2": p2_metrics}


def _save_stages(stages, folder, noisy_img):
    """Зберігає вхідне зображення та результат кожної стадії."""
    save_image(noisy_img, folder / "00_input.png")
    for stage in stages:
        save_image(stage.preview, folder / f"{stage.name}.png")


def write_table(rows, csv_path, json_path):
    """Записує список словників у CSV і JSON."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{k: _clean(v) for k, v in row.items()} for row in rows]
    if rows:
        fields = list(rows[0].keys())
        for row in rows[1:]:
            fields += [k for k in row if k not in fields]
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)


def select_per_class(paths, root, per_class):
    """Перші per_class зображень кожного класу (для швидких експериментів)."""
    chosen, counts = [], {}
    for path in paths:
        cls = _class_of(path, root)
        if counts.get(cls, 0) < per_class:
            counts[cls] = counts.get(cls, 0) + 1
            chosen.append(path)
    return chosen


def run_experiment(cfg, pipeline_names, input_dir=None, output_dir=None,  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
                   noise_names=None, limit=None, log=print, per_class=None):
    """Обробляє всю вибірку обраними пайплайнами за всіх умов шуму.

    Args:
        per_class: якщо задано — брати лише перші N зображень кожного класу.

    Для кожного зображення спочатку виконується прогін без шуму (еталон),
    потім — із кожним видом шуму. Метрики кожного запуску пишуться у
    results/metrics_<pipeline>.csv і .json, помилки читання — в errors.csv.

    Returns:
        Словник {назва пайплайна: список рядків метрик}.
    """
    root = Path(input_dir or cfg["data"]["input_dir"])
    out = Path(output_dir or cfg["output"]["dir"])
    seed = int(cfg.get("seed", 42))
    noises = [n for n in cfg["noise"] if noise_names is None or n["name"] in noise_names]
    if not any(n["type"] == "none" for n in noises):
        noises = [{"name": "none", "type": "none"}] + noises
    save_noise = set(cfg["output"].get("save_noise", []))
    paths = list_images(root)
    if per_class:
        paths = select_per_class(paths, root, per_class)
    if limit:
        paths = paths[:limit]
    if not paths:
        log(f"У папці {root} не знайдено зображень")

    all_rows = {name: [] for name in pipeline_names}
    errors = []
    for idx, path in enumerate(paths, 1):
        try:
            clean_img = load_image(path, min_size=int(cfg["data"].get("min_size", 3)))
        except ImageLoadError as exc:
            log(f"[пропущено] {exc}")
            errors.append({"image": str(path), "error": str(exc)})
            continue
        cls = _class_of(path, root)
        for name in pipeline_names:
            run = PIPELINES[name]
            params = cfg["pipeline1" if name == "p1" else "pipeline2"]
            reference = None
            for n_idx, noise in enumerate(noises):
                noisy = apply_noise(clean_img, noise, seed + idx * 100 + n_idx)
                start = time.perf_counter()
                stages = run(noisy, params, seed)
                elapsed = time.perf_counter() - start
                if noise["type"] == "none":
                    reference = stages
                row = {"pipeline": name, "image": path.stem, "class": cls,
                       "noise": noise["name"], "time_s": round(elapsed, 4)}
                row.update(METRIC_FUNCS[name](clean_img, noisy, stages, reference))
                all_rows[name].append(row)
                if cfg["output"].get("save_stages", True) and noise["name"] in save_noise:
                    _save_stages(stages, out / name / noise["name"] / cls / path.stem, noisy)
        log(f"[{idx}/{len(paths)}] {path.name} — готово")

    for name, rows in all_rows.items():
        write_table(rows, out / f"metrics_{name}.csv", out / f"metrics_{name}.json")
    if errors:
        write_table(errors, out / "errors.csv", out / "errors.json")
    return all_rows
