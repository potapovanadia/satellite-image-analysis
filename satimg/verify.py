"""Перевірка коректності за еталонними сценаріями (критерій C1).

Кожен тест-кейс — це конкретне зображення + сценарій + порогові значення.
Результат записується у results/verify_results.csv; підсумок — частка
тест-кейсів, що пройшли перевірку.
"""
import math
from pathlib import Path

from . import hough, metrics, synthetic
from .core import to_gray
from .histogram import clahe
from .io_utils import list_images, load_image
from .noise import apply_noise
from .pipelines import run_pipeline1, run_pipeline2
from .runner import write_table

ROAD_SCENES = [
    ((80.0, 30.0), (-40.0, -60.0)),
    ((128.0, 0.0), (-100.0, -90.0)),
    ((180.0, 45.0), (150.0, 10.0), (60.0, -30.0)),
]


def _real_images(cfg):
    """Перші N знімків кожного класу вибірки (N = verify.real_images_per_class)."""
    per_class = int(cfg["verify"]["real_images_per_class"])
    root = Path(cfg["data"]["input_dir"])
    chosen, counts = [], {}
    for path in list_images(root):
        cls = path.relative_to(root).parts[0]
        if counts.get(cls, 0) < per_class:
            counts[cls] = counts.get(cls, 0) + 1
            chosen.append(path)
    return chosen


def _case(scenario, case, passed, **values):
    row = {"scenario": scenario, "case": case, "passed": bool(passed)}
    row.update({k: round(v, 4) if isinstance(v, float) and math.isfinite(v) else v
                for k, v in values.items()})
    return row


def check_p1_denoise(cfg, noise, psnr_gain_min, rows):
    """Сценарій: стадія 1 пайплайна №1 має покращувати зашумлене зображення.

    Пройдено, якщо PSNR після фільтра вищий за PSNR зашумленого входу
    щонайменше на psnr_gain_min дБ і SSIM не погіршився.
    """
    seed = int(cfg["seed"])
    for path in _real_images(cfg):
        clean = load_image(path)
        noisy = apply_noise(clean, noise, seed)
        filtered = run_pipeline1(noisy, cfg["pipeline1"], seed)[0].data
        gain = metrics.psnr(clean, filtered) - metrics.psnr(clean, noisy)
        ssim_in, ssim_out = metrics.ssim(clean, noisy), metrics.ssim(clean, filtered)
        rows.append(_case(f"P1 Кувахара, {noise['name']}", path.stem,
                          gain >= psnr_gain_min and ssim_out >= ssim_in,
                          psnr_gain=gain, ssim_in=ssim_in, ssim_out=ssim_out,
                          threshold=f"gain>={psnr_gain_min} dB, SSIM_out>=SSIM_in"))


def check_p1_synthetic(cfg, iou_min, rows):
    """Сценарій: сегментація синтетичних областей з відомою розміткою."""
    seed = int(cfg["seed"])
    for n_regions in (2, 3, 4):
        for noise in ({"name": "none", "type": "none"},
                      {"name": "gaussian_10", "type": "gaussian", "sigma": 10}):
            img, truth = synthetic.make_texture_regions(128, n_regions, seed=n_regions)
            params = dict(cfg["pipeline1"])
            params["kmeans"] = dict(params["kmeans"], k=n_regions)
            labels = run_pipeline1(apply_noise(img, noise, seed), params, seed)[3].data
            iou = metrics.mean_iou(labels, truth)
            rows.append(_case("P1 синтетика, сегментація",
                              f"{n_regions} області, {noise['name']}", iou >= iou_min,
                              iou=iou, threshold=f"IoU>={iou_min}"))


def check_p2_clahe(cfg, gain_min, psnr_min, rows):
    """Сценарій: CLAHE на знімку зі штучно зниженим контрастом.

    Пройдено, якщо контраст (std) зріс щонайменше у gain_min разів, а
    PSNR відносно оригінального (неспотвореного) знімка >= psnr_min дБ.
    """
    cp = cfg["pipeline2"]["clahe"]
    for path in _real_images(cfg):
        gray = to_gray(load_image(path))
        hazy = synthetic.low_contrast(gray)
        out = clahe(hazy, tuple(cp["tiles"]), float(cp["clip_limit"]))
        gain = metrics.contrast_std(out) / metrics.contrast_std(hazy)
        value = metrics.psnr(gray, out)
        rows.append(_case("P2 CLAHE, низький контраст", path.stem,
                          gain >= gain_min and value >= psnr_min,
                          contrast_gain=gain, psnr_vs_original=value,
                          threshold=f"gain>={gain_min}, PSNR>={psnr_min} dB"))


def check_p2_lines(cfg, thresholds, rows):
    """Сценарій: пошук меж доріг на синтетичних сценах з відомими прямими."""
    seed = int(cfg["seed"])
    a_tol, r_tol = float(thresholds["p2_angle_tol"]), float(thresholds["p2_rho_tol"])
    recall_min = float(thresholds["p2_noisy_recall"])
    for idx, roads in enumerate(ROAD_SCENES):
        img, expected = synthetic.make_road_scene(256, roads, 16, seed=idx)
        for noise in ({"name": "none", "type": "none"},
                      {"name": "gaussian_20", "type": "gaussian", "sigma": 20}):
            lines = run_pipeline2(apply_noise(img, noise, seed), cfg["pipeline2"], seed)[3].data
            found, errors = hough.match_lines(lines, expected, a_tol, r_tol)
            recall = found / len(expected)
            need = 1.0 if noise["type"] == "none" else recall_min
            max_angle = max((e[1] for e in errors), default=float("nan"))
            rows.append(_case("P2 синтетика, прямі", f"сцена {idx + 1}, {noise['name']}",
                              recall >= need, recall=recall, found=found,
                              expected=len(expected), extra_lines=len(lines) - found,
                              max_angle_err=max_angle,
                              threshold=f"recall>={need}, dθ<={a_tol}°, dρ<={r_tol}"))


def run_verification(cfg, output_dir=None, log=print):
    """Виконує всі сценарії C1 і зберігає результати.

    Returns:
        (список рядків, частка пройдених тест-кейсів).
    """
    th = cfg["verify"]["thresholds"]
    rows = []
    check_p1_denoise(cfg, {"name": "gaussian_20", "type": "gaussian", "sigma": 20},
                     float(th["p1_gauss20_psnr_gain"]), rows)
    check_p1_denoise(cfg, {"name": "salt_pepper_5", "type": "salt_pepper", "amount": 0.05},
                     float(th["p1_sp5_psnr_gain"]), rows)
    check_p1_synthetic(cfg, float(th["p1_synthetic_iou"]), rows)
    check_p2_clahe(cfg, float(th["p2_clahe_contrast_gain"]), float(th["p2_clahe_psnr"]), rows)
    check_p2_lines(cfg, th, rows)
    out = Path(output_dir or cfg["output"]["dir"])
    write_table(rows, out / "verify_results.csv", out / "verify_results.json")
    passed = sum(r["passed"] for r in rows)
    share = passed / len(rows) if rows else 0.0
    by_scenario = {}
    for r in rows:
        ok, total = by_scenario.get(r["scenario"], (0, 0))
        by_scenario[r["scenario"]] = (ok + r["passed"], total + 1)
    for name, (ok, total) in by_scenario.items():
        log(f"{name}: {ok}/{total}")
    log(f"Разом пройдено {passed}/{len(rows)} тест-кейсів ({share:.0%})")
    return rows, share
