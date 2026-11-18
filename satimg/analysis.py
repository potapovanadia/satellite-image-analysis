"""Аналіз результатів: зведені таблиці, порівняння пайплайнів і графіки.

matplotlib використовується тільки для побудови графіків.
"""
import csv
import math
from pathlib import Path

import numpy as np

NOISE_ORDER = ["none", "gaussian_10", "gaussian_20", "gaussian_30", "salt_pepper_5"]
NOISE_LABELS = {"none": "без шуму", "gaussian_10": "гаус σ=10", "gaussian_20": "гаус σ=20",
                "gaussian_30": "гаус σ=30", "salt_pepper_5": "імпульсний 5%"}

# Метрики для зведених таблиць: (стовпець, підпис).
P1_METRICS = [("input_psnr", "PSNR входу, дБ"), ("s1_psnr", "PSNR після Кувахари, дБ"),
              ("s1_psnr_gain", "Приріст PSNR, дБ"), ("input_ssim", "SSIM входу"),
              ("s1_ssim", "SSIM після Кувахари"), ("seg_miou_vs_clean", "mIoU сегментації"),
              ("seg_acc_vs_clean", "Точність сегментації"), ("time_s", "Час, с")]
P2_METRICS = [("clahe_contrast_gain", "Приріст контрасту CLAHE"),
              ("s1_psnr", "PSNR CLAHE (шум/чистий), дБ"), ("s1_ssim", "SSIM CLAHE (шум/чистий)"),
              ("edge_density", "Щільність контурів"), ("edges_f1_vs_clean", "F1 контурів Кенні"),
              ("morph_f1_vs_clean", "F1 після морфології"), ("n_lines", "Кількість прямих"),
              ("lines_recall_vs_clean", "Повнота прямих"), ("time_s", "Час, с")]

# Палітра (валідована категоріальна послідовність) і кольори оформлення.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = ("#0b0b0b", "#52514e", "#898781", "#e1e0d9",
                                         "#c3c2b7", "#fcfcfb")


def read_metrics(path):
    """Читає CSV з метриками; числа перетворює на float ('inf'/'nan' теж)."""
    with Path(path).open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
    return rows


def stats(values):
    """Середнє, стандартне відхилення, медіана і 95% довірчий інтервал середнього.

    Нескінченні та NaN-значення не враховуються.
    """
    vals = np.array([v for v in values if isinstance(v, float) and math.isfinite(v)])
    if vals.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "ci95": float("nan")}
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return {"n": int(vals.size), "mean": float(vals.mean()), "std": std,
            "median": float(np.median(vals)), "ci95": 1.96 * std / math.sqrt(vals.size)}


def group(rows, *keys):
    """Групує рядки за значеннями вказаних стовпців."""
    out = {}
    for row in rows:
        out.setdefault(tuple(row[k] for k in keys), []).append(row)
    return out


def summary_table(rows, metrics, by=("noise",)):
    """Зведена таблиця: для кожної групи — середнє ± CI95 кожної метрики."""
    table = []
    for key, items in sorted(group(rows, *by).items(), key=lambda kv: _sort_key(kv[0])):
        line = dict(zip(by, key))
        line["n"] = len(items)
        for col, _ in metrics:
            st = stats([r.get(col) for r in items])
            line[f"{col}_mean"] = round(st["mean"], 4)
            line[f"{col}_ci95"] = round(st["ci95"], 4)
        table.append(line)
    return table


def _sort_key(key):
    return tuple(NOISE_ORDER.index(k) if k in NOISE_ORDER else 99 for k in key[-1:]) + key


def write_csv(table, path):
    """Записує список словників у CSV."""
    if not table:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)


def markdown_table(table, metrics, by=("noise",)):
    """Таблиця Markdown «середнє ± CI95» для звіту."""
    header = [*by, *(label for _, label in metrics)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in table:
        cells = [NOISE_LABELS.get(row[k], row[k]) for k in by]
        for col, _ in metrics:
            mean, ci = row[f"{col}_mean"], row[f"{col}_ci95"]
            cells.append("—" if not math.isfinite(mean) else f"{mean:.3f} ± {ci:.3f}")
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    return "\n".join(lines)


def _style(ax, title, ylabel):
    """Спільне оформлення осей: тонка сітка, приглушені осі, заголовок."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)


def _mean_by(rows, col, key="noise", order=None):
    groups = group(rows, key)
    order = order or [k[0] for k in groups]
    return order, [stats([r[col] for r in groups.get((k,), [])])["mean"] for k in order]


def plot_all(p1_rows, p2_rows, out_dir):
    """Будує графіки для звіту (PNG) і повертає список файлів."""
    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    noisy = [n for n in NOISE_ORDER if n != "none"]

    def save(fig, name):
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=150)
        plt.close(fig)
        files.append(out_dir / name)

    if p1_rows:
        # 1. Якість фільтрації Кувахари: PSNR і SSIM до/після (дві окремі осі-панелі).
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        x = np.arange(len(noisy))
        for ax, (col_in, col_out, label) in zip(
                axes, [("input_psnr", "s1_psnr", "PSNR, дБ"), ("input_ssim", "s1_ssim", "SSIM")]):
            _, before = _mean_by(p1_rows, col_in, order=noisy)
            _, after = _mean_by(p1_rows, col_out, order=noisy)
            ax.bar(x - 0.2, before, 0.38, color=SERIES[0], label="зашумлений вхід")
            ax.bar(x + 0.2, after, 0.38, color=SERIES[1], label="після Кувахари")
            ax.set_xticks(x, [NOISE_LABELS[n] for n in noisy])
            _style(ax, f"Пайплайн №1, стадія 1: {label}", label)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        save(fig, "p1_denoise_quality.png")

        # 2. Приріст PSNR за класами при гаусовому шумі σ=20.
        g20 = [r for r in p1_rows if r["noise"] == "gaussian_20"]
        classes, gains = _mean_by(g20, "s1_psnr_gain", key="class")
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.bar(classes, gains, color=SERIES[0], width=0.6)
        for i, val in enumerate(gains):
            ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        _style(ax, "Приріст PSNR після Кувахари за класами (гаус σ=20)", "дБ")
        save(fig, "p1_gain_by_class.png")

    if p1_rows and p2_rows:
        # 3. Порівняння стійкості пайплайнів до шуму (одна шкала 0..1).
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        series = [(p1_rows, "seg_miou_vs_clean", "П1: mIoU сегментації"),
                  (p2_rows, "edges_f1_vs_clean", "П2: F1 контурів"),
                  (p2_rows, "lines_recall_vs_clean", "П2: повнота прямих")]
        for idx, (rows, col, label) in enumerate(series):
            _, vals = _mean_by(rows, col, order=NOISE_ORDER)
            ax.plot(range(len(NOISE_ORDER)), vals, marker="o", linewidth=2, markersize=6,
                    color=SERIES[idx], label=label)
            ax.annotate(label, (len(NOISE_ORDER) - 1, vals[-1]), xytext=(6, 0),
                        textcoords="offset points", fontsize=8, color=INK2, va="center")
        ax.set_xticks(range(len(NOISE_ORDER)), [NOISE_LABELS[n] for n in NOISE_ORDER])
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.3, len(NOISE_ORDER) + 0.9)
        _style(ax, "Стійкість результату до шуму (збіг з результатом на чистому знімку)",
               "частка збігу")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
        save(fig, "pipelines_stability.png")

        # 4. Стійкість за класами при σ=20: окремо для кожного пайплайна (small multiples).
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
        for ax, (rows, col, label) in zip(axes, [series[0], series[2]]):
            sub = [r for r in rows if r["noise"] == "gaussian_20"]
            classes, vals = _mean_by(sub, col, key="class")
            ax.bar(classes, vals, color=SERIES[0], width=0.6)
            for i, val in enumerate(vals):
                ax.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
            ax.tick_params(axis="x", rotation=20)
            _style(ax, f"{label} (гаус σ=20)", "частка збігу")
        save(fig, "stability_by_class.png")

        # 5. Середній час обробки одного знімка.
        fig, ax = plt.subplots(figsize=(5, 3))
        times = [stats([r["time_s"] for r in rows])["mean"] for rows in (p1_rows, p2_rows)]
        ax.bar(["Пайплайн №1", "Пайплайн №2"], times, color=SERIES[0], width=0.5)
        for i, val in enumerate(times):
            ax.text(i, val, f"{val:.2f} с", ha="center", va="bottom", fontsize=8, color=INK2)
        _style(ax, "Середній час обробки знімка 256×256", "секунди")
        save(fig, "pipelines_time.png")
    return files


def compare_pipelines(p1_rows, p2_rows):
    """Пряме порівняння пайплайнів за спільними показниками для кожної умови шуму."""
    table = []
    for noise in NOISE_ORDER:
        a = [r for r in p1_rows if r["noise"] == noise]
        b = [r for r in p2_rows if r["noise"] == noise]
        if not a or not b:
            continue
        table.append({
            "noise": noise,
            "p1_stage1_psnr_vs_ref": round(stats([r["s1_psnr"] for r in a])["mean"], 3),
            "p2_stage1_psnr_vs_ref": round(stats([r["s1_psnr"] for r in b])["mean"], 3),
            "p1_stage1_ssim_vs_ref": round(stats([r["s1_ssim"] for r in a])["mean"], 4),
            "p2_stage1_ssim_vs_ref": round(stats([r["s1_ssim"] for r in b])["mean"], 4),
            "p1_final_stability": round(stats([r["seg_miou_vs_clean"] for r in a])["mean"], 4),
            "p2_final_stability": round(stats([r["lines_recall_vs_clean"] for r in b])["mean"], 4),
            "p1_time_s": round(stats([r["time_s"] for r in a])["mean"], 3),
            "p2_time_s": round(stats([r["time_s"] for r in b])["mean"], 3),
        })
    return table


def correlation(rows, col_x, col_y):
    """Коефіцієнт кореляції Пірсона між двома метриками (скінченні пари)."""
    pairs = [(r[col_x], r[col_y]) for r in rows
             if isinstance(r.get(col_x), float) and isinstance(r.get(col_y), float)
             and math.isfinite(r[col_x]) and math.isfinite(r[col_y])]
    if len(pairs) < 3:
        return float("nan")
    x, y = np.array(pairs).T
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def run_analysis(results_dir, log=print):
    """Будує всі таблиці й графіки з results/metrics_*.csv у results/analysis/."""
    results_dir = Path(results_dir)
    out = results_dir / "analysis"
    p1_path, p2_path = results_dir / "metrics_p1.csv", results_dir / "metrics_p2.csv"
    p1 = read_metrics(p1_path) if p1_path.exists() else []
    p2 = read_metrics(p2_path) if p2_path.exists() else []
    if not p1 and not p2:
        log(f"У {results_dir} немає metrics_p1.csv / metrics_p2.csv — спершу виконайте run")
        return False

    report = ["# Зведені результати", ""]
    for name, rows, metrics in (("p1", p1, P1_METRICS), ("p2", p2, P2_METRICS)):
        if not rows:
            continue
        by_noise = summary_table(rows, metrics, ("noise",))
        by_class = summary_table([r for r in rows if r["noise"] == "gaussian_20"],
                                 metrics, ("class",))
        write_csv(by_noise, out / f"{name}_by_noise.csv")
        write_csv(by_class, out / f"{name}_by_class_gaussian20.csv")
        report += [f"## Пайплайн {name}: за рівнем шуму (середнє ± 95% ДІ)", "",
                   markdown_table(by_noise, metrics, ("noise",)), "",
                   f"## Пайплайн {name}: за класами, гаус σ=20", "",
                   markdown_table(by_class, metrics, ("class",)), ""]
    if p1 and p2:
        comparison = compare_pipelines(p1, p2)
        write_csv(comparison, out / "pipelines_comparison.csv")
        report += ["## Порівняння пайплайнів", "", "| " + " | ".join(comparison[0]) + " |",
                   "|" + "---|" * len(comparison[0])]
        report += ["| " + " | ".join(str(v) for v in row.values()) + " |" for row in comparison]
        report += ["", "## Кореляції (усі зашумлені запуски)", ""]
        noisy1 = [r for r in p1 if r["noise"] != "none"]
        noisy2 = [r for r in p2 if r["noise"] != "none"]
        report += [
            f"- П1: PSNR входу ↔ mIoU сегментації: r = "
            f"{correlation(noisy1, 'input_psnr', 'seg_miou_vs_clean'):.3f}",
            f"- П2: PSNR входу ↔ повнота прямих: r = "
            f"{correlation(noisy2, 'input_psnr', 'lines_recall_vs_clean'):.3f}",
            f"- П2: щільність контурів ↔ повнота прямих: r = "
            f"{correlation(noisy2, 'edge_density', 'lines_recall_vs_clean'):.3f}", ""]
    files = plot_all(p1, p2, out)
    (out / "summary.md").write_text("\n".join(report), encoding="utf-8")
    log(f"Таблиці та звіт: {out / 'summary.md'}")
    for f in files:
        log(f"Графік: {f}")
    return True


SWEEP_METRICS = {
    "p1": ["s1_psnr_gain", "s1_ssim", "seg_miou_vs_clean", "time_s"],
    "p2": ["clahe_contrast_gain", "edges_f1_vs_clean", "lines_recall_vs_clean", "n_lines",
           "time_s"],
}


def run_sweep(cfg, param, values, pipeline, noise_names, per_class, output_dir,  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
              log=print):
    """Підбір параметра: запуск пайплайна для кожного значення та порівняння метрик.

    Наприклад, param='pipeline1.kuwahara.window', values=['3', '5', '7'].
    Результат — results/sweeps/<param>.csv і графік з середніми метриками.
    """
    from .config import apply_overrides  # pylint: disable=import-outside-toplevel
    from .runner import run_experiment  # pylint: disable=import-outside-toplevel
    out = Path(output_dir) / "sweeps"
    table = []
    for value in values:
        run_cfg = apply_overrides(cfg, [f"{param}={value}", "output.save_stages=false"])
        rows = run_experiment(run_cfg, [pipeline], output_dir=out / f"{param}={value}",
                              noise_names=noise_names, per_class=per_class,
                              log=lambda *_: None)[pipeline]
        noisy = [r for r in rows if r["noise"] != "none"] or rows
        line = {"param": param, "value": value, "runs": len(noisy)}
        for col in SWEEP_METRICS[pipeline]:
            line[col] = round(stats([float(r[col]) for r in noisy])["mean"], 4)
        table.append(line)
        log(", ".join(f"{k}={v}" for k, v in line.items()))
    write_csv(table, out / f"{param}.csv")

    import matplotlib  # pylint: disable=import-outside-toplevel
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    cols = SWEEP_METRICS[pipeline]
    fig, axes = plt.subplots(1, len(cols), figsize=(3 * len(cols), 2.8))
    for ax, col in zip(np.atleast_1d(axes), cols):
        ax.plot([str(r["value"]) for r in table], [r[col] for r in table], marker="o",
                linewidth=2, markersize=6, color=SERIES[0])
        _style(ax, col, "")
        ax.set_xlabel(param.split(".")[-1], color=INK2, fontsize=8)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out / f"{param}.png", dpi=150)
    plt.close(fig)
    log(f"Таблиця: {out / f'{param}.csv'}, графік: {out / f'{param}.png'}")
    return table
