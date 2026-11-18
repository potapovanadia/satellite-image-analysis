"""Тести пайплайнів і запуску на вибірці (автоматичне виконання всіх стадій)."""
import csv

import numpy as np
from PIL import Image

from satimg.cli import main
from satimg.config import load_config
from satimg.pipelines import METHODS, run_pipeline1, run_pipeline2
from satimg.runner import run_experiment
from satimg.synthetic import make_road_scene, make_texture_regions

CFG = load_config("config.yaml")


def test_pipeline1_stages_and_segmentation():
    img, truth = make_texture_regions(64, 2, seed=0)
    stages = run_pipeline1(img, CFG["pipeline1"])
    assert [s.name for s in stages] == ["01_kuwahara", "02_hsv", "03_glcm", "04_kmeans"]
    assert len({s.category for s in stages}) == 4  # 4 різні категорії методів
    labels = stages[-1].data
    # Дві половини з різною текстурою мають потрапити в різні сегменти.
    assert labels[32, 5] != labels[32, 60]
    assert np.mean(labels == labels[0, 0]) > 0.3
    assert truth.shape == labels.shape


def test_pipeline2_stages_and_lines():
    img, expected = make_road_scene(128, ((64.0, 0.0),), road_width=16, seed=0)
    stages = run_pipeline2(img, CFG["pipeline2"])
    assert [s.name for s in stages] == ["01_clahe", "02_canny", "03_morphology", "04_hough"]
    assert len({s.category for s in stages}) == 4
    thetas = sorted(round(line[1]) for line in stages[-1].data)
    assert thetas == [0, 0] and len(expected) == 2  # обидві межі вертикальної дороги


def test_every_method_runs():
    img = np.random.default_rng(0).integers(0, 256, (24, 24, 3)).astype(float)
    for name, func in METHODS.items():
        out = func(img, CFG, 1)
        assert out.shape[:2] == (24, 24), name


def _tiny_dataset(root):
    for cls in ("a", "b"):
        (root / cls).mkdir(parents=True)
        img, _ = make_texture_regions(32, 2, seed=1)
        Image.fromarray(img.astype(np.uint8)).save(root / cls / f"{cls}0.png")
    (root / "b" / "broken.png").write_bytes(b"")  # пошкоджений файл не має зупиняти запуск


def test_runner_writes_metrics_and_stages(tmp_path):
    _tiny_dataset(tmp_path / "data")
    rows = run_experiment(CFG, ["p1", "p2"], tmp_path / "data", tmp_path / "out",
                          noise_names=["none", "gaussian_20"], log=lambda *_: None)
    assert len(rows["p1"]) == 4 and len(rows["p2"]) == 4  # 2 зображення x 2 умови
    with (tmp_path / "out" / "metrics_p1.csv").open(encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert {"s1_psnr", "s1_ssim", "s1_mse", "seg_miou_vs_clean"} <= set(table[0])
    assert (tmp_path / "out" / "errors.csv").exists()
    stage_dir = tmp_path / "out" / "p2" / "gaussian_20" / "a" / "a0"
    assert sorted(p.name for p in stage_dir.iterdir()) == [
        "00_input.png", "01_clahe.png", "02_canny.png", "03_morphology.png", "04_hough.png"]


def test_cli_run_and_method(tmp_path):
    _tiny_dataset(tmp_path / "data")
    code = main(["run", "--pipeline", "p2", "--input", str(tmp_path / "data"),
                 "--output", str(tmp_path / "out"), "--noise", "none",
                 "--set", "pipeline2.canny.sigma=2.0"])
    assert code == 0 and (tmp_path / "out" / "metrics_p2.json").exists()
    out_png = tmp_path / "m.png"
    assert main(["method", "kuwahara", "--image", str(tmp_path / "data" / "a" / "a0.png"),
                 "--output", str(out_png)]) == 0
    assert out_png.exists()
    assert main(["method", "canny", "--image", str(tmp_path / "missing.png")]) == 1
