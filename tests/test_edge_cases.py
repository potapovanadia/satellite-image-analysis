"""Крайні випадки (критерій C2): нестандартні розміри, монохромні та однорідні зображення."""
import numpy as np
import pytest
from PIL import Image

from satimg.cli import main
from satimg.config import load_config
from satimg.pipelines import run_pipeline1, run_pipeline2

CFG = load_config("config.yaml")


@pytest.mark.parametrize("shape", [(16, 16, 3), (20, 41, 3), (64, 17, 3)])
def test_pipelines_on_unusual_sizes(shape):
    img = np.random.default_rng(0).integers(0, 256, shape).astype(float)
    for run in (run_pipeline1, run_pipeline2):
        stages = run(img, CFG["pipeline1" if run is run_pipeline1 else "pipeline2"])
        assert all(s.preview.shape[:2] == shape[:2] for s in stages)


@pytest.mark.parametrize("value", [0.0, 128.0, 255.0])
def test_constant_images(value):
    img = np.full((32, 32, 3), value)
    labels = run_pipeline1(img, CFG["pipeline1"])[-1].data
    assert len(np.unique(labels)) == 1          # один сегмент, без помилки k-means
    stages = run_pipeline2(img, CFG["pipeline2"])
    assert not stages[1].data.any()             # немає контурів
    assert stages[-1].data == []                # немає прямих


def test_monochrome_file_through_cli(tmp_path):
    root = tmp_path / "data" / "gray"
    root.mkdir(parents=True)
    gray = np.random.default_rng(1).integers(0, 256, (32, 32)).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(root / "g.png")
    Image.fromarray(gray).save(root / "g.jpg")
    code = main(["run", "--input", str(tmp_path / "data"), "--output", str(tmp_path / "out"),
                 "--noise", "none"])
    assert code == 0


def test_only_broken_files(tmp_path):
    root = tmp_path / "data" / "x"
    root.mkdir(parents=True)
    (root / "a.png").write_bytes(b"")
    (root / "b.tif").write_bytes(b"garbage")
    code = main(["run", "--input", str(tmp_path / "data"), "--output", str(tmp_path / "out")])
    assert code == 1                             # нічого не оброблено, але без аварії
    assert (tmp_path / "out" / "errors.csv").exists()


def test_missing_input_folder(tmp_path):
    assert main(["run", "--input", str(tmp_path / "nope"), "--output", str(tmp_path)]) == 1


def test_bad_cli_values(tmp_path):
    assert main(["run", "--noise", "fog", "--output", str(tmp_path)]) == 2
    assert main(["config", "--set", "pipeline1.unknown=1"]) == 2
    assert main(["method", "hsv", "--image", "data/dataset/forest/forest00.tif",
                 "--noise", "fog"]) == 2
