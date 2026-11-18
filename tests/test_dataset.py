"""Тести опису вибірки."""
import csv

from PIL import Image

from satimg.dataset import build_manifest


def test_manifest_counts_classes_and_errors(tmp_path):
    root = tmp_path / "data"
    for cls, count in (("forest", 2), ("river", 1)):
        (root / cls).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (8, 8), (10 * i, 20, 30)).save(root / cls / f"{cls}{i}.png")
    (root / "river" / "broken.tif").write_bytes(b"not an image")

    ok, per_class, bad = build_manifest(root, tmp_path / "manifest.csv")

    assert ok == 3 and bad == 1
    assert per_class == {"forest": 2, "river": 1}
    with (tmp_path / "manifest.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    assert {r["status"] for r in rows if "broken" in r["file"]} != {"ok"}


def test_project_dataset_is_complete():
    ok, per_class, bad = build_manifest("data/dataset", "results/test_manifest.csv")
    assert ok == 100 and bad == 0
    assert set(per_class.values()) == {20}
