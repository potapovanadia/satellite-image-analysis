"""Підготовка та опис вибірки зображень."""
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from .io_utils import ImageLoadError, list_images, load_image


def build_manifest(root, manifest_path):
    """Перевіряє всі зображення вибірки та записує їх опис у CSV.

    Для кожного файлу зберігаються клас, розмір і середня яскравість;
    файли, що не читаються, позначаються як помилкові.

    Returns:
        (кількість коректних зображень, Counter за класами, кількість помилок).
    """
    root = Path(root)
    rows, per_class, bad = [], Counter(), 0
    for path in list_images(root):
        rel = path.relative_to(root)
        cls = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        try:
            img = load_image(path)
            rows.append({"file": rel.as_posix(), "class": cls, "width": img.shape[1],
                         "height": img.shape[0], "mean": round(float(np.mean(img)), 2),
                         "status": "ok"})
            per_class[cls] += 1
        except ImageLoadError as exc:
            rows.append({"file": rel.as_posix(), "class": cls, "width": "", "height": "",
                         "mean": "", "status": f"error: {exc}"})
            bad += 1
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "class", "width", "height",
                                                "mean", "status"])
        writer.writeheader()
        writer.writerows(rows)
    return sum(per_class.values()), per_class, bad
