"""Введення/виведення зображень і перевірка вхідних файлів.

Бібліотека Pillow використовується ТІЛЬКИ для читання та запису файлів.
Уся обробка зображень реалізована вручну в інших модулях пакета.
"""
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


class ImageLoadError(Exception):
    """Помилка завантаження: файл порожній, пошкоджений або непідтримуваний."""


def load_image(path, min_size=3):
    """Завантажує зображення як RGB-масив float64 зі значеннями 0..255.

    Обробляє крайні випадки:
      * монохромні (L, 1, I;16, F) зображення перетворюються на 3 канали;
      * RGBA / палітрові зображення перетворюються на RGB;
      * 16-бітні зображення масштабуються до діапазону 0..255.

    Args:
        path: шлях до файлу.
        min_size: мінімальна допустима ширина/висота у пікселях.

    Returns:
        np.ndarray форми (H, W, 3), dtype float64.

    Raises:
        ImageLoadError: якщо файл відсутній, порожній, пошкоджений,
            має непідтримуваний формат або замалий розмір.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"Файл не знайдено: {path}")
    if path.stat().st_size == 0:
        raise ImageLoadError(f"Порожній файл: {path}")
    try:
        with Image.open(path) as img:
            img.load()
            if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
                arr = np.asarray(img, dtype=np.float64)
                peak = arr.max()
                if peak > 255:
                    arr = arr / peak * 255.0
            else:
                arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Не вдалося прочитати {path}: {exc}") from exc

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[0] < min_size or arr.shape[1] < min_size:
        raise ImageLoadError(
            f"Замале зображення {arr.shape[1]}x{arr.shape[0]} (мінімум {min_size}): {path}"
        )
    return arr


def to_uint8(arr):
    """Перетворює масив (bool або float) на uint8 з обрізанням до 0..255."""
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def save_image(arr, path):
    """Зберігає масив (H, W) або (H, W, 3) у файл, створюючи потрібні папки."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(arr)).save(path)


def list_images(folder):
    """Повертає відсортований список шляхів до зображень у папці (рекурсивно)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
