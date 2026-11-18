"""Моделювання шуму для експериментів (з фіксованим seed)."""
import numpy as np


def add_gaussian_noise(img, sigma, rng):
    """Адитивний гаусів шум N(0, sigma^2), результат обрізається до 0..255."""
    noisy = img + rng.normal(0.0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255)


def add_salt_pepper_noise(img, amount, rng):
    """Імпульсний шум: частка amount пікселів стає чорною (0) або білою (255).

    Для кольорового зображення пошкоджуються всі канали пікселя одночасно.
    """
    noisy = np.array(img, dtype=np.float64, copy=True)
    h, w = noisy.shape[:2]
    mask = rng.random((h, w)) < amount
    salt = rng.random((h, w)) < 0.5
    noisy[mask & salt] = 255.0
    noisy[mask & ~salt] = 0.0
    return noisy


def apply_noise(img, spec, seed):
    """Накладає шум за описом зі словника конфігурації.

    Args:
        img: вхідне зображення.
        spec: словник {'type': 'none'|'gaussian'|'salt_pepper', 'sigma'|'amount': ...}.
        seed: зерно генератора випадкових чисел (відтворюваність).
    """
    rng = np.random.default_rng(seed)
    kind = spec.get("type", "none")
    if kind == "none":
        return np.array(img, dtype=np.float64, copy=True)
    if kind == "gaussian":
        return add_gaussian_noise(img, float(spec["sigma"]), rng)
    if kind == "salt_pepper":
        return add_salt_pepper_noise(img, float(spec["amount"]), rng)
    raise ValueError(f"Невідомий тип шуму: {kind}")
