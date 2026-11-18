"""Гістограмна корекція: гістограма та CLAHE (Pizer et al., 1987; Zuiderveld, 1994)."""
import numpy as np


def histogram(gray, bins=256):
    """Гістограма цілих рівнів яскравості 0..bins-1."""
    values = np.clip(np.rint(np.asarray(gray)), 0, bins - 1).astype(np.int64)
    return np.bincount(values.ravel(), minlength=bins).astype(np.float64)


def clip_histogram(hist, limit):
    """Обрізає стовпці гістограми на рівні limit і рівномірно розподіляє надлишок."""
    excess = np.sum(np.maximum(hist - limit, 0.0))
    clipped = np.minimum(hist, limit)
    return clipped + excess / hist.size


def tile_mapping(tile, clip_limit, bins=256):
    """Функція перетворення яскравості для однієї плитки.

    Будується гістограма плитки, обрізається на рівні
    clip_limit * (кількість пікселів / bins) (обмеження підсилення
    контрасту, а отже й шуму), далі за кумулятивною сумою (CDF):
    map(v) = round(CDF(v) / N * (bins - 1)).
    """
    hist = histogram(tile, bins)
    n = tile.size
    if clip_limit is not None and clip_limit > 0:
        hist = clip_histogram(hist, max(clip_limit * n / bins, 1.0))
    cdf = np.cumsum(hist)
    return np.rint(cdf / n * (bins - 1))


def clahe(gray, tiles=(8, 8), clip_limit=2.0):
    """Адаптивне вирівнювання гістограми з обмеженням контрасту (CLAHE).

    1. Зображення ділиться на tiles[0] x tiles[1] плиток.
    2. Для кожної плитки рахується власна функція перетворення (tile_mapping).
    3. Щоб не було видно меж плиток, значення кожного пікселя отримують
       білінійною інтерполяцією між перетвореннями чотирьох найближчих
       центрів плиток.

    Args:
        gray: масив (H, W) зі значеннями 0..255.
        tiles: кількість плиток по вертикалі та горизонталі.
        clip_limit: коефіцієнт обрізання (<= 0 або None — без обрізання).

    Returns:
        Масив (H, W), float64, 0..255.
    """
    gray = np.clip(np.rint(np.asarray(gray, dtype=np.float64)), 0, 255).astype(np.int64)
    h, w = gray.shape
    ty, tx = max(1, min(int(tiles[0]), h)), max(1, min(int(tiles[1]), w))
    # Межі плиток (останні плитки можуть бути трохи більшими).
    ys = np.linspace(0, h, ty + 1).astype(int)
    xs = np.linspace(0, w, tx + 1).astype(int)
    maps = np.empty((ty, tx, 256))
    for a in range(ty):
        for b in range(tx):
            maps[a, b] = tile_mapping(gray[ys[a]:ys[a + 1], xs[b]:xs[b + 1]], clip_limit)

    # Координати пікселів у «системі центрів плиток».
    cy = (ys[:-1] + ys[1:]) / 2.0
    cx = (xs[:-1] + xs[1:]) / 2.0
    fy = np.interp(np.arange(h) + 0.5, cy, np.arange(ty)) if ty > 1 else np.zeros(h)
    fx = np.interp(np.arange(w) + 0.5, cx, np.arange(tx)) if tx > 1 else np.zeros(w)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    y1 = np.minimum(y0 + 1, ty - 1)
    x1 = np.minimum(x0 + 1, tx - 1)
    wy = (fy - y0)[:, None]
    wx = (fx - x0)[None, :]

    yy0, xx0 = y0[:, None], x0[None, :]
    yy1, xx1 = y1[:, None], x1[None, :]
    top = (1 - wx) * maps[yy0, xx0, gray] + wx * maps[yy0, xx1, gray]
    bottom = (1 - wx) * maps[yy1, xx0, gray] + wx * maps[yy1, xx1, gray]
    return (1 - wy) * top + wy * bottom
