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

    y0, y1, wy = _interpolation_weights(ys)
    x0, x1, wx = _interpolation_weights(xs)
    y0, y1, wy = y0[:, None], y1[:, None], wy[:, None]
    x0, x1, wx = x0[None, :], x1[None, :], wx[None, :]
    top = (1 - wx) * maps[y0, x0, gray] + wx * maps[y0, x1, gray]
    bottom = (1 - wx) * maps[y1, x0, gray] + wx * maps[y1, x1, gray]
    return (1 - wy) * top + wy * bottom


def _interpolation_weights(edges):
    """Сусідні плитки та ваги білінійної інтерполяції вздовж однієї осі.

    Для кожного пікселя знаходить індекси двох найближчих центрів плиток
    (i0, i1) і вагу другого з них (0..1). Біля країв зображення обидва
    індекси збігаються, тобто використовується одна плитка.
    """
    count = len(edges) - 1
    size = edges[-1]
    centers = (edges[:-1] + edges[1:]) / 2.0
    if count > 1:
        pos = np.interp(np.arange(size) + 0.5, centers, np.arange(count))
    else:
        pos = np.zeros(size)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, count - 1)
    return i0, i1, pos - i0
