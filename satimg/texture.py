"""Аналіз текстур: матриця суміжності рівнів сірого (GLCM, Haralick, 1973)."""
import numpy as np

from .core import box_sum

DEFAULT_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


def quantize(gray, levels):
    """Квантує яскравість 0..255 у levels рівнів: q = floor(I * levels / 256)."""
    gray = np.asarray(gray, dtype=np.float64)
    q = np.floor(np.clip(gray, 0, 255) * levels / 256.0).astype(np.int64)
    return np.clip(q, 0, levels - 1)


def glcm(quantized, levels, offset=(0, 1), symmetric=True, normed=True):
    """Матриця GLCM для одного фрагмента.

    Елемент P[i, j] — кількість пар пікселів (p, p + offset), де перший
    має рівень i, а другий — j. Симетрична матриця додатково враховує
    пару у зворотному напрямку. Нормована матриця ділиться на суму.

    Args:
        quantized: масив цілих рівнів 0..levels-1.
        levels: кількість рівнів сірого.
        offset: зсув (dy, dx).
    """
    q = np.asarray(quantized)
    dy, dx = offset
    h, w = q.shape
    y0, y1 = max(0, -dy), min(h, h - dy)
    x0, x1 = max(0, -dx), min(w, w - dx)
    first = q[y0:y1, x0:x1].ravel()
    second = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
    matrix = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(matrix, (first, second), 1.0)
    if symmetric:
        matrix = matrix + matrix.T
    if normed and matrix.sum() > 0:
        matrix = matrix / matrix.sum()
    return matrix


def glcm_features(matrix):
    """Ознаки Haralick для нормованої GLCM P.

    contrast    = sum (i - j)^2 P[i, j]       — локальні перепади яскравості;
    energy      = sum P[i, j]^2               — впорядкованість текстури;
    homogeneity = sum P[i, j] / (1 + |i - j|) — близькість до діагоналі;
    entropy     = -sum P log2 P               — хаотичність текстури.
    """
    p = np.asarray(matrix, dtype=np.float64)
    levels = p.shape[0]
    i, j = np.indices((levels, levels))
    nz = p[p > 0]
    return {
        "contrast": float(np.sum((i - j) ** 2 * p)),
        "energy": float(np.sum(p ** 2)),
        "homogeneity": float(np.sum(p / (1.0 + np.abs(i - j)))),
        "entropy": float(-np.sum(nz * np.log2(nz))),
    }


FEATURE_NAMES = ("contrast", "energy", "homogeneity", "entropy")


def glcm_feature_maps(gray, window=9, levels=8, offsets=DEFAULT_OFFSETS):
    """Карти ознак GLCM у ковзному вікні для кожного пікселя.

    Замість побудови окремої матриці для кожного вікна (дуже повільно)
    використано прийом: для кожної пари рівнів (i, j) будується бінарна
    карта «у цьому пікселі починається пара (i, j)», після чого віконна
    сума через інтегральне зображення дає GLCM усіх вікон одночасно.
    Матриці симетричні та усереднені за всіма зсувами.

    Returns:
        Масив (H, W, 4) з ознаками у порядку FEATURE_NAMES.
    """
    q = quantize(gray, levels)
    h, w = q.shape
    counts = np.zeros((levels * levels, h, w), dtype=np.float64)
    rows, cols = np.indices((h, w))
    for dy, dx in offsets:
        y0, y1 = max(0, -dy), min(h, h - dy)
        x0, x1 = max(0, -dx), min(w, w - dx)
        first = q[y0:y1, x0:x1]
        second = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        r = rows[y0:y1, x0:x1]
        c = cols[y0:y1, x0:x1]
        np.add.at(counts, (first * levels + second, r, c), 1.0)
        np.add.at(counts, (second * levels + first, r, c), 1.0)

    sums = box_sum(counts, window)
    total = sums.sum(axis=0)
    total[total == 0] = 1.0
    p = sums / total

    i, j = np.divmod(np.arange(levels * levels), levels)
    diff = (i - j).astype(np.float64)[:, None, None]
    contrast = np.sum(diff ** 2 * p, axis=0)
    energy = np.sum(p ** 2, axis=0)
    homogeneity = np.sum(p / (1.0 + np.abs(diff)), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(np.where(p > 0, p, 1.0)), 0.0)
    entropy = -np.sum(p * logp, axis=0)
    return np.stack([contrast, energy, homogeneity, entropy], axis=-1)


def features_to_display(features):
    """Псевдокольорове зображення ознак: R=контраст, G=однорідність, B=ентропія."""
    picks = features[..., [0, 2, 3]]
    lo = picks.min(axis=(0, 1))
    hi = picks.max(axis=(0, 1))
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    return (picks - lo) / span * 255.0
