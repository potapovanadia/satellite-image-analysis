"""Синтетичні еталонні зображення з відомою «правильною відповіддю».

Використовуються для перевірки коректності (критерій C1): для них заздалегідь
відома розмітка областей (пайплайн №1) або параметри прямих (пайплайн №2).
"""
import numpy as np

# Текстури областей: (базовий колір RGB, тип текстури).
REGION_STYLES = [
    ((60, 120, 50), "smooth"),     # «поле»: рівний зелений
    ((110, 90, 70), "stripes"),    # «рілля»: коричневі смуги
    ((40, 60, 110), "noise"),      # «вода»: темно-синя з дрібним шумом
    ((170, 170, 160), "checker"),  # «забудова»: світла шахівниця
]


def _texture(kind, shape, rng):
    """Генерує текстурну складову (зсув яскравості) заданого типу."""
    h, w = shape
    yy, xx = np.indices((h, w))
    if kind == "smooth":
        return rng.normal(0, 3, size=(h, w))
    if kind == "stripes":
        return 25.0 * np.sign(np.sin(xx * np.pi / 3.0))
    if kind == "noise":
        return rng.normal(0, 12, size=(h, w))
    if kind == "checker":
        return 30.0 * (((yy // 4) + (xx // 4)) % 2 * 2 - 1)
    raise ValueError(kind)


def make_texture_regions(size=128, n_regions=4, seed=0):
    """Зображення з n_regions (2..4) областями різного кольору й текстури.

    Області розділені вертикальною і (для 3–4 областей) горизонтальною
    межами. Returns: (зображення (H, W, 3) 0..255, мітки (H, W)).
    """
    if not 2 <= n_regions <= 4:
        raise ValueError("n_regions має бути від 2 до 4")
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), dtype=np.int64)
    half = size // 2
    labels[:, half:] = 1
    if n_regions >= 3:
        labels[half:, :half] = 2
    if n_regions == 4:
        labels[half:, half:] = 3
    img = np.zeros((size, size, 3))
    for lab in range(n_regions):
        base, kind = REGION_STYLES[lab]
        tex = _texture(kind, (size, size), rng)
        for c in range(3):
            img[..., c] = np.where(labels == lab, base[c] + tex, img[..., c])
    return np.clip(img, 0, 255), labels


def make_road_scene(size=128, roads=((40.0, 30.0), (-20.0, -60.0)), road_width=8, seed=0):
    """Сцена «поле з дорогами»: світлі смуги на текстурованому фоні.

    Кожна дорога задана центральною лінією (rho, theta у градусах) і
    шириною. Межі дороги — дві паралельні прямі rho -+ road_width/2,
    саме їх має знайти пайплайн №2.

    Returns:
        (зображення (H, W, 3), список очікуваних прямих [(rho, theta), ...]).
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    field = 90 + rng.normal(0, 4, size=(size, size))
    road_mask = np.zeros((size, size), dtype=bool)
    expected = []
    for rho, theta in roads:
        t = np.deg2rad(theta)
        dist = xx * np.cos(t) + yy * np.sin(t) - rho
        road_mask |= np.abs(dist) <= road_width / 2.0
        expected += [(rho - road_width / 2.0, theta), (rho + road_width / 2.0, theta)]
    gray = np.where(road_mask, 200.0, field)
    img = np.stack([gray * 0.9, gray, gray * 0.8], axis=-1)
    return np.clip(img, 0, 255), expected


def low_contrast(img, low=100.0, high=150.0):
    """Стискає діапазон яскравості зображення до [low, high] (імітація серпанку)."""
    img = np.asarray(img, dtype=np.float64)
    lo, hi = img.min(), img.max()
    span = hi - lo if hi > lo else 1.0
    return low + (img - lo) / span * (high - low)
