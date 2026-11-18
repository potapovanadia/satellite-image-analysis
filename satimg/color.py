"""Колірні перетворення: RGB -> HSV і назад."""
import numpy as np


def rgb_to_hsv(img):
    """Перетворює RGB (0..255) у HSV.

    V = max(R, G, B); S = (max - min) / max (0, якщо max = 0);
    H (у градусах 0..360) залежить від того, який канал максимальний:
      R: H = 60 * ((G - B) / delta mod 6)
      G: H = 60 * ((B - R) / delta + 2)
      B: H = 60 * ((R - G) / delta + 4)
    Для сірих пікселів (delta = 0) H = 0.

    Returns:
        Масив (H, W, 3): H у градусах [0, 360), S і V у [0, 1].
    """
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = rgb.max(axis=-1)
    cmin = rgb.min(axis=-1)
    delta = cmax - cmin
    safe = np.where(delta == 0, 1.0, delta)

    hue = np.zeros_like(cmax)
    is_r = (cmax == r) & (delta > 0)
    is_g = (cmax == g) & (delta > 0) & ~is_r
    is_b = (delta > 0) & ~is_r & ~is_g
    hue[is_r] = (60.0 * ((g - b) / safe) % 360.0)[is_r]
    hue[is_g] = (60.0 * ((b - r) / safe + 2.0))[is_g]
    hue[is_b] = (60.0 * ((r - g) / safe + 4.0))[is_b]

    sat = np.where(cmax == 0, 0.0, delta / np.where(cmax == 0, 1.0, cmax))
    return np.stack([hue, sat, cmax], axis=-1)


def hsv_to_rgb(hsv):
    """Зворотне перетворення HSV -> RGB (0..255), використовується для перевірки."""
    hsv = np.asarray(hsv, dtype=np.float64)
    h, s, v = hsv[..., 0] % 360.0, hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c
    sector = (h // 60).astype(int)
    zeros = np.zeros_like(h)
    table = [(c, x, zeros), (x, c, zeros), (zeros, c, x),
             (zeros, x, c), (x, zeros, c), (c, zeros, x)]
    rgb = np.zeros(h.shape + (3,))
    for idx, (rr, gg, bb) in enumerate(table):
        mask = sector == idx
        rgb[mask] = np.stack([rr[mask], gg[mask], bb[mask]], axis=-1)
    return (rgb + m[..., np.newaxis]) * 255.0


def hsv_to_display(hsv):
    """Представлення HSV як зображення для збереження: H, S, V -> 0..255."""
    hsv = np.asarray(hsv, dtype=np.float64)
    return np.stack([hsv[..., 0] / 360.0 * 255.0,
                     hsv[..., 1] * 255.0,
                     hsv[..., 2] * 255.0], axis=-1)
