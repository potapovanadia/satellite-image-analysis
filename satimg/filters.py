"""Фільтрація: фільтр Кувахари (Kuwahara), що зберігає межі об'єктів."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .core import to_gray


def kuwahara(img, window=5):
    """Фільтр Кувахари.

    Вікно window x window (window = 2a + 1) навколо пікселя ділиться на
    чотири квадранти розміром (a+1) x (a+1), які перетинаються по
    центральному рядку і стовпцю. Для кожного квадранта рахуються середнє
    і дисперсія яскравості. Піксель отримує середнє значення (по кожному
    каналу) того квадранта, де дисперсія найменша, тобто найбільш
    однорідного. Тому шум у однорідних областях згладжується, а межа між
    областями не розмивається: квадрант, що перетинає межу, має велику
    дисперсію і не обирається.

    Args:
        img: масив (H, W) або (H, W, C).
        window: непарний розмір вікна >= 3.

    Returns:
        Відфільтроване зображення тієї ж форми, float64.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window має бути непарним і >= 3")
    img = np.asarray(img, dtype=np.float64)
    single = img.ndim == 2
    data = img[..., np.newaxis] if single else img
    a = window // 2
    h, w = data.shape[:2]
    q = a + 1

    def quadrant_means(channel):
        """Середні у вікнах q x q; повертає 4 квадранти для кожного пікселя."""
        padded = np.pad(channel, a, mode="reflect")
        means = sliding_window_view(padded, (q, q)).mean(axis=(-2, -1))
        # Для пікселя (i, j) квадранти починаються в (i, j), (i, j+a),
        # (i+a, j), (i+a, j+a) у координатах доповненого масиву.
        return np.stack([means[0:h, 0:w], means[0:h, a:a + w],
                         means[a:a + h, 0:w], means[a:a + h, a:a + w]])

    lum = to_gray(data) if data.shape[2] == 3 else data[..., 0]
    mean_l = quadrant_means(lum)
    var_l = quadrant_means(lum * lum) - mean_l ** 2
    best = np.argmin(var_l, axis=0)  # номер найоднорідніших квадрантів, (H, W)

    out = np.empty_like(data)
    rows, cols = np.indices((h, w))
    for c in range(data.shape[2]):
        out[..., c] = quadrant_means(data[..., c])[best, rows, cols]
    return out[..., 0] if single else out
