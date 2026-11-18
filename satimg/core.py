"""Базові операції над масивами, спільні для всіх методів.

Тут реалізовано вручну: згортку, гаусове ядро, швидку віконну суму через
інтегральне зображення та переведення у відтінки сірого.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def to_gray(img):
    """Яскравість (luma) за ITU-R BT.601: Y = 0.299 R + 0.587 G + 0.114 B."""
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def convolve2d(channel, kernel, pad_mode="reflect"):
    """Двовимірна кореляція каналу з ядром (як фільтрація у ковзному вікні).

    Межі доповнюються дзеркальним відображенням, тому розмір виходу
    дорівнює розміру входу.

    Args:
        channel: масив (H, W).
        kernel: масив (kh, kw) з непарними розмірами.
        pad_mode: режим доповнення меж для np.pad.
    """
    channel = np.asarray(channel, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Розміри ядра мають бути непарними")
    padded = np.pad(channel, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode=pad_mode)
    windows = sliding_window_view(padded, (kh, kw))
    # Сума добутків кожного вікна на ядро.
    return np.einsum("ijkl,kl->ij", windows, kernel)


def gaussian_kernel_1d(sigma, radius=None):
    """Одновимірне нормоване гаусове ядро з радіусом ceil(3*sigma)."""
    if sigma <= 0:
        raise ValueError("sigma має бути > 0")
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(channel, sigma, radius=None):
    """Гаусове згладжування як дві послідовні 1D-згортки (сепарабельність)."""
    k = gaussian_kernel_1d(sigma, radius)
    tmp = convolve2d(channel, k[np.newaxis, :])
    return convolve2d(tmp, k[:, np.newaxis])


def box_sum(arr, size):
    """Сума у квадратному вікні size x size навколо кожного пікселя.

    Працює через інтегральне зображення за O(1) на піксель. За межами
    зображення значення вважаються нулями. Підтримує додаткові провідні
    осі: обробляються дві останні осі масиву.
    """
    if size % 2 == 0:
        raise ValueError("Розмір вікна має бути непарним")
    r = size // 2
    arr = np.asarray(arr, dtype=np.float64)
    pad = [(0, 0)] * (arr.ndim - 2) + [(r + 1, r), (r + 1, r)]
    padded = np.pad(arr, pad)
    integral = padded.cumsum(axis=-2).cumsum(axis=-1)
    h, w = arr.shape[-2:]
    return (integral[..., size:size + h, size:size + w]
            - integral[..., 0:h, size:size + w]
            - integral[..., size:size + h, 0:w]
            + integral[..., 0:h, 0:w])
