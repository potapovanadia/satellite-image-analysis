"""Контурний аналіз: оператор Собеля та детектор Кенні (Canny, 1986)."""
import numpy as np

from .core import convolve2d, gaussian_blur

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = SOBEL_X.T


def sobel(gray):
    """Градієнти Собеля: повертає (gx, gy)."""
    return convolve2d(gray, SOBEL_X), convolve2d(gray, SOBEL_Y)


def non_maximum_suppression(magnitude, angle):
    """Придушення немаксимумів: залишає лише локальні максимуми вздовж градієнта.

    Напрям градієнта округлюється до 0, 45, 90 або 135 градусів. Піксель
    зберігається, якщо його модуль строго більший за сусіда «позаду» і не
    менший за сусіда «попереду». Несиметрична умова потрібна, щоб на
    ідеальній сходинці з двома однаковими максимумами лишалась лінія
    товщиною рівно 1 піксель.
    """
    h, w = magnitude.shape
    padded = np.pad(magnitude, 1)
    deg = (np.rad2deg(angle) + 180.0) % 180.0
    sector = (np.rint(deg / 45.0).astype(int)) % 4
    # Зсуви (dy, dx) сусіда «попереду» для кожного сектора.
    steps = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}
    keep = np.zeros((h, w), dtype=bool)
    for s, (dy, dx) in steps.items():
        ahead = padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        behind = padded[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        mask = sector == s
        keep |= mask & (magnitude > behind) & (magnitude >= ahead)
    return np.where(keep, magnitude, 0.0)


def hysteresis(strong, weak):
    """Гістерезис: слабкі контури залишаються, лише якщо зв'язані з сильними.

    Реалізовано як багаторазове розростання множини сильних пікселів у
    межах слабких (8-зв'язність), доки множина змінюється.
    """
    candidates = strong | weak
    result = strong.copy()
    h, w = result.shape
    while True:
        padded = np.pad(result, 1)
        grown = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                grown |= padded[dy:dy + h, dx:dx + w]
        grown &= candidates
        if np.array_equal(grown, result):
            return result
        result = grown


def canny(gray, sigma=1.4, low=0.1, high=0.25):
    """Детектор контурів Кенні.

    1. Гаусове згладжування (sigma) для зменшення шуму.
    2. Градієнти Собеля, модуль і напрям градієнта.
    3. Придушення немаксимумів — тонкі контури товщиною 1 піксель.
    4. Подвійний поріг: low і high задаються як частка від максимального
       модуля градієнта.
    5. Гістерезис — відкидаються слабкі контури, не з'єднані із сильними.

    Returns:
        Бінарна карта контурів (H, W), bool.
    """
    if not 0 <= low <= high:
        raise ValueError("Потрібно 0 <= low <= high")
    smoothed = gaussian_blur(gray, sigma) if sigma > 0 else np.asarray(gray, dtype=np.float64)
    gx, gy = sobel(smoothed)
    magnitude = np.hypot(gx, gy)
    thin = non_maximum_suppression(magnitude, np.arctan2(gy, gx))
    peak = thin.max()
    if peak == 0:
        return np.zeros_like(thin, dtype=bool)
    strong = thin >= high * peak
    weak = (thin >= low * peak) & ~strong
    return hysteresis(strong, weak)
