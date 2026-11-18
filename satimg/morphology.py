"""Морфологічна обробка бінарних зображень."""
from collections import deque

import numpy as np


def structuring_element(size=3, shape="square"):
    """Структурний елемент size x size: 'square' (квадрат) або 'cross' (хрест)."""
    if size < 1 or size % 2 == 0:
        raise ValueError("size має бути непарним і >= 1")
    if shape == "square":
        return np.ones((size, size), dtype=bool)
    if shape == "cross":
        se = np.zeros((size, size), dtype=bool)
        se[size // 2, :] = True
        se[:, size // 2] = True
        return se
    raise ValueError(f"Невідома форма структурного елемента: {shape}")


def _shifts(mask, se, fill):
    """Генерує зсунуті копії mask для кожного активного елемента se."""
    r = se.shape[0] // 2
    h, w = mask.shape
    padded = np.pad(mask, r, constant_values=fill)
    for dy, dx in zip(*np.nonzero(se)):
        yield padded[dy:dy + h, dx:dx + w]


def dilate(mask, size=3, shape="square"):
    """Дилатація: піксель = 1, якщо хоч один піксель під елементом дорівнює 1."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask)
    for shifted in _shifts(mask, structuring_element(size, shape), False):
        out |= shifted
    return out


def erode(mask, size=3, shape="square"):
    """Ерозія: піксель = 1, лише якщо всі пікселі під елементом дорівнюють 1.

    За межами зображення вважаємо 1, щоб об'єкти біля краю не «з'їдались».
    """
    mask = np.asarray(mask, dtype=bool)
    out = np.ones_like(mask)
    for shifted in _shifts(mask, structuring_element(size, shape), True):
        out &= shifted
    return out


def opening(mask, size=3, shape="square"):
    """Відкриття = ерозія, потім дилатація (прибирає дрібні виступи й точки)."""
    return dilate(erode(mask, size, shape), size, shape)


def closing(mask, size=3, shape="square"):
    """Закриття = дилатація, потім ерозія (заповнює дрібні розриви контурів)."""
    return erode(dilate(mask, size, shape), size, shape)


def connected_components(mask, connectivity=8):
    """Позначення зв'язних компонент обходом у ширину (BFS).

    Returns:
        (labels, count): labels — масив (H, W), 0 для фону, 1..count для об'єктів.
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int64)
    if connectivity == 8:
        neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    elif connectivity == 4:
        neighbours = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    else:
        raise ValueError("connectivity має бути 4 або 8")
    count = 0
    for sy, sx in zip(*np.nonzero(mask)):
        if labels[sy, sx]:
            continue
        count += 1
        labels[sy, sx] = count
        queue = deque([(sy, sx)])
        while queue:
            y, x = queue.popleft()
            for dy, dx in neighbours:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = count
                    queue.append((ny, nx))
    return labels, count


def remove_small_objects(mask, min_size=20, connectivity=8):
    """Видаляє зв'язні компоненти, площа яких менша за min_size пікселів."""
    labels, count = connected_components(mask, connectivity)
    if count == 0:
        return np.zeros_like(np.asarray(mask, dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = sizes >= min_size
    keep[0] = False
    return keep[labels]


def morphological_gradient(mask, size=3, shape="square"):
    """Морфологічний градієнт = дилатація XOR ерозія (контур об'єктів)."""
    return dilate(mask, size, shape) ^ erode(mask, size, shape)
