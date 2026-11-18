"""Метрики якості обробки: MSE, PSNR, SSIM, IoU, точність, F1 для контурів."""
from itertools import permutations

import numpy as np

from .core import gaussian_blur, to_gray


def mse(reference, processed):
    """Середньоквадратична помилка: MSE = (1/N) * sum (I - I')^2."""
    reference = np.asarray(reference, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    if reference.shape != processed.shape:
        raise ValueError("Розміри зображень не збігаються")
    return float(np.mean((reference - processed) ** 2))


def psnr(reference, processed, max_value=255.0):
    """Пікове відношення сигнал/шум: PSNR = 10 * log10(MAX^2 / MSE), дБ.

    Для ідентичних зображень (MSE = 0) повертає inf.
    """
    err = mse(reference, processed)
    if err == 0:
        return float("inf")
    return float(10.0 * np.log10(max_value ** 2 / err))


def ssim(reference, processed, max_value=255.0, sigma=1.5):
    """Індекс структурної подібності (Wang et al., 2004).

    SSIM = ((2 mu_x mu_y + C1)(2 sigma_xy + C2)) /
           ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)),
    де локальні статистики рахуються з гаусовим вікном (sigma = 1.5,
    радіус 5, тобто вікно 11x11). Кольорові зображення порівнюються
    за яскравістю. Повертає середнє значення SSIM по зображенню.
    """
    x = to_gray(reference)
    y = to_gray(processed)
    if x.shape != y.shape:
        raise ValueError("Розміри зображень не збігаються")
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    radius = min(5, (min(x.shape) - 1) // 2) or 1

    def blur(a):
        return gaussian_blur(a, sigma, radius)

    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x ** 2
    syy = blur(y * y) - mu_y ** 2
    sxy = blur(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def match_labels(pred, ref):
    """Перенумеровує мітки pred так, щоб вони найкраще збігалися з ref.

    Номери кластерів k-means довільні, тому перед порівнянням шукаємо
    відповідність міток, яка дає найбільшу кількість збігів пікселів.
    Для <= 7 міток перебираються всі перестановки, інакше — жадібно.
    """
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    p_labels = list(np.unique(pred))
    r_labels = list(np.unique(ref))
    overlap = np.array([[np.sum((pred == p) & (ref == r)) for r in r_labels]
                        for p in p_labels])
    mapping = {}
    if len(p_labels) <= 7 and len(r_labels) <= 7:
        best, best_score = None, -1
        n = max(len(p_labels), len(r_labels))
        for perm in permutations(range(n), len(p_labels)):
            score = sum(overlap[i, j] for i, j in enumerate(perm) if j < len(r_labels))
            if score > best_score:
                best, best_score = perm, score
        for i, j in enumerate(best):
            mapping[p_labels[i]] = r_labels[j] if j < len(r_labels) else -1 - i
    else:
        used = set()
        for i in np.argsort(-overlap.max(axis=1)):
            order = [j for j in np.argsort(-overlap[i]) if j not in used]
            j = order[0] if order else None
            mapping[p_labels[i]] = r_labels[j] if j is not None else -1 - i
            if j is not None:
                used.add(j)
    out = np.empty_like(pred)
    for p, r in mapping.items():
        out[pred == p] = r
    return out


def mean_iou(pred, ref):
    """Середній IoU (Intersection over Union) по класах еталону після узгодження міток."""
    matched = match_labels(pred, ref)
    ious = []
    for label in np.unique(ref):
        inter = np.sum((matched == label) & (ref == label))
        union = np.sum((matched == label) | (ref == label))
        ious.append(inter / union if union else 1.0)
    return float(np.mean(ious))


def pixel_accuracy(pred, ref):
    """Частка пікселів з правильною міткою після узгодження міток."""
    matched = match_labels(pred, ref)
    return float(np.mean(matched == np.asarray(ref)))


def edge_f1(pred, ref, tolerance=1):
    """F1-міра збігу двох бінарних карт контурів з допуском tolerance пікселів.

    Точність — частка пікселів pred, що лежать поруч із контуром ref;
    повнота — частка пікселів ref, поруч з якими є контур pred.
    """
    pred = np.asarray(pred, dtype=bool)
    ref = np.asarray(ref, dtype=bool)
    if not pred.any() and not ref.any():
        return 1.0
    if not pred.any() or not ref.any():
        return 0.0
    size = 2 * tolerance + 1

    def grow(mask):
        padded = np.pad(mask, tolerance)
        out = np.zeros_like(mask)
        for dy in range(size):
            for dx in range(size):
                out |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        return out

    precision = np.sum(pred & grow(ref)) / np.sum(pred)
    recall = np.sum(ref & grow(pred)) / np.sum(ref)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def contrast_std(img):
    """Контраст як стандартне відхилення яскравості."""
    return float(np.std(to_gray(img)))
