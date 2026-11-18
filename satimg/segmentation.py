"""Сегментація кластеризацією k-means (алгоритм Ллойда з ініціалізацією k-means++)."""
import numpy as np

PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
    [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
], dtype=np.float64)


def kmeans(points, k, max_iter=30, tol=1e-4, seed=42, n_init=1):
    """Кластеризація k-means.

    1. Ініціалізація k-means++: перший центр — випадкова точка, кожен
       наступний обирається з імовірністю, пропорційною квадрату відстані
       до найближчого вже обраного центру.
    2. Кожна точка приєднується до найближчого центру (евклідова відстань).
    3. Кожен центр переноситься в середнє своїх точок.
    4. Кроки 2–3 повторюються, доки зсув центрів > tol або не вичерпано max_iter.

    k-means знаходить лише локальний мінімум, тому алгоритм запускається
    n_init разів з різних початкових центрів і обирається розв'язок з
    найменшою сумою квадратів відстаней до центрів (inertia).

    Args:
        points: масив (N, D).
        k: кількість кластерів (зменшується, якщо різних точок менше).
        seed: зерно генератора (результат відтворюваний).
        n_init: кількість перезапусків.

    Returns:
        (labels (N,), centers (k, D)).
    """
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    k = int(min(k, len(np.unique(points, axis=0))))
    best = None
    for _ in range(max(1, int(n_init))):
        labels, centers = _lloyd(points, k, max_iter, tol, rng)
        inertia = float(np.sum((points - centers[labels]) ** 2))
        if best is None or inertia < best[0]:
            best = (inertia, labels, centers)
    return best[1], best[2]


def _lloyd(points, k, max_iter, tol, rng):
    """Один запуск k-means: ініціалізація k-means++ та ітерації Ллойда."""
    n = points.shape[0]
    centers = [points[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers.append(points[rng.choice(n, p=probs)])
    centers = np.array(centers)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist = ((points[:, None, :] - centers[None]) ** 2).sum(-1)
        labels = np.argmin(dist, axis=1)
        new_centers = np.array([
            points[labels == c].mean(axis=0) if np.any(labels == c) else centers[c]
            for c in range(k)
        ])
        shift = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if shift <= tol:
            break
    return labels, centers


def standardize(features):
    """Z-нормалізація кожної ознаки (середнє 0, стандартне відхилення 1)."""
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    return (features - mean) / std


def segment_kmeans(feature_map, k=4, max_iter=30, tol=1e-4, seed=42, weights=None,
                   n_init=3):  # pylint: disable=too-many-arguments,too-many-positional-arguments
    """Сегментує зображення за картою ознак (H, W, D).

    Ознаки нормалізуються, множаться на ваги, кластеризуються k-means.
    Для стабільних кольорів кластери впорядковуються за зростанням
    першої ознаки центру.

    Returns:
        Масив міток (H, W).
    """
    h, w, d = feature_map.shape
    feats = standardize(feature_map.reshape(-1, d))
    if weights is not None:
        feats = feats * np.asarray(weights, dtype=np.float64)
    labels, centers = kmeans(feats, k, max_iter, tol, seed, n_init)
    order = np.argsort(centers[:, 0])
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[labels].reshape(h, w)


def labels_to_color(labels):
    """Розфарбовує карту міток кольорами з палітри."""
    return PALETTE[np.asarray(labels) % len(PALETTE)]
