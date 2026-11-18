"""Перетворення Хафа для пошуку прямих (Hough, 1962; Duda & Hart, 1972)."""
import numpy as np


def hough_accumulator(mask, theta_step=1.0, rho_step=1.0):
    """Будує акумулятор Хафа.

    Пряма задається нормальним рівнянням rho = x cos(theta) + y sin(theta),
    theta у [-90, 90) градусів. Кожен піксель контуру «голосує» за всі
    прямі (rho, theta), що через нього проходять.

    Returns:
        (accumulator (n_rho, n_theta), rhos, thetas_deg).
    """
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    thetas = np.arange(-90.0, 90.0, theta_step)
    diag = np.hypot(h, w)
    rhos = np.arange(-np.ceil(diag), np.ceil(diag) + rho_step, rho_step)
    acc = np.zeros((len(rhos), len(thetas)), dtype=np.int64)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return acc, rhos, thetas
    rad = np.deg2rad(thetas)
    rho_vals = xs[:, None] * np.cos(rad)[None, :] + ys[:, None] * np.sin(rad)[None, :]
    rho_idx = np.rint((rho_vals - rhos[0]) / rho_step).astype(np.int64)
    theta_idx = np.broadcast_to(np.arange(len(thetas)), rho_idx.shape)
    np.add.at(acc, (rho_idx.ravel(), theta_idx.ravel()), 1)
    return acc, rhos, thetas


def merge_rho_bins(acc, rho_window=3):
    """Сумує голоси сусідніх комірок rho (вікно rho_window) при тому ж theta.

    Реальний контур рідко лежить точно на одній дискретній прямій: через
    округлення голоси однієї лінії розподіляються між сусідніми rho.
    Сума по вікну збирає їх разом, і кількість голосів знову приблизно
    дорівнює довжині лінії в пікселях.
    """
    if rho_window <= 1:
        return acc.astype(np.float64)
    r = rho_window // 2
    padded = np.pad(acc.astype(np.float64), ((r, r), (0, 0)))
    return sum(padded[k:k + acc.shape[0]] for k in range(rho_window))


def find_lines(acc, rhos, thetas, threshold=0.5, min_votes=30, num_peaks=10,
               nms_rho=10, nms_theta=10, rho_window=3):
    """Пошук піків акумулятора з придушенням сусідніх піків.

    Спершу голоси сусідніх rho об'єднуються (merge_rho_bins). Пік
    приймається, якщо голосів >= max(min_votes, threshold * максимум).
    Після вибору піку його околиця (nms_rho x nms_theta комірок, з
    урахуванням циклічності theta) обнуляється, щоб одна пряма не давала
    кілька майже однакових ліній. rho піку уточнюється як зважене
    середнє по вікну rho_window.

    Returns:
        Список кортежів (rho, theta_deg, votes), відсортований за голосами.
    """
    # Невелика добавка «сирих» голосів розв'язує нічиї на користь комірки,
    # де лінія лежить точно (важливо для коротких ліній).
    work = merge_rho_bins(acc, rho_window) + 1e-3 * acc
    if work.max() <= 0:
        return []
    limit = max(min_votes, threshold * work.max())
    n_rho, n_theta = work.shape
    lines = []
    while len(lines) < num_peaks:
        idx = np.argmax(work)
        r, t = np.unravel_index(idx, work.shape)
        votes = work[r, t]
        if votes < limit:
            break
        lo, hi = max(0, r - rho_window // 2), min(n_rho, r + rho_window // 2 + 1)
        col = acc[lo:hi, t].astype(np.float64)
        rho = float(np.sum(rhos[lo:hi] * col) / col.sum()) if col.sum() else float(rhos[r])
        lines.append((round(rho, 2), float(thetas[t]), int(votes)))
        r0, r1 = max(0, r - nms_rho), min(n_rho, r + nms_rho + 1)
        for dt in range(-nms_theta, nms_theta + 1):
            tt = t + dt
            if 0 <= tt < n_theta:
                work[r0:r1, tt] = 0
            else:
                # theta циклічна: (rho, theta) = (-rho, theta -+ 180).
                tt %= n_theta
                rr = n_rho - 1 - r
                work[max(0, rr - nms_rho):min(n_rho, rr + nms_rho + 1), tt] = 0
    return lines


def hough_lines(mask, theta_step=1.0, rho_step=1.0, **peak_params):
    """Повний пошук прямих: акумулятор + вибір піків."""
    acc, rhos, thetas = hough_accumulator(mask, theta_step, rho_step)
    return find_lines(acc, rhos, thetas, **peak_params), acc


def draw_lines(img, lines, color=(255, 0, 0), thickness=1):
    """Малює прямі (rho, theta) поверх копії зображення."""
    out = np.array(img, dtype=np.float64, copy=True)
    if out.ndim == 2:
        out = np.stack([out] * 3, axis=-1)
    h, w = out.shape[:2]
    mask = line_mask((h, w), lines, thickness)
    out[mask] = color
    return out


def line_mask(shape, lines, thickness=1):
    """Бінарна маска пікселів, через які проходять прямі (rho, theta, ...)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for line in lines:
        rho, theta = line[0], np.deg2rad(line[1])
        c, s = np.cos(theta), np.sin(theta)
        if abs(s) >= abs(c):  # ближче до горизонтальної: y як функція x
            xs = np.arange(w)
            ys = np.rint((rho - xs * c) / s).astype(int)
        else:
            ys = np.arange(h)
            xs = np.rint((rho - ys * s) / c).astype(int)
        for off in range(-(thickness // 2), thickness // 2 + 1):
            yy, xx = (ys + off, xs) if abs(s) >= abs(c) else (ys, xs + off)
            ok = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
            mask[yy[ok], xx[ok]] = True
    return mask


def line_distance(line_a, line_b):
    """Різниця (|d rho|, |d theta|) двох прямих з урахуванням циклічності theta."""
    ra, ta = line_a[0], line_a[1]
    rb, tb = line_b[0], line_b[1]
    direct = (abs(ra - rb), abs(ta - tb))
    wrapped = (abs(ra + rb), 180.0 - abs(ta - tb))
    return min(direct, wrapped, key=lambda d: d[1] + d[0] / 10.0)


def match_lines(found, expected, angle_tol=2.0, rho_tol=3.0):
    """Скільки очікуваних прямих знайдено з допуском за кутом і відстанню.

    Кожна знайдена пряма може відповідати лише одній очікуваній.

    Returns:
        (matched_count, list of (d_rho, d_theta) для знайдених пар).
    """
    used = set()
    errors = []
    for exp in expected:
        best, best_err = None, None
        for idx, cand in enumerate(found):
            if idx in used:
                continue
            d_rho, d_theta = line_distance(cand, exp)
            if d_rho <= rho_tol and d_theta <= angle_tol:
                if best_err is None or d_theta + d_rho / 10 < best_err[1] + best_err[0] / 10:
                    best, best_err = idx, (d_rho, d_theta)
        if best is not None:
            used.add(best)
            errors.append(best_err)
    return len(errors), errors
