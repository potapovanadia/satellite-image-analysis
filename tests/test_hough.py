"""Тести перетворення Хафа на зображеннях 5x5 з відомими прямими."""
import numpy as np

from satimg.hough import (find_lines, hough_accumulator, hough_lines, line_mask,
                          match_lines)

PEAKS = {"threshold": 0.5, "min_votes": 3, "num_peaks": 1, "nms_rho": 2, "nms_theta": 5}


def test_vertical_line_accumulator_by_hand():
    # Пряма x = 1: rho = x cos 0 + y sin 0 = 1 при theta = 0; голосують усі 5 пікселів.
    m = np.zeros((5, 5), dtype=bool)
    m[:, 1] = True
    acc, rhos, thetas = hough_accumulator(m)
    assert acc[np.where(rhos == 1)[0][0], np.where(thetas == 0)[0][0]] == 5


def test_horizontal_line_accumulator_by_hand():
    # Пряма y = 2: theta = -90 -> rho = y * sin(-90) = -2.
    m = np.zeros((5, 5), dtype=bool)
    m[2, :] = True
    acc, rhos, thetas = hough_accumulator(m)
    assert acc[np.where(rhos == -2)[0][0], np.where(thetas == -90)[0][0]] == 5


def test_found_lines_match_expected():
    # На відрізку з 5 пікселів кутова роздільність грубо +-7 градусів,
    # тому на 5x5 перевіряємо знайдену пряму з допуском 8 градусів і 1 піксель.
    m = np.zeros((5, 5), dtype=bool)
    m[:, 1] = True
    lines, _ = hough_lines(m, **PEAKS)
    assert lines[0][2] == 5
    assert match_lines(lines, [(1.0, 0.0)], angle_tol=8, rho_tol=1)[0] == 1
    diag, _ = hough_lines(np.eye(5, dtype=bool), **PEAKS)
    assert match_lines(diag, [(0.0, -45.0)], angle_tol=8, rho_tol=1)[0] == 1


def test_long_line_found_precisely():
    # На довгому відрізку (41 піксель) кут і rho знаходяться точно.
    m = np.zeros((41, 41), dtype=bool)
    m[:, 10] = True
    lines, _ = hough_lines(m, threshold=0.5, min_votes=20, num_peaks=1)
    assert lines[0][:2] == (10.0, 0.0)


def test_accumulator_votes_sum():
    m = np.zeros((5, 5), dtype=bool)
    m[1, 1] = m[3, 3] = True
    acc, _, thetas = hough_accumulator(m)
    # Кожен піксель голосує рівно один раз за кожен кут.
    assert acc.sum() == 2 * len(thetas)


def test_empty_image_has_no_lines():
    acc, rhos, thetas = hough_accumulator(np.zeros((5, 5), dtype=bool))
    assert not find_lines(acc, rhos, thetas)


def test_match_lines_with_angle_wrap():
    # (-2, -90) і (2, 89.5) — майже та сама горизонтальна пряма.
    found = [(2.0, 89.5, 10)]
    count, errors = match_lines(found, [(-2.0, -90.0)], angle_tol=2, rho_tol=3)
    assert count == 1 and errors[0][1] <= 0.5


def test_line_mask_draws_row():
    mask = line_mask((5, 5), [(-2.0, -90.0)])
    assert mask[2].all() and mask.sum() == 5
