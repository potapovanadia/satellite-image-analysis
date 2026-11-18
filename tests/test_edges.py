"""Тести оператора Собеля і детектора Кенні."""
import numpy as np
import pytest

from satimg.edges import canny, hysteresis, non_maximum_suppression, sobel


def test_sobel_by_hand():
    # Вертикальна сходинка 0 | 1 у правому стовпці.
    img = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
    gx, gy = sobel(img)
    # Центр: (-1*0 + 1*1) + (-2*0 + 2*1) + (-1*0 + 1*1) = 4; по вертикалі змін немає.
    assert gx[1, 1] == 4 and gy[1, 1] == 0


def test_canny_step_gives_single_thin_line():
    # 7x7: стовпці 0..3 = 0, 4..6 = 255. Межа між стовпцями 3 і 4.
    img = np.zeros((7, 7))
    img[:, 4:] = 255
    edges = canny(img, sigma=0.5, low=0.1, high=0.3)
    expected = np.zeros((7, 7), dtype=bool)
    expected[:, 3] = True  # рівно одна лінія товщиною 1 піксель
    assert np.array_equal(edges, expected)


def test_canny_constant_image_has_no_edges():
    assert not canny(np.full((5, 5), 90.0)).any()


def test_nms_keeps_only_ridge():
    mag = np.array([[0, 1, 3, 1, 0]] * 3, dtype=float)
    out = non_maximum_suppression(mag, np.zeros_like(mag))  # градієнт горизонтальний
    assert np.array_equal(out[1] > 0, [False, False, True, False, False])


def test_hysteresis_by_hand():
    strong = np.array([[1, 0, 0, 0, 0]], dtype=bool)
    weak = np.array([[0, 1, 1, 0, 1]], dtype=bool)
    # Слабкі пікселі 1, 2 з'єднані з сильним -> лишаються; піксель 4 ізольований.
    assert hysteresis(strong, weak).tolist() == [[True, True, True, False, False]]


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        canny(np.zeros((5, 5)), low=0.5, high=0.2)
