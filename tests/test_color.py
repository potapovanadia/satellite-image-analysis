"""Тести перетворення RGB -> HSV на пікселях з відомою відповіддю."""
import numpy as np
import pytest

from satimg.color import hsv_to_display, hsv_to_rgb, rgb_to_hsv


@pytest.mark.parametrize("rgb, hsv", [
    ((255, 0, 0), (0.0, 1.0, 1.0)),       # червоний
    ((0, 255, 0), (120.0, 1.0, 1.0)),     # зелений
    ((0, 0, 255), (240.0, 1.0, 1.0)),     # синій
    ((255, 255, 0), (60.0, 1.0, 1.0)),    # жовтий
    ((0, 0, 0), (0.0, 0.0, 0.0)),         # чорний
    ((51, 102, 102), (180.0, 0.5, 0.4)),  # бірюзовий: max=102 -> V=0.4, S=(102-51)/102
])
def test_known_colors(rgb, hsv):
    img = np.array([[rgb]], dtype=float)
    assert np.allclose(rgb_to_hsv(img)[0, 0], hsv, atol=1e-6)


def test_gray_has_zero_saturation():
    out = rgb_to_hsv(np.full((3, 3, 3), 128.0))
    assert np.all(out[..., 1] == 0)
    assert np.allclose(out[..., 2], 128 / 255)


def test_roundtrip():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (5, 5, 3)).astype(float)
    assert np.allclose(hsv_to_rgb(rgb_to_hsv(img)), img, atol=1e-6)


def test_display_range():
    disp = hsv_to_display(rgb_to_hsv(np.array([[[0, 0, 255]]], dtype=float)))
    assert np.allclose(disp[0, 0], [240 / 360 * 255, 255, 255])
