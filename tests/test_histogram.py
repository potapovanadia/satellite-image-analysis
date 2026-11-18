"""Тести гістограмної корекції CLAHE з ручним розрахунком."""
import numpy as np
import pytest

from satimg.histogram import clahe, clip_histogram, histogram, tile_mapping


def test_histogram_counts():
    h = histogram(np.array([[0, 0, 5], [255, 5, 5]]))
    assert h[0] == 2 and h[5] == 3 and h[255] == 1 and h.sum() == 6


def test_equalization_without_clipping():
    # 4x4: половина пікселів 0, половина 255. CDF(0) = 8, CDF(255) = 16.
    # map(0) = round(8 / 16 * 255) = 128, map(255) = 255.
    img = np.zeros((4, 4))
    img[:, 2:] = 255
    out = clahe(img, tiles=(1, 1), clip_limit=None)
    assert np.all(out[:, :2] == 128) and np.all(out[:, 2:] == 255)


def test_clipping_limits_contrast_by_hand():
    # Константа 100 у плитці 4x4 (N = 16). Ліміт = 2 * 16 / 256 = 0.125 -> не менше 1.
    # Стовпець 100: 16 -> 1, надлишок 15 розподіляється по 256 бінах (15/256 кожному).
    # CDF(100) = 101 * 15/256 + 1 = 6.918; map = round(6.918 / 16 * 255) = 110.
    tile = np.full((4, 4), 100)
    assert tile_mapping(tile, clip_limit=2.0)[100] == 110
    assert tile_mapping(tile, clip_limit=None)[100] == 255


def test_clip_histogram_keeps_total():
    hist = np.array([10.0, 0.0, 2.0, 0.0])
    clipped = clip_histogram(hist, 4.0)
    assert clipped.sum() == pytest.approx(hist.sum())
    assert clipped.max() <= 4.0 + 6.0 / 4


def test_low_contrast_is_enhanced():
    rng = np.random.default_rng(0)
    img = 120 + rng.random((32, 32)) * 20  # діапазон лише 120..140
    out = clahe(img, tiles=(4, 4), clip_limit=2.0)
    assert out.std() > 1.5 * img.std()
    assert out.shape == img.shape and 0 <= out.min() and out.max() <= 255


def test_tiles_larger_than_image():
    out = clahe(np.full((3, 3), 50.0), tiles=(8, 8))
    assert out.shape == (3, 3)
