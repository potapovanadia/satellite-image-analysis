"""Тести GLCM на маленьких зображеннях з ручним розрахунком."""
import numpy as np
import pytest

from satimg.texture import glcm, glcm_feature_maps, glcm_features, quantize

# Фрагмент 3x3 з рівнями 0..2.
Q = np.array([[0, 0, 1],
              [0, 1, 1],
              [2, 2, 1]])


def test_glcm_counts_by_hand():
    # Горизонтальні пари (зсув (0, 1)): (0,0) (0,1) (0,1) (1,1) (2,2) (2,1).
    # Симетрична матриця = лічильник + транспонований лічильник.
    expected = np.array([[2, 2, 0],
                         [2, 2, 1],
                         [0, 1, 2]], dtype=float)
    assert np.array_equal(glcm(Q, 3, (0, 1), normed=False), expected)


def test_glcm_contrast_by_hand():
    # contrast = (4 * 1^2 + 2 * 1^2) / 12 = 0.5
    feats = glcm_features(glcm(Q, 3, (0, 1)))
    assert feats["contrast"] == pytest.approx(0.5)


def test_checkerboard_features():
    # Чергування 0/1 по горизонталі: P[0,1] = P[1,0] = 0.5.
    q = np.tile([0, 1, 0, 1, 0], (5, 1))
    f = glcm_features(glcm(q, 2, (0, 1)))
    assert f["contrast"] == pytest.approx(1.0)
    assert f["energy"] == pytest.approx(0.5)
    assert f["homogeneity"] == pytest.approx(0.5)
    assert f["entropy"] == pytest.approx(1.0)


def test_constant_image_maps():
    maps = glcm_feature_maps(np.full((5, 5), 77.0), window=3, levels=4)
    contrast, energy, homogeneity, entropy = np.moveaxis(maps, -1, 0)
    assert np.allclose(contrast, 0) and np.allclose(energy, 1)
    assert np.allclose(homogeneity, 1) and np.allclose(entropy, 0)


def test_maps_distinguish_textures():
    img = np.zeros((10, 20))
    img[:, 10:] = np.tile([0, 255], (10, 5))  # права половина — смуги
    maps = glcm_feature_maps(img, window=3, levels=2, offsets=((0, 1),))
    assert maps[5, 2, 0] == 0          # гладка ліва частина: контраст 0
    assert maps[5, 15, 0] > 0.5        # смугаста права: високий контраст


def test_quantize():
    assert quantize(np.array([0, 31, 32, 255]), 8).tolist() == [0, 0, 1, 7]
