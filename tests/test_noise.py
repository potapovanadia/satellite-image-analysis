"""Тести моделювання шуму."""
import numpy as np
import pytest

from satimg.noise import apply_noise


def test_noise_is_reproducible():
    img = np.full((8, 8, 3), 128.0)
    spec = {"type": "gaussian", "sigma": 20}
    assert np.array_equal(apply_noise(img, spec, 1), apply_noise(img, spec, 1))


def test_salt_pepper_amount():
    img = np.full((100, 100, 3), 128.0)
    noisy = apply_noise(img, {"type": "salt_pepper", "amount": 0.05}, 0)
    changed = np.mean(np.any(noisy != 128.0, axis=-1))
    assert 0.04 < changed < 0.06
    assert set(np.unique(noisy)) <= {0.0, 128.0, 255.0}


def test_none_and_unknown():
    img = np.ones((3, 3))
    assert np.array_equal(apply_noise(img, {"type": "none"}, 0), img)
    with pytest.raises(ValueError):
        apply_noise(img, {"type": "rain"}, 0)
