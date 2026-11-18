"""Тести базових операцій (згортка, гаусове ядро, віконна сума)."""
import numpy as np
import pytest

from satimg.core import box_sum, convolve2d, gaussian_kernel_1d, to_gray


def test_convolve_matches_hand_calculation():
    # Ядро бере різницю «правий сусід - лівий сусід».
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    kernel = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]], dtype=float)
    out = convolve2d(img, kernel)
    # Центр: 6 - 4 = 2. Лівий край з дзеркалом: сусід зліва = 2, справа = 2 -> 0.
    assert out[1, 1] == 2
    assert out[1, 0] == 0


def test_identity_kernel():
    img = np.random.default_rng(0).random((5, 5))
    kernel = np.zeros((3, 3))
    kernel[1, 1] = 1
    assert np.allclose(convolve2d(img, kernel), img)


def test_even_kernel_rejected():
    with pytest.raises(ValueError):
        convolve2d(np.ones((3, 3)), np.ones((2, 2)))


def test_gaussian_kernel_normalized_and_symmetric():
    k = gaussian_kernel_1d(1.0)
    assert k.sum() == pytest.approx(1.0)
    assert np.allclose(k, k[::-1])
    assert len(k) == 7  # радіус ceil(3 * 1) = 3


def test_box_sum_3x3():
    img = np.arange(1, 10, dtype=float).reshape(3, 3)
    out = box_sum(img, 3)
    assert out[1, 1] == 45          # сума всіх 9 чисел
    assert out[0, 0] == 1 + 2 + 4 + 5  # кут: поза межами нулі


def test_to_gray_weights():
    img = np.zeros((1, 1, 3))
    img[0, 0] = [100, 100, 100]
    assert to_gray(img)[0, 0] == pytest.approx(100.0)
