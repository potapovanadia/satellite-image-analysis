"""Тести фільтра Кувахари на зображеннях 5x5 з ручним розрахунком."""
import numpy as np
import pytest

from satimg.filters import kuwahara


def test_edge_is_preserved():
    # Сходинка: ліві 2 стовпці = 0, праві 3 = 100.
    # Для кожного пікселя знайдеться квадрант 2x2 повністю по один бік межі
    # (дисперсія 0), тому результат точно дорівнює входу — межа не розмита.
    img = np.zeros((5, 5))
    img[:, 2:] = 100
    assert np.array_equal(kuwahara(img, 3), img)


def test_isolated_noise_pixel():
    # Константа 10 з одним викидом 250 у центрі.
    # Вікно 3x3 -> 4 квадранти 2x2; кожен містить центр, тобто {250, 10, 10, 10}.
    # Усі дисперсії однакові, середнє = (250 + 30) / 4 = 70.
    img = np.full((5, 5), 10.0)
    img[2, 2] = 250
    out = kuwahara(img, 3)
    assert out[2, 2] == pytest.approx(70.0)
    # Сусід (2, 1): його лівий верхній квадрант {10,10,10,10} однорідний -> 10.
    assert out[2, 1] == pytest.approx(10.0)


def test_color_image_shape_and_constant():
    img = np.full((5, 5, 3), 42.0)
    out = kuwahara(img, 5)
    assert out.shape == img.shape
    assert np.allclose(out, 42.0)


def test_reduces_gaussian_noise():
    rng = np.random.default_rng(0)
    clean = np.full((32, 32), 120.0)
    noisy = clean + rng.normal(0, 20, clean.shape)
    out = kuwahara(noisy, 5)
    assert np.mean((out - clean) ** 2) < np.mean((noisy - clean) ** 2)


@pytest.mark.parametrize("window", [2, 4, 1])
def test_invalid_window(window):
    with pytest.raises(ValueError):
        kuwahara(np.zeros((5, 5)), window)
