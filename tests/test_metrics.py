"""Тести метрик якості на маленьких зображеннях з ручним розрахунком."""
import numpy as np
import pytest

from satimg import metrics


def test_mse_and_psnr_by_hand():
    a = np.array([[0, 0], [0, 0]], dtype=float)
    b = np.array([[2, 0], [0, 2]], dtype=float)
    # MSE = (4 + 0 + 0 + 4) / 4 = 2
    assert metrics.mse(a, b) == 2.0
    # PSNR = 10 log10(255^2 / 2) = 45.12 дБ
    assert metrics.psnr(a, b) == pytest.approx(10 * np.log10(255 ** 2 / 2))


def test_psnr_identical_is_inf():
    a = np.ones((3, 3))
    assert metrics.psnr(a, a) == float("inf")


def test_shape_mismatch():
    with pytest.raises(ValueError):
        metrics.mse(np.ones((3, 3)), np.ones((3, 4)))


def test_ssim_identical_is_one_and_noise_lowers_it():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (16, 16)).astype(float)
    assert metrics.ssim(img, img) == pytest.approx(1.0)
    noisy = np.clip(img + rng.normal(0, 40, img.shape), 0, 255)
    assert metrics.ssim(img, noisy) < 0.9


def test_iou_ignores_label_numbers():
    ref = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    pred = 1 - ref  # ті самі області, але мітки переставлені
    assert metrics.mean_iou(pred, ref) == 1.0
    assert metrics.pixel_accuracy(pred, ref) == 1.0


def test_iou_by_hand():
    ref = np.array([[0, 0, 1, 1]])
    pred = np.array([[0, 0, 0, 1]])
    # клас 0: перетин 2, об'єднання 3 -> 2/3; клас 1: 1/2. Середнє = 7/12.
    assert metrics.mean_iou(pred, ref) == pytest.approx(7 / 12)


def test_edge_f1_with_tolerance():
    ref = np.zeros((5, 5), dtype=bool)
    ref[:, 2] = True
    shifted = np.zeros((5, 5), dtype=bool)
    shifted[:, 3] = True
    assert metrics.edge_f1(shifted, ref, tolerance=1) == 1.0
    assert metrics.edge_f1(shifted, ref, tolerance=0) == 0.0
