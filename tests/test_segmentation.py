"""Тести k-means на маленьких наборах точок і зображеннях."""
import numpy as np

from satimg.segmentation import kmeans, labels_to_color, segment_kmeans, standardize


def test_two_obvious_clusters():
    points = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
    labels, centers = kmeans(points, 2, seed=0)
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    # Центри = середні своїх точок: (0, 0.5) і (10, 10.5).
    assert sorted(map(tuple, np.round(centers, 3))) == [(0.0, 0.5), (10.0, 10.5)]


def test_reproducible_with_seed():
    rng = np.random.default_rng(3)
    pts = rng.random((50, 3))
    assert np.array_equal(kmeans(pts, 3, seed=7)[0], kmeans(pts, 3, seed=7)[0])


def test_k_reduced_when_too_few_distinct_points():
    labels, centers = kmeans(np.zeros((5, 2)), 3)
    assert len(centers) == 1 and np.all(labels == 0)


def test_segment_5x5_two_halves():
    # Ліва частина темна (V = 0.1), права світла (V = 0.9).
    feats = np.zeros((5, 5, 1))
    feats[:, :2, 0] = 0.1
    feats[:, 2:, 0] = 0.9
    labels = segment_kmeans(feats, k=2, seed=1)
    expected = np.zeros((5, 5), dtype=int)
    expected[:, 2:] = 1  # кластери впорядковані за першою ознакою: темний = 0
    assert np.array_equal(labels, expected)


def test_standardize_constant_feature():
    out = standardize(np.array([[1.0, 5.0], [3.0, 5.0]]))
    assert np.allclose(out[:, 0], [-1, 1]) and np.allclose(out[:, 1], 0)


def test_labels_to_color_shape():
    assert labels_to_color(np.zeros((2, 3), dtype=int)).shape == (2, 3, 3)
