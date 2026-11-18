"""Тести морфологічних операцій на бінарних зображеннях 5x5."""
import numpy as np
import pytest

from satimg.morphology import (closing, connected_components, dilate, erode,
                               morphological_gradient, opening, remove_small_objects,
                               structuring_element)


def point():
          """Бінарне зображення 5x5 з одним білим пікселем у центрі."""
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    return m


def test_dilate_point_square_and_cross():
    sq = dilate(point(), 3, "square")
    assert sq.sum() == 9 and sq[1:4, 1:4].all()
    cross = dilate(point(), 3, "cross")
    assert cross.sum() == 5 and cross[2, 1] and not cross[1, 1]


def test_erode_block_back_to_point():
    block = np.zeros((5, 5), dtype=bool)
    block[1:4, 1:4] = True
    assert np.array_equal(erode(block, 3), point())


def test_closing_fills_one_pixel_gap():
    line = np.zeros((5, 5), dtype=bool)
    line[2, :] = True
    line[2, 2] = False  # розрив у лінії
    assert closing(line, 3)[2, 2]


def test_opening_removes_isolated_point():
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0] = True
    assert not opening(m, 3).any()


def test_components_and_small_object_removal():
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0:3] = True    # компонента з 3 пікселів
    m[4, 4] = True      # окремий піксель
    labels, count = connected_components(m)
    assert count == 2 and labels[0, 0] == labels[0, 2] != labels[4, 4]
    cleaned = remove_small_objects(m, min_size=2)
    assert cleaned[0, 0:3].all() and not cleaned[4, 4]


def test_diagonal_connectivity():
    m = np.eye(3, dtype=bool)
    assert connected_components(m, 8)[1] == 1
    assert connected_components(m, 4)[1] == 3


def test_gradient_is_outline():
    block = np.zeros((5, 5), dtype=bool)
    block[1:4, 1:4] = True
    grad = morphological_gradient(block, 3)
    assert not grad[2, 2] and grad[0, 0] and grad[1, 1]


def test_invalid_element():
    with pytest.raises(ValueError):
        structuring_element(2)
