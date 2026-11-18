"""Тести введення/виведення та крайніх випадків (критерій C2)."""
import numpy as np
import pytest
from PIL import Image

from satimg.io_utils import ImageLoadError, list_images, load_image, save_image, to_uint8


def test_roundtrip_rgb(tmp_path):
    img = np.arange(5 * 5 * 3).reshape(5, 5, 3).astype(float)
    save_image(img, tmp_path / "a.png")
    assert np.array_equal(load_image(tmp_path / "a.png"), img)


def test_grayscale_becomes_three_channels(tmp_path):
    Image.fromarray(np.full((4, 6), 100, dtype=np.uint8), mode="L").save(tmp_path / "g.png")
    img = load_image(tmp_path / "g.png")
    assert img.shape == (4, 6, 3)
    assert np.all(img == 100)


def test_rgba_is_converted(tmp_path):
    Image.new("RGBA", (5, 5), (10, 20, 30, 128)).save(tmp_path / "rgba.png")
    assert load_image(tmp_path / "rgba.png").shape == (5, 5, 3)


def test_16bit_is_scaled(tmp_path):
    arr = np.array([[0, 65535], [32768, 65535]], dtype=np.uint16)
    Image.fromarray(arr).save(tmp_path / "w.png")
    img = load_image(tmp_path / "w.png", min_size=1)
    assert img.max() == pytest.approx(255.0)


def test_empty_file(tmp_path):
    (tmp_path / "empty.png").write_bytes(b"")
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "empty.png")


def test_corrupted_file(tmp_path):
    (tmp_path / "bad.tif").write_bytes(b"this is not an image")
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "bad.tif")


def test_missing_file(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.png")


def test_too_small(tmp_path):
    Image.new("RGB", (2, 2)).save(tmp_path / "tiny.png")
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "tiny.png", min_size=3)


def test_to_uint8_clips_and_handles_bool():
    assert to_uint8(np.array([-5.0, 300.0])).tolist() == [0, 255]
    assert to_uint8(np.array([True, False])).tolist() == [255, 0]


def test_list_images_filters_extensions(tmp_path):
    (tmp_path / "cls").mkdir()
    Image.new("RGB", (3, 3)).save(tmp_path / "cls" / "x.png")
    (tmp_path / "cls" / "notes.txt").write_text("hi")
    assert [p.name for p in list_images(tmp_path)] == ["x.png"]
    assert not list_images(tmp_path / "nothing")
