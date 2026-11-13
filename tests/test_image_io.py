import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage

from gravity_studio.image_io import load_image_pixels


def test_loads_actual_image_pixels_and_composites_transparency(tmp_path):
    image = QImage(3, 2, QImage.Format.Format_ARGB32)
    image.fill(QColor(255, 0, 0, 0))
    image.setPixelColor(1, 1, QColor(0, 0, 255))
    path = tmp_path / "sample.png"
    assert image.save(str(path))
    pixels = load_image_pixels(path)
    assert pixels.shape == (2, 3, 3)
    np.testing.assert_array_equal(pixels[0, 0], [0, 0, 0])
    np.testing.assert_array_equal(pixels[1, 1], [0, 0, 255])


@pytest.mark.parametrize("dimensions", [(1000, 500), (5000, 5000)])
def test_large_images_keep_original_pixel_dimensions(tmp_path, dimensions):
    width, height = dimensions
    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(QColor("red"))
    path = tmp_path / "wide.png"
    assert image.save(str(path))
    pixels = load_image_pixels(path)
    assert pixels.shape == (height, width, 3)
    np.testing.assert_array_equal(pixels[30, 30], [255, 0, 0])


def test_oversized_file_is_rejected_before_decoding(tmp_path):
    path = tmp_path / "oversized.png"
    with path.open("wb") as handle:
        handle.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="32 MiB"):
        load_image_pixels(path)


def test_corrupt_or_missing_images_fail_explicitly(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image_pixels(bad)
    with pytest.raises((ValueError, OSError)):
        load_image_pixels(tmp_path / "missing.png")
