from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra import numpy as hnp
from PySide6.QtGui import QImage

from gravity_studio.image_field import ImageField
from gravity_studio.image_io import load_image_pixels


@settings(max_examples=35)
@given(
    rgba=hnp.arrays(
        np.uint8,
        hnp.array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=10).map(
            lambda shape: (*shape, 4)
        ),
    )
)
def test_png_round_trip_composites_alpha_onto_black(rgba):
    height, width, _ = rgba.shape
    source = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "rgba.png"
        assert source.save(str(path))
        pixels = load_image_pixels(path)

    expected = np.rint(rgba[:, :, :3].astype(float) * rgba[:, :, 3:] / 255).astype(np.uint8)
    np.testing.assert_array_equal(pixels, expected)


def test_large_png_keeps_full_resolution_through_palette_extraction(tmp_path):
    # 4,196,352 pixels crosses the app's former 4 Mi-pixel ceiling.
    width, height = 2048, 2049
    source = QImage(width, height, QImage.Format.Format_RGB32)
    source.fill(0xFFFF0000)
    source.setPixel(width - 1, height - 1, 0xFF0000FF)
    path = tmp_path / "large.png"
    assert source.save(str(path))

    field = ImageField.from_pixels(load_image_pixels(path), 2)

    assert field.labels.shape == (height, width)
    assert field.colors[field.labels[0, 0]] == (255, 0, 0)
    assert field.colors[field.labels[-1, -1]] == (0, 0, 255)


def test_files_over_import_limit_are_rejected(tmp_path):
    source = QImage(1, 1, QImage.Format.Format_RGB32)
    source.fill(0xFFFF0000)
    path = tmp_path / "padded.png"
    assert source.save(str(path))
    with path.open("r+b") as handle:
        handle.truncate(32 * 1024 * 1024)
    np.testing.assert_array_equal(load_image_pixels(path), [[[255, 0, 0]]])

    with path.open("r+b") as handle:
        handle.truncate(32 * 1024 * 1024 + 1)
    with pytest.raises(ValueError):
        load_image_pixels(path)
