import numpy as np
import pytest

from gravity_studio.image_field import ImageField
from gravity_studio.model import Simulation


def test_simulation_routes_group_ids_above_int16_without_wrapping():
    group = 32769
    indices = np.arange(group + 1, dtype=np.uint32)
    palette = np.column_stack((indices >> 16, (indices >> 8) & 255, indices & 255)).astype(np.uint8)
    field = ImageField(palette, np.array([[0, group]], dtype=np.int32))
    simulation = Simulation(200, 100, seed=1)
    simulation.set_image(field)
    assert len(simulation.dots) == 2  # Empty classes need no attractors.
    simulation.dots[0].x = 150  # A wrong-class source must not steal this capture.
    simulation.add_ball(150, 50)
    assert simulation.target_groups.tolist() == [group]
    simulation.step(0.01)
    assert simulation.ball_count == 0
    assert simulation.captured_count == 1
    assert simulation.specks[0].color == field.colors[group]
    simulation.spawn_random(100)
    assert set(simulation.target_groups) == {0, group}


def test_nearest_groups_matches_exact_rgb_distances_and_first_ties_across_chunks():
    rng = np.random.default_rng(42)
    palette = rng.integers(4, 256, size=(2051, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    palette[1024] = (2, 0, 0)
    pixels = rng.integers(0, 256, size=(300, 3), dtype=np.uint8)
    pixels[0] = (1, 0, 0)
    delta = pixels.astype(np.float64)[:, None, :] - palette
    expected = np.argmin(np.sum(delta * delta, axis=2), axis=1)
    actual = ImageField._nearest_groups(pixels, palette)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == 0


def test_all_unique_colours_retained_bypass_quantization_and_nearest_search(monkeypatch):
    def unnecessary_search(*args):
        pytest.fail("Retaining every unique colour needs neither splitting nor nearest search.")

    monkeypatch.setattr(ImageField, "_median_cut", unnecessary_search)
    monkeypatch.setattr(ImageField, "_nearest_groups", unnecessary_search)
    indices = np.arange(32770, dtype=np.uint32)
    pixels = np.column_stack((indices >> 16, (indices >> 8) & 255, indices & 255)).astype(np.uint8)
    # Noncontiguous input, repeated colours and a request beyond the RGB domain.
    pixels = np.stack((pixels, pixels), axis=0)[:, ::-1]
    field = ImageField.from_pixels(pixels, 10**100)
    assert len(field.palette) == 32770
    assert field.labels.max() == 32769
    np.testing.assert_array_equal(field.palette[field.labels], pixels)


def test_reduced_large_palette_classifies_by_exact_nearest_rgb():
    pixels = np.random.default_rng(17).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    pixels[8:] = pixels[:8]  # Classification must also preserve repeated-colour locations.
    field = ImageField.from_pixels(pixels, 48)
    assert len(field.palette) == 48
    delta = pixels.astype(np.float64)[:, :, None, :] - field.palette
    expected = np.argmin(np.sum(delta * delta, axis=3), axis=2)
    np.testing.assert_array_equal(field.labels, expected)


def test_median_cut_preserves_weighted_split_order():
    pixels = np.random.default_rng(17).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    field = ImageField.from_pixels(pixels, 8)
    assert field.colors == [
        (49, 73, 189),
        (74, 46, 42),
        (80, 197, 53),
        (92, 193, 170),
        (146, 67, 207),
        (189, 186, 84),
        (190, 172, 183),
        (200, 70, 93),
    ]


def test_dot_seeds_are_centroids_of_present_groups_only():
    field = ImageField(
        np.zeros((5, 3), dtype=np.uint8),
        np.array([[2, 0, 2], [0, 0, 4]], dtype=np.int32),
    )
    np.testing.assert_allclose(
        field.dot_seeds(300, 200),
        [(350 / 3, 350 / 3, 0), (150, 50, 2), (250, 150, 4)],
    )


def test_palette_averages_nearby_colours_instead_of_dropping_them():
    pixels = np.array([[[0, 0, 255], [0, 0, 255], [220, 0, 0], [255, 30, 0]]], dtype=np.uint8)
    field = ImageField.from_pixels(pixels, 2)
    assert len(field.colors) == 2
    warm = max(field.colors, key=lambda color: color[0])
    assert warm[0] >= 230 and warm[1] >= 10
    assert field.labels[0, 2] == field.labels[0, 3]
    assert field.labels[0, 0] != field.labels[0, 2]


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((0, 1, 3), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=float),
    ],
)
def test_invalid_pixel_shapes_or_types_are_rejected(pixels):
    with pytest.raises(ValueError):
        ImageField.from_pixels(pixels)


@pytest.mark.parametrize("colors", [-1, 0, 1, True, False, 2.5, "8", None, np.int64(8)])
def test_palette_size_is_validated(colors):
    with pytest.raises(ValueError):
        ImageField.from_pixels(np.zeros((1, 1, 3), dtype=np.uint8), colors)


def test_image_larger_than_old_pixel_ceiling_is_not_resized_or_rejected():
    pixels = np.full((2049, 2048, 3), 127, dtype=np.uint8)
    field = ImageField.from_pixels(pixels, 2)
    assert field.labels.shape == pixels.shape[:2]
    assert field.colors == [(127, 127, 127)]
    assert np.count_nonzero(field.labels) == 0
    assert field.contrast is not None
    assert np.count_nonzero(field.contrast) == 0


def test_palette_extraction_retains_more_than_32_real_colours():
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    field = ImageField.from_pixels(pixels, 64)
    assert len(field.palette) == 64
    np.testing.assert_array_equal(field.palette[field.labels], pixels)


def test_monochrome_image_has_one_real_palette_class():
    field = ImageField.from_pixels(np.zeros((2, 2, 3), dtype=np.uint8), 8)
    assert field.colors == [(0, 0, 0)]
    assert len(field.dot_seeds(200, 200)) == 1


def test_two_color_image_retains_colors_and_maps_spawn_positions():
    pixels = np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8)
    field = ImageField.from_pixels(pixels, 8)
    assert set(field.colors) == {(255, 0, 0), (0, 0, 255)}
    groups = field.sample_groups(np.array([[25.0, 50.0], [175.0, 50.0], [25.0, 0.0]]), 200, 200)
    assert field.colors[groups[0]] == (255, 0, 0)
    assert field.colors[groups[1]] == (0, 0, 255)
    assert groups[2] == -1  # Letterboxing is not stretched into the image.
    seeds = field.dot_seeds(200, 200)
    assert len(seeds) == 2
    assert all(0 <= x <= 200 and 0 <= y <= 200 for x, y, group in seeds)


@pytest.mark.parametrize("shape", [(1, 1), (1, 8), (8, 1), (8, 8)])
def test_flat_images_have_no_artificial_border_contrast(shape):
    field = ImageField.from_pixels(np.full((*shape, 3), 127, dtype=np.uint8), 2)
    np.testing.assert_array_equal(field.contrast, np.zeros(shape))


@pytest.mark.parametrize("axis", [0, 1])
def test_contrast_detects_both_sides_of_colour_edges_before_palette_reduction(axis):
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    pixels[:] = (255, 0, 0)
    if axis == 0:
        pixels[4:] = (0, 130, 0)  # Nearly equal luminance, but a strong colour edge.
    else:
        pixels[:, 4:] = (0, 130, 0)
    field = ImageField.from_pixels(pixels, 2)
    expected = np.zeros((8, 8))
    if axis == 0:
        expected[3:5] = 1
    else:
        expected[:, 3:5] = 1
    np.testing.assert_array_equal(field.contrast, expected)


def test_contrast_matches_full_resolution_rgb_differences_across_chunks():
    pixels = np.random.default_rng(8).integers(0, 256, (513, 257, 3), dtype=np.uint8)
    rgb = pixels.astype(np.float64) / 255.0
    expected = np.zeros(pixels.shape[:2])
    horizontal = np.max(np.abs(rgb[:, 1:] - rgb[:, :-1]), axis=2)
    vertical = np.max(np.abs(rgb[1:] - rgb[:-1]), axis=2)
    expected[:, 1:] = np.maximum(expected[:, 1:], horizontal)
    expected[:, :-1] = np.maximum(expected[:, :-1], horizontal)
    expected[1:] = np.maximum(expected[1:], vertical)
    expected[:-1] = np.maximum(expected[:-1], vertical)
    field = ImageField.from_pixels(pixels, 2)
    np.testing.assert_array_equal(field.contrast, expected)


def test_subtle_pixel_noise_is_not_normalised_into_strong_contrast():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, 2:] = 1
    field = ImageField.from_pixels(pixels, 2)
    assert field.contrast is not None
    assert field.contrast.max() == pytest.approx(1 / 255)
