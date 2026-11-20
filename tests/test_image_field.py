import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from gravity_studio.image_field import ImageField
from gravity_studio.model import Simulation

RGB = st.tuples(*(st.integers(0, 255) for _ in range(3)))


@st.composite
def images(draw):
    rows, cols = draw(st.tuples(st.integers(1, 12), st.integers(1, 12)))
    palette = np.array(draw(st.lists(RGB, min_size=1, max_size=15, unique=True)), dtype=np.uint8)
    labels = draw(hnp.arrays(np.intp, (rows, cols), elements=st.integers(0, len(palette) - 1)))
    return palette[labels]


@settings(max_examples=50)
@given(pixels=images(), colors=st.integers(2, 16))
def test_palette_reconstruction_is_idempotent(pixels, colors):
    field = ImageField.from_pixels(pixels, colors)
    reconstructed = field.palette[field.labels]
    reduced_again = ImageField.from_pixels(reconstructed, colors)
    np.testing.assert_array_equal(reduced_again.palette[reduced_again.labels], reconstructed)
    assert 1 <= len(field.palette) <= colors
    if len({tuple(pixel) for pixel in pixels.reshape(-1, 3)}) <= colors:
        np.testing.assert_array_equal(reconstructed, pixels)


@settings(max_examples=40)
@given(pixels=images(), colors=st.integers(2, 8))
def test_quantized_pixels_choose_nearest_palette_colour(pixels, colors):
    field = ImageField.from_pixels(pixels, colors)
    # Signed arithmetic preserves the distance across the full RGB range.
    delta = pixels.astype(np.int32)[:, :, None, :] - field.palette
    nearest = np.argmin(np.sum(delta**2, axis=3), axis=2)
    np.testing.assert_array_equal(field.labels, nearest)


def test_nearest_palette_matches_reference_across_search_tiles():
    rng = np.random.default_rng(42)
    palette = rng.integers(4, 256, size=(2051, 3), dtype=np.uint8)
    palette[0] = (0, 0, 0)
    palette[1024] = (2, 0, 0)
    pixels = rng.integers(0, 256, size=(300, 3), dtype=np.uint8)
    pixels[0] = (1, 0, 0)
    delta = pixels.astype(np.int32)[:, None, :] - palette
    expected = np.argmin(np.sum(delta**2, axis=2), axis=1)
    np.testing.assert_array_equal(ImageField._nearest_groups(pixels, palette), expected)
    assert expected[0] == 0


def test_large_palette_round_trips_and_routes_high_group_ids():
    group = 32769
    indices = np.arange(group + 1, dtype=np.uint32)
    palette = np.column_stack((indices >> 16, (indices >> 8) & 255, indices & 255)).astype(np.uint8)
    # Keep the order reversed and repeat rows so this also exercises noncontiguous input.
    pixels = np.stack((palette, palette), axis=0)[:, ::-1]
    field = ImageField.from_pixels(pixels, 10**100)
    assert len(field.palette) == len(palette)
    np.testing.assert_array_equal(field.palette[field.labels], pixels)

    # Use only two widely separated labels from the extracted palette so that the
    # real simulation exercises the high label without processing 32,770 sources.
    sparse_field = ImageField(field.palette, np.array([[0, group]], dtype=np.int32))
    simulation = Simulation(200, 100, seed=1)
    simulation.set_image(sparse_field)
    assert len(simulation.dots) == 2
    simulation.dots[0].x = 150  # A wrong-class source must leave this capture to group 32769.
    simulation.add_ball(150, 50)
    simulation.step(0.01)
    assert simulation.ball_count == 0
    assert simulation.specks[0].color == sparse_field.colors[group]
    simulation.spawn_random(100)
    assert set(simulation.target_groups) == {0, group}


@settings(max_examples=40)
@given(pixels=images(), width=st.integers(40, 600), height=st.integers(40, 600))
def test_fitted_pixel_centres_round_trip_and_define_source_centroids(pixels, width, height):
    field = ImageField.from_pixels(pixels, 16)
    left, top, fitted_width, fitted_height = field.fit_rect(width, height)
    rows, cols = pixels.shape[:2]
    assert fitted_width / cols == pytest.approx(fitted_height / rows)
    assert max(fitted_width / width, fitted_height / height) == pytest.approx(1)
    assert 2 * left + fitted_width == pytest.approx(width)
    assert 2 * top + fitted_height == pytest.approx(height)

    row, col = np.indices((rows, cols))
    centers = np.column_stack(
        (
            left + (col.ravel() + 0.5) * fitted_width / cols,
            top + (row.ravel() + 0.5) * fitted_height / rows,
        )
    )
    groups = field.sample_groups(centers, width, height)
    np.testing.assert_array_equal(field.palette[groups], pixels.reshape(-1, 3))
    seeds = {group: (x, y) for x, y, group in field.dot_seeds(width, height)}
    assert seeds.keys() == set(groups)
    for group, center in seeds.items():
        np.testing.assert_allclose(center, centers[groups == group].mean(axis=0))

    if left > 1:
        assert field.sample_groups(np.array([[0, height / 2]]), width, height)[0] == -1
    if top > 1:
        assert field.sample_groups(np.array([[width / 2, 0]]), width, height)[0] == -1


@given(
    cool_repetitions=st.integers(3, 15),
    warm=st.integers(180, 204),
    spread=st.integers(20, 50),
)
def test_palette_reduction_blends_nearby_colours(cool_repetitions, warm, spread):
    cool = (0, 0, 255)
    first_warm = (warm, 0, 0)
    second_warm = (warm + spread, spread, 0)
    pixels = np.array([[cool] * cool_repetitions + [first_warm, second_warm]], dtype=np.uint8)

    field = ImageField.from_pixels(pixels, 2)

    blend = (round(warm + spread / 2), round(spread / 2), 0)
    assert set(field.colors) == {cool, blend}
    assert all(field.colors[group] == blend for group in field.labels[0, -2:])


@given(
    rows=st.integers(1, 10),
    cols=st.integers(1, 10),
    channels=st.sampled_from([0, 1, 2, 4, 5]),
    dtype=st.sampled_from([np.float32, np.int16, np.bool_]),
)
def test_invalid_pixel_shapes_or_types_are_rejected(rows, cols, channels, dtype):
    invalid_arrays = [
        np.zeros((0, cols, 3), dtype=np.uint8),
        np.zeros((rows, 0, 3), dtype=np.uint8),
        np.zeros((rows, cols), dtype=np.uint8),
        np.zeros((rows, cols, channels), dtype=np.uint8),
        np.zeros((rows, cols, 3), dtype=dtype),
    ]
    for pixels in invalid_arrays:
        with pytest.raises(ValueError):
            ImageField.from_pixels(pixels)


@given(colors=st.one_of(st.integers(max_value=1), st.booleans(), st.none(), st.floats(), st.text()))
@example(colors=np.int64(8))
def test_invalid_palette_sizes_are_rejected(colors):
    with pytest.raises(ValueError):
        ImageField.from_pixels(np.zeros((1, 1, 3), dtype=np.uint8), colors)


@settings(max_examples=40)
@given(
    edge=st.integers(1, 12),
    trailing=st.integers(1, 12),
    columns=st.integers(1, 8),
    strength=st.integers(0, 255),
    transpose=st.booleans(),
)
@example(edge=256, trailing=2, columns=256, strength=255, transpose=False)
@example(edge=4, trailing=4, columns=4, strength=1, transpose=True)
def test_rgb_contrast_marks_both_sides_without_amplifying_small_changes(
    edge, trailing, columns, strength, transpose
):
    pixels = np.zeros((edge + trailing, columns, 3), dtype=np.uint8)
    pixels[:edge] = (strength, 0, 0)
    pixels[edge:] = (0, strength // 2, 0)
    expected = np.zeros(pixels.shape[:2])
    expected[edge - 1 : edge + 1] = strength / 255
    if transpose:
        pixels = pixels.transpose(1, 0, 2)
        expected = expected.T
    np.testing.assert_array_equal(ImageField.from_pixels(pixels, 2).contrast, expected)
