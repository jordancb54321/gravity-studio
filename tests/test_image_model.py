import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from gravity_studio.image_field import ImageField
from gravity_studio.model import BALL_RADIUS, Simulation

RED = (255, 0, 0)
BLUE = (0, 0, 255)


def field():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, :2] = RED
    pixels[:, 2:] = BLUE
    return ImageField.from_pixels(pixels, 8)


@given(y=st.integers(25, 175), distance=st.integers(20, 35))
def test_image_attraction_excludes_nearer_wrong_colour_sources(y, distance):
    sim = Simulation(200, 200)
    sim.set_image(field())
    red = next(dot for dot in sim.dots if dot.color == RED)
    blue = next(dot for dot in sim.dots if dot.color == BLUE)
    red.x, red.y = 150 + distance, y
    blue.x, blue.y = 50 - distance, y
    sim.add_ball(50, y)
    sim.add_ball(150, y)

    sim.step(1 / 120)

    assert sim.velocities[0, 0] > 0
    assert sim.velocities[1, 0] < 0


def test_reflected_image_trajectory_skips_wrong_colour_on_the_way_to_its_source():
    sim = Simulation(200, 200)
    sim.set_image(field())
    red = next(dot for dot in sim.dots if dot.color == RED)
    blue = next(dot for dot in sim.dots if dot.color == BLUE)
    # The ball passes through the blue dot outbound and the red dot after bouncing.
    blue.x, blue.y = 65, 38
    red.x, red.y = 145, 112
    sim.gravity_strength = sim.drag = 0
    sim.add_ball(20, 20, 5000, 2000)

    sim.step(0.05)

    assert sim.ball_count == 0
    assert [(mark.x, mark.y, mark.color) for mark in sim.specks] == [(20, 20, RED)]


def test_recolouring_image_class_changes_future_captures_by_all_its_sources():
    sim = Simulation(200, 200)
    sim.set_image(field())
    red = next(dot for dot in sim.dots if dot.color == RED)
    sim.add_dot(25, 30, RED)
    sim.add_ball(26, 30)
    sim.add_ball(red.x, red.y)

    green = (0, 255, 0)
    sim.recolor_dot(red, green)
    sim.step(1 / 120)

    assert sim.ball_count == 0
    assert [mark.color for mark in sim.specks] == [green, green]
    # A new ball sampling red in the original image still joins the recoloured class.
    sim.add_ball(25, 30)
    sim.step(1 / 120)
    assert [mark.color for mark in sim.specks] == [green, green, green]


def test_target_identity_survives_swap_removal_compaction_and_resizing():
    sim = Simulation(200, 200)
    sim.set_image(field())
    sim.gravity_strength = 0
    sim.add_ball(20, 20)
    sim.add_ball(180, 20)
    red = next(dot for dot in sim.dots if dot.color == RED)
    sim.add_ball(red.x, red.y)
    assert sim.erase_ball_near(20, 20)

    sim.resize(100, 180)
    sim.step(1 / 120)

    assert sim.ball_count == 1
    assert sim.origins.tolist() == [[180, 20]]
    assert (sim.specks[0].x, sim.specks[0].y, sim.specks[0].color) == (red.x, red.y, RED)
    blue = next(dot for dot in sim.dots if dot.color == BLUE)
    sim.remove_dot(blue)
    sim.add_dot(*sim.positions[0], RED)
    sim.step(1 / 120)
    assert sim.ball_count == 1

    sim.add_dot(*sim.positions[0], BLUE)
    sim.step(1 / 120)
    assert sim.ball_count == 0
    assert (sim.specks[-1].x, sim.specks[-1].y, sim.specks[-1].color) == (180, 20, BLUE)


@settings(max_examples=40)
@given(
    width=st.integers(80, 500),
    height=st.integers(80, 500),
    favor_contrast=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
@example(width=400, height=200, favor_contrast=True, seed=42)
def test_random_origins_and_colour_classes_use_original_image_fit_after_resize(
    width, height, favor_contrast, seed
):
    sim = Simulation(width, height, seed=seed)
    sim.set_image(field())
    sim.resize(0.7 * width, 0.9 * height)
    sim.spawn_random(0, favor_contrast=favor_contrast)
    sim.spawn_random(200, favor_contrast=favor_contrast)

    fitted_side = min(width, height)
    for axis, (before, after) in enumerate(((width, sim.width), (height, sim.height))):
        assert (sim.positions[:, axis] >= max(BALL_RADIUS, (before - fitted_side) / 2)).all()
        assert (
            sim.positions[:, axis] <= min(after - BALL_RADIUS, (before + fitted_side) / 2)
        ).all()
    np.testing.assert_array_equal(sim.positions, sim.origins)
    assert len(sim.origins) == 200
    groups = sim.target_groups
    assert (groups >= 0).all()
    expected = np.where((sim.origins[:, 0] < width / 2)[:, None], RED, BLUE)
    np.testing.assert_array_equal(sim.image_field.palette[groups], expected)


@pytest.mark.parametrize("has_image", [False, True])
def test_contrast_preference_is_ignored_in_sandbox_mode(has_image):
    biased = Simulation(200, 200, seed=42)
    normal = Simulation(200, 200, seed=42)
    if has_image:
        for sim in (biased, normal):
            sim.set_image(field())
            sim.image_mode = False
    biased.spawn_random(256, favor_contrast=True)
    normal.spawn_random(256)
    np.testing.assert_array_equal(biased.origins, normal.origins)


def test_manual_ball_in_letterbox_uses_sandbox_gravity():
    sim = Simulation(400, 200)
    sim.set_image(field())
    yellow = (255, 255, 0)
    sim.add_dot(40, 100, yellow)
    sim.add_ball(40, 100)

    sim.step(1 / 120)

    assert sim.ball_count == 0
    assert sim.specks[0].color == yellow
