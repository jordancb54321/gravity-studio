import math

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from gravity_studio.model import BALL_RADIUS, Simulation, Speck

RED = (255, 0, 0)
BLUE = (0, 0, 255)
RGB = st.tuples(*(st.integers(0, 255) for _ in range(3)))
NONFINITE = st.sampled_from([math.nan, math.inf, -math.inf])


@given(
    start_x=st.integers(10, 100),
    y=st.integers(60, 140),
    first_distance=st.integers(20, 100),
    gap=st.integers(6, 50),
    reverse_order=st.booleans(),
)
def test_swept_capture_selects_first_intersection_and_paints_spawn(
    start_x, y, first_distance, gap, reverse_order
):
    sim = Simulation(500, 200)
    sim.gravity_strength = 0
    sim.drag = 0
    sim.paint_spawn_points = True
    near_x = start_x + first_distance
    # The front circle overlaps the rear one in some scenes. An off-path dot
    # closer to the launch point must never be selected.
    sim.add_dot(start_x + 10, y + 20, (0, 255, 0), radius=3)
    dots = [(near_x, RED), (near_x + gap, BLUE)]
    for x, color in reversed(dots) if reverse_order else dots:
        sim.add_dot(x, y, color, radius=5)

    sim.add_ball(start_x, y, 4000, 0)
    sim.step(0.05)

    assert sim.ball_count == 0
    assert sim.captured_count == 1
    assert sim.specks == [Speck(start_x, y, RED)]


@given(
    start=st.integers(75, 90),
    return_distance=st.integers(15, 55),
    corner=st.booleans(),
)
def test_capture_follows_wall_and_corner_reflections(start, return_distance, corner):
    sim = Simulation(100, 100)
    sim.gravity_strength = 0
    sim.drag = 0
    sim.paint_spawn_points = True
    target = start - return_distance
    dot_y = target if corner else 50
    start_y = start if corner else 50
    sim.add_dot(target, dot_y, RED, radius=1)
    sim.add_ball(start, start_y, 2000, 2000 if corner else 0)

    sim.step(0.05)

    assert sim.ball_count == 0
    assert sim.captured_count == 1
    assert sim.specks == [Speck(start, start_y, RED)]


@given(radius=st.floats(1e-300, 1e-150, allow_subnormal=False))
def test_tiny_dot_captures_ball_at_rest_without_poisoning_survivor(radius):
    sim = Simulation()
    sim.add_dot(100, 100, RED, radius=radius)
    sim.add_ball(100, 100)
    sim.add_ball(120, 100)

    sim.step(0)
    assert sim.ball_count == 2
    with np.errstate(all="raise"):
        sim.step(1 / 120)

    assert sim.captured_count == 1
    np.testing.assert_array_equal(sim.origins, [[120, 100]])
    assert 100 < sim.positions[0, 0] < 120
    assert sim.velocities[0, 0] < 0


@given(captures=st.lists(st.tuples(st.booleans(), RGB), min_size=1, max_size=12))
@example(captures=[(False, BLUE), (True, RED), (False, BLUE), (True, BLUE)])
def test_paint_decisions_and_capture_colors_are_preserved_across_later_captures(captures):
    sim = Simulation(200, 200)
    source = sim.add_dot(100, 100, RED)
    expected_marks = []
    for index, (paint, color) in enumerate(captures):
        x = 94 + index
        sim.add_ball(x, 100)
        sim.paint_spawn_points = paint
        sim.recolor_dot(source, color)
        sim.step(1 / 120)

        if paint:
            expected_marks.append(Speck(x, 100, color))
        assert sim.specks == expected_marks
        assert sim.ball_count == 0
        assert sim.captured_count == index + 1


@given(
    width=st.integers(6, 200),
    height=st.integers(6, 200),
    x_fraction=st.floats(0, 1),
    y_fraction=st.floats(0, 1),
    vx=st.integers(-100_000, 100_000),
    vy=st.integers(-100_000, 100_000),
)
@example(width=6, height=10, x_fraction=0, y_fraction=1, vx=0, vy=0)
def test_ball_motion_without_forces_is_reversible_and_preserves_speed_and_origins(
    width, height, x_fraction, y_fraction, vx, vy
):
    sim = Simulation(width, height)
    sim.drag = 0
    origin = (width * x_fraction, height * y_fraction)
    sim.add_ball(*origin, vx, vy)
    limits = np.array([width, height]) - BALL_RADIUS
    initial_position = np.clip(origin, BALL_RADIUS, limits)
    np.testing.assert_array_equal(sim.positions, [initial_position])
    sim.step(1 / 32)

    assert (sim.positions >= BALL_RADIUS).all()
    assert (sim.positions <= limits).all()
    np.testing.assert_array_equal(sim.origins, [origin])
    np.testing.assert_array_equal(np.abs(sim.velocities), [[abs(vx), abs(vy)]])

    reverse = Simulation(width, height)
    reverse.drag = 0
    reverse.add_ball(*sim.positions[0], *-sim.velocities[0])
    reverse.step(1 / 32)
    np.testing.assert_allclose(reverse.positions, [initial_position], atol=1e-10, rtol=0)


@given(
    canvas_size=st.tuples(st.integers(6, 160), st.integers(6, 160)),
    ball=st.tuples(st.integers(0, 160), st.integers(0, 160)),
)
def test_resizing_twice_clamps_live_positions_but_preserves_spawn_and_paint(canvas_size, ball):
    sim = Simulation(160, 160)
    sim.paint_spawn_points = True
    source = sim.add_dot(130, 140, RED)
    sim.add_ball(130, 140)
    sim.step(1 / 120)
    sim.add_ball(*ball)

    sim.resize(*canvas_size)
    positions = sim.positions.copy()
    clamped_ball = [
        min(max(coordinate, BALL_RADIUS), size - BALL_RADIUS)
        for coordinate, size in zip(ball, canvas_size, strict=True)
    ]
    np.testing.assert_array_equal(positions, [clamped_ball])
    dot_position = (min(130, canvas_size[0]), min(140, canvas_size[1]))
    assert (source.x, source.y) == dot_position

    sim.resize(*canvas_size)
    np.testing.assert_array_equal(sim.positions, positions)
    np.testing.assert_array_equal(sim.origins, [ball])
    assert (source.x, source.y) == dot_position
    assert sim.specks == [Speck(130, 140, RED)]
    assert sim.captured_count == 1


@given(
    drag=st.integers(0, 20),
    vx=st.integers(-80, 80),
    vy=st.integers(-80, 80),
    subdivisions=st.integers(1, 10),
)
@example(drag=0, vx=10, vy=-20, subdivisions=10)
def test_exponential_drag_is_independent_of_step_partition(drag, vx, vy, subdivisions):
    single = Simulation()
    divided = Simulation()
    for sim in (single, divided):
        sim.drag = drag
        sim.add_ball(500, 350, vx, vy)

    duration = 0.05
    single.step(duration)
    for _ in range(subdivisions):
        divided.step(duration / subdivisions)

    expected = np.array([[vx, vy]]) * math.exp(-drag * duration)
    np.testing.assert_allclose(single.velocities, expected, atol=1e-12)
    np.testing.assert_allclose(divided.velocities, expected, atol=1e-12)


@given(
    distance=st.integers(20, 100),
    axis=st.integers(0, 1),
    direction=st.sampled_from([-1, 1]),
)
def test_attraction_points_toward_sources_weakens_with_distance_and_balances(
    distance, axis, direction
):
    origin = np.array([400, 400])
    offset = np.array([0, 0])
    offset[axis] = direction * distance
    near, far, combined, balanced = (Simulation(800, 800) for _ in range(4))
    for sim in (near, far, combined, balanced):
        sim.add_ball(*origin)
    near.add_dot(*(origin + offset), RED)
    far.add_dot(*(origin + 2 * offset), RED)
    combined.add_dot(*(origin + offset), RED)
    combined.add_dot(*(origin + 2 * offset), RED)
    balanced.add_dot(*(origin + offset), RED)
    balanced.add_dot(*(origin - offset), BLUE)

    for sim in (near, far, combined, balanced):
        sim.step(1 / 120)

    assert direction * near.velocities[0, axis] > direction * far.velocities[0, axis] > 0
    assert near.velocities[0, 1 - axis] == 0
    np.testing.assert_allclose(combined.velocities, near.velocities + far.velocities)
    np.testing.assert_array_equal(balanced.velocities, [[0, 0]])


@settings(max_examples=20)
@given(
    seed=st.integers(0, 2**32 - 1),
    first=st.integers(9000, 10_000),
    second=st.integers(1001, 1250),
    width=st.integers(6, 1000),
    height=st.integers(6, 1000),
)
def test_seeded_random_batches_compose_above_ten_thousand_balls(seed, first, second, width, height):
    chunked = Simulation(width, height, seed=seed)
    whole = Simulation(width, height, seed=seed)
    chunked.spawn_random(first)
    chunked.spawn_random(second)
    whole.spawn_random(first + second)

    assert chunked.ball_count == first + second
    np.testing.assert_array_equal(chunked.positions, whole.positions)
    np.testing.assert_array_equal(chunked.origins, chunked.positions)
    assert (chunked.positions >= BALL_RADIUS).all()
    assert (chunked.positions <= np.array([width, height]) - BALL_RADIUS).all()


def test_dot_beyond_one_thousand_can_capture_and_paint():
    sim = Simulation()
    sim.paint_spawn_points = True
    sim.gravity_strength = 0
    for _ in range(1024):
        sim.add_dot(100, 100, RED)
    sim.add_dot(150, 150, BLUE)
    sim.add_ball(150, 150)

    sim.step(1 / 120)

    assert sim.specks == [Speck(150, 150, BLUE)]


@given(
    bad_coordinate=st.one_of(st.integers(-1000, -1), st.integers(101, 1100), NONFINITE),
    bad_velocity=st.one_of(
        st.integers(-2_000_000, -1_000_001), st.integers(1_000_001, 2_000_000), NONFINITE
    ),
    bad_radius=st.one_of(st.integers(-1000, 0), st.integers(1_000_001, 2_000_000), NONFINITE),
    bad_color=st.one_of(
        st.tuples(st.integers(256, 1000), st.just(0), st.just(0)),
        st.sampled_from([(2, 3), (0.5, 3, 4), (True, 0, 0)]),
    ),
    reverse=st.booleans(),
)
def test_rejected_placements_preserve_existing_entities(
    bad_coordinate, bad_velocity, bad_radius, bad_color, reverse
):
    sim = Simulation(100, 100)
    ball = (50, 50)
    dot = sim.add_dot(20, 20, RED)
    sim.add_ball(*ball)
    position = (20, bad_coordinate) if reverse else (bad_coordinate, 20)
    velocity = (0, bad_velocity) if reverse else (bad_velocity, 0)

    with pytest.raises(ValueError):
        sim.add_ball(*position)
    with pytest.raises(ValueError):
        sim.add_dot(*position, RED)
    with pytest.raises(ValueError):
        sim.add_ball(*ball, *velocity)
    with pytest.raises(ValueError):
        sim.add_dot(20, 20, RED, radius=bad_radius)
    with pytest.raises(ValueError):
        sim.add_dot(20, 20, bad_color)

    assert sim.dots == [dot]
    np.testing.assert_array_equal(sim.positions, [ball])


@given(
    bad_size=st.one_of(st.integers(-1000, 5), st.integers(1_000_001, 2_000_000), NONFINITE),
    bad_count=st.one_of(st.integers(-1000, -1), st.sampled_from([True, 2.5])),
    reverse=st.booleans(),
)
def test_rejected_world_sizes_and_batch_counts_leave_current_scene_intact(
    bad_size, bad_count, reverse
):
    dimensions = (100, bad_size) if reverse else (bad_size, 100)
    with pytest.raises(ValueError):
        Simulation(*dimensions)

    sim = Simulation(100, 100)
    sim.add_ball(50, 50)
    with pytest.raises(ValueError):
        sim.resize(*dimensions)
    with pytest.raises(ValueError):
        sim.spawn_random(bad_count)

    assert (sim.width, sim.height) == (100, 100)
    np.testing.assert_array_equal(sim.positions, [[50, 50]])


@given(
    bad_dt=st.one_of(st.floats(-1, -0.0001), st.floats(0.051, 1), NONFINITE),
    bad_setting=st.one_of(st.integers(-1000, -1), st.floats(1e10, 1e308), NONFINITE),
    setting=st.sampled_from(["drag", "gravity_strength"]),
)
@example(bad_dt=-0.01, bad_setting=1e308, setting="gravity_strength")
def test_rejected_steps_do_not_change_position_or_velocity(bad_dt, bad_setting, setting):
    sim = Simulation(100, 100)
    sim.add_dot(50, 50, RED, radius=1)
    sim.add_ball(54, 50, vx=1, vy=2)

    with pytest.raises(ValueError):
        sim.step(bad_dt)
    setattr(sim, setting, bad_setting)
    with pytest.raises(ValueError):
        sim.step(0.05)

    np.testing.assert_array_equal(sim.positions, [[54, 50]])
    np.testing.assert_array_equal(sim.velocities, [[1, 2]])
    assert sim.captured_count == 0
