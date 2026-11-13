import math

import numpy as np
import pytest

from gravity_studio.model import Simulation


@pytest.mark.parametrize("paint", [False, True])
def test_capture_removes_ball_and_paints_its_original_spawn_when_enabled(paint):
    sim = Simulation()
    sim.gravity_strength = 0
    sim.drag = 0
    sim.paint_spawn_points = paint
    dot = sim.add_dot(200, 100, (240, 100, 90))
    sim.add_ball(100, 100, 1000, 0)
    sim.step(0.05)
    assert sim.ball_count == 1
    dot.color = (50, 200, 220)
    sim.step(0.05)
    assert sim.ball_count == 0
    assert sim.captured_count == 1
    assert len(sim.specks) == int(paint)
    if paint:
        assert (sim.specks[0].x, sim.specks[0].y) == (100, 100)
        assert sim.specks[0].color == dot.color


def test_fast_ball_hits_first_dot_on_its_path_not_first_in_list():
    sim = Simulation()
    sim.gravity_strength = 0
    sim.drag = 0
    sim.paint_spawn_points = True
    sim.add_dot(400, 100, (0, 0, 255))
    sim.add_dot(200, 100, (255, 0, 0))
    sim.add_ball(100, 100, 10000, 0)
    sim.step(0.05)
    assert sim.ball_count == 0
    assert sim.specks[0].color == (255, 0, 0)


def test_ball_spawned_inside_dot_is_captured_without_nans():
    sim = Simulation()
    sim.add_dot(200, 200, (255, 0, 0))
    sim.add_ball(200, 200)
    sim.step(1 / 120)
    assert sim.ball_count == 0
    assert np.isfinite(sim.positions).all()


def test_opposing_dots_balance_and_many_dots_are_not_capped():
    sim = Simulation()
    sim.add_dot(200, 350, (255, 0, 0))
    sim.add_dot(800, 350, (0, 0, 255))
    sim.add_ball(500, 350)
    sim.step(1 / 120)
    np.testing.assert_allclose(sim.positions, [[500, 350]])
    for _ in range(1500):
        sim.add_dot(100, 100, (100, 100, 100))
    assert len(sim.dots) == 1502


def test_paint_toggle_only_affects_future_captures_and_old_colors_are_frozen():
    sim = Simulation()
    dot = sim.add_dot(200, 200, (255, 0, 0))
    sim.add_ball(200, 200)
    sim.step(1 / 120)
    sim.paint_spawn_points = True
    assert not sim.specks
    sim.add_ball(200, 200)
    sim.step(1 / 120)
    dot.color = (0, 0, 255)
    sim.paint_spawn_points = False
    sim.add_ball(200, 200)
    sim.step(1 / 120)
    assert len(sim.specks) == 1
    assert sim.specks[0].color == (255, 0, 0)
    assert sim.captured_count == 3


@pytest.mark.parametrize("radius", [-1, 0, float("nan")])
def test_invalid_dot_radius_does_not_modify_the_scene(radius):
    sim = Simulation()
    with pytest.raises(ValueError):
        sim.add_dot(200, 200, (255, 0, 0), radius)
    assert not sim.dots


def test_zero_time_and_invalid_velocity_are_safe():
    sim = Simulation()
    with pytest.raises(ValueError):
        sim.add_ball(200, 200, float("inf"), 0)
    sim.add_dot(200, 200, (255, 0, 0))
    sim.add_ball(200, 200)
    sim.step(0)
    assert sim.ball_count == 1


def test_capture_follows_the_reflected_wall_path():
    sim = Simulation(100, 100)
    sim.gravity_strength = 0
    sim.drag = 0
    sim.paint_spawn_points = True
    sim.add_dot(70, 50, (255, 0, 0), radius=1)
    sim.add_ball(90, 50, 1000, 0)
    sim.step(0.05)
    assert sim.ball_count == 0
    assert (sim.specks[0].x, sim.specks[0].y) == (90, 50)


def test_diagonal_corner_bounce_can_capture_in_the_same_step():
    sim = Simulation(100, 100)
    sim.gravity_strength = 0
    sim.drag = 0
    sim.add_dot(70, 70, (0, 255, 0), radius=1)
    sim.add_ball(90, 90, 1000, 1000)
    sim.step(0.05)
    assert sim.ball_count == 0


def test_tiny_positive_dot_radius_does_not_underflow_softening():
    sim = Simulation()
    sim.add_dot(100, 100, (255, 0, 0), radius=1e-300)
    sim.add_ball(100, 100)
    with np.errstate(all="raise"):
        sim.step(1 / 120)
    assert sim.ball_count == 0


def test_edge_spawn_is_inset_once_not_reflected_away_from_wall():
    sim = Simulation(100, 100)
    sim.drag = 0
    sim.add_ball(0, 50)
    assert sim.positions[0, 0] == 2.5
    sim.step(1 / 120)
    assert sim.positions[0, 0] == 2.5
    assert sim.origins[0, 0] == 0


def test_extreme_gravity_is_rejected_before_overflow():
    sim = Simulation(100, 100)
    sim.add_dot(50, 50, (255, 0, 0), radius=1)
    sim.add_ball(54, 50)
    sim.gravity_strength = 1e308
    with pytest.raises(ValueError):
        sim.step(0.05)
    np.testing.assert_array_equal(sim.positions, [[54, 50]])


def test_random_batches_have_no_total_count_cap_and_preserve_origins():
    sim = Simulation(width=900, height=600, seed=42)
    sim.spawn_random(10000)
    sim.spawn_random(250)
    assert sim.ball_count == 10250
    np.testing.assert_array_equal(sim.origins, sim.positions)
    assert ((sim.positions[:, 0] >= 0) & (sim.positions[:, 0] <= 900)).all()
    assert ((sim.positions[:, 1] >= 0) & (sim.positions[:, 1] <= 600)).all()
    same_seed = Simulation(width=900, height=600, seed=42)
    same_seed.spawn_random(10000)
    np.testing.assert_array_equal(sim.positions[:10000], same_seed.positions)


def test_balls_reflect_at_walls_and_spawn_coordinates_do_not_move():
    sim = Simulation(width=100, height=100)
    sim.drag = 0
    sim.add_ball(95, 50, 1000, 0)
    sim.step(0.05)
    assert 0 <= sim.positions[0, 0] <= 100
    assert sim.velocities[0, 0] < 0
    np.testing.assert_array_equal(sim.origins, [[95, 50]])


def test_clear_reset_resize_and_removal_have_separate_scopes():
    sim = Simulation()
    dot = sim.add_dot(200, 200, (255, 0, 0))
    sim.paint_spawn_points = True
    sim.add_ball(200, 200)
    sim.step(1 / 120)
    sim.add_ball(600, 600)
    assert not sim.erase_ball_near(0, 0)
    assert sim.erase_ball_near(600, 600)
    sim.spawn_random(20)
    sim.clear_balls()
    assert sim.ball_count == 0
    assert len(sim.specks) == 1
    sim.resize(100, 100)
    assert 0 <= dot.x <= 100 and 0 <= dot.y <= 100
    generation = sim.paint_generation
    sim.clear_paint()
    assert not sim.specks and sim.paint_generation > generation
    assert sim.captured_count == 1
    sim.remove_dot(dot)
    assert not sim.dots
    sim.add_dot(50, 50, (0, 255, 0))
    sim.spawn_random(10)
    sim.reset()
    assert sim.ball_count == 0 and not sim.dots and not sim.specks
    assert sim.captured_count == 0


@pytest.mark.parametrize("width,height", [(0, 100), (100, 5), (float("nan"), 100)])
def test_invalid_world_sizes_are_rejected(width, height):
    with pytest.raises(ValueError):
        Simulation(width, height)


@pytest.mark.parametrize("x,y", [(-1, 2), (1001, 3), (float("nan"), 4), (3, float("inf"))])
def test_invalid_placements_are_rejected_without_changing_scene(x, y):
    sim = Simulation()
    with pytest.raises(ValueError):
        sim.add_ball(x, y)
    with pytest.raises(ValueError):
        sim.add_dot(x, y, (255, 0, 0))
    assert sim.ball_count == 0 and not sim.dots


@pytest.mark.parametrize("color", [(256, 0, 0), (2, 3), (0.5, 3, 4)])
def test_invalid_colors_are_rejected(color):
    sim = Simulation()
    with pytest.raises(ValueError):
        sim.add_dot(50, 50, color)


@pytest.mark.parametrize(
    "setting,value", [("drag", -1), ("drag", float("nan")), ("gravity_strength", -1)]
)
def test_invalid_physics_settings_do_not_poison_positions(setting, value):
    sim = Simulation()
    sim.add_ball(100, 100)
    setattr(sim, setting, value)
    with pytest.raises(ValueError):
        sim.step(1 / 120)
    np.testing.assert_array_equal(sim.positions, [[100, 100]])


@pytest.mark.parametrize("dt", [-0.01, 1, float("nan"), float("inf")])
def test_invalid_timestep_is_rejected(dt):
    with pytest.raises(ValueError):
        Simulation().step(dt)


def test_invalid_random_count_and_resize_are_atomic():
    sim = Simulation()
    sim.add_ball(100, 100)
    for count in (-1, 2.5, True):
        with pytest.raises(ValueError):
            sim.spawn_random(count)
    with pytest.raises(ValueError):
        sim.resize(0, 0)
    assert (sim.width, sim.height) == (1000, 700)
    assert sim.ball_count == 1


def test_drag_is_exponential_and_step_rate_independent_without_forces():
    sim = Simulation()
    sim.drag = 2
    sim.add_ball(100, 100, 10, 0)
    for _ in range(120):
        sim.step(1 / 120)
    assert sim.velocities[0, 0] == pytest.approx(10 * math.exp(-2))
    no_drag = Simulation()
    no_drag.drag = 0
    no_drag.add_ball(100, 100, 10, 0)
    no_drag.step(1 / 120)
    assert no_drag.velocities[0, 0] == 10


def test_balls_accelerate_toward_gravity_dots():
    sim = Simulation()
    sim.add_dot(500, 350, (240, 100, 90))
    sim.add_ball(100, 350)
    sim.step(1 / 120)
    assert sim.positions[0, 0] > 100
    assert sim.velocities[0, 0] > 0
    assert sim.positions[0, 1] == 350
    assert np.isfinite(sim.positions).all()
