import numpy as np

from gravity_studio.image_field import ImageField
from gravity_studio.model import Simulation


def field():
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[:, :2] = (255, 0, 0)
    pixels[:, 2:] = (0, 0, 255)
    return ImageField.from_pixels(pixels, 8)


def test_image_balls_are_pulled_only_by_their_spawn_colour_class():
    sim = Simulation(200, 200)
    sim.set_image(field())
    red = next(dot for dot in sim.dots if dot.color == (255, 0, 0))
    blue = next(dot for dot in sim.dots if dot.color == (0, 0, 255))
    red.x, red.y = 180, 50
    blue.x, blue.y = 10, 50
    sim.add_ball(50, 50)
    sim.add_ball(150, 50)
    sim.step(1 / 120)
    assert sim.velocities[0, 0] > 0
    assert sim.velocities[1, 0] < 0
    assert sim.paint_spawn_points


def test_wrong_colour_dot_cannot_capture_ball_even_on_bounced_sweep():
    sim = Simulation(200, 200)
    sim.set_image(field())
    red = next(dot for dot in sim.dots if dot.color == (255, 0, 0))
    blue = next(dot for dot in sim.dots if dot.color == (0, 0, 255))
    red.x, red.y = 170, 50
    blue.x, blue.y = 60, 50
    sim.gravity_strength = sim.drag = 0
    sim.add_ball(20, 50, 4000, 0)
    sim.step(0.05)
    assert sim.ball_count == 0
    assert sim.specks[0].color == (255, 0, 0)
    assert (sim.specks[0].x, sim.specks[0].y) == (20, 50)


def test_recoloring_a_palette_dot_updates_its_class_without_retargeting_balls():
    sim = Simulation(200, 200)
    sim.set_image(field())
    dot = next(dot for dot in sim.dots if dot.color == (255, 0, 0))
    copy = sim.add_dot(30, 30, dot.color)
    sim.add_ball(20, 50)
    group = sim.target_groups[0]
    sim.recolor_dot(dot, (0, 255, 0))
    assert copy.color == dot.color == (0, 255, 0)
    assert sim.image_field.colors[group] == (0, 255, 0)
    assert sim.target_groups[0] == group
    sim.gravity_strength = 0
    sim.positions[0] = (dot.x, dot.y)
    sim.step(1 / 120)
    assert sim.specks[0].color == (0, 255, 0)


def test_real_physics_gradually_paints_image_without_teleporting_particles():
    sim = Simulation(200, 200, seed=4)
    sim.set_image(field())
    sim.spawn_random(300)
    for _ in range(2400):
        sim.step(1 / 120)
        if not sim.ball_count:
            break
    assert sim.captured_count == 300 and sim.ball_count == 0
    assert len(sim.specks) == 300
    for mark in sim.specks:
        assert mark.color == ((255, 0, 0) if mark.x < 100 else (0, 0, 255))


def test_target_identity_survives_swap_removal_compaction_and_resizing():
    sim = Simulation(200, 200)
    sim.set_image(field())
    sim.gravity_strength = 0
    sim.add_ball(20, 20)
    sim.add_ball(180, 20)
    sim.add_ball(30, 30)
    assert sim.erase_ball_near(20, 20)
    groups = sim.target_groups.copy()
    sim.resize(180, 180)
    np.testing.assert_array_equal(sim.target_groups, groups)
    red = next(dot for dot in sim.dots if dot.color == (255, 0, 0))
    sim.positions[0] = (red.x, red.y)
    sim.step(1 / 120)
    assert sim.ball_count == 1
    remaining = next(dot for dot in sim.dots if dot.palette_index == sim.target_groups[0])
    assert remaining.color == (0, 0, 255)
    sim.reset()
    assert sim.image_field is None and not sim.image_mode


def test_image_random_spawns_stay_in_the_fitted_image():
    sim = Simulation(200, 200, seed=5)
    sim.set_image(ImageField.from_pixels(np.array([[[255, 0, 0], [0, 0, 255]]], dtype=np.uint8), 2))
    sim.spawn_random(1000)
    assert (sim.positions[:, 1] >= 50).all() and (sim.positions[:, 1] <= 150).all()
    assert (sim.target_groups >= 0).all()


def test_contrast_spawns_preserve_fitted_bounds_groups_and_origins_after_resize():
    sim = Simulation(400, 200, seed=42)
    sim.set_image(field())
    sim.resize(250, 180)
    sim.spawn_random(0, favor_contrast=True)
    sim.spawn_random(2000, favor_contrast=True)
    assert sim.ball_count == 2000
    assert (sim.positions[:, 0] >= 100).all()
    assert (sim.positions[:, 0] <= 247.5).all()
    assert (sim.positions[:, 1] >= 2.5).all()
    assert (sim.positions[:, 1] <= 177.5).all()
    np.testing.assert_array_equal(sim.positions, sim.origins)
    np.testing.assert_array_equal(
        sim.target_groups, sim.image_field.sample_groups(sim.origins, 400, 200)
    )


def test_contrast_preference_is_ignored_outside_image_mode():
    biased = Simulation(200, 200, seed=42)
    normal = Simulation(200, 200, seed=42)
    for sim in (biased, normal):
        sim.set_image(field())
        sim.image_mode = False
    biased.spawn_random(2000, favor_contrast=True)
    normal.spawn_random(2000)
    np.testing.assert_array_equal(biased.origins, normal.origins)
    biased.reset()
    normal.reset()
    biased.spawn_random(100, favor_contrast=True)
    normal.spawn_random(100)
    np.testing.assert_array_equal(biased.origins, normal.origins)
