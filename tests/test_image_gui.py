import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QCheckBox, QFileDialog, QMessageBox, QPushButton

from gravity_studio.image_field import ImageField
from gravity_studio.model import Simulation
from gravity_studio.ui import MainWindow


def test_origin_mark_size_redraws_existing_image_paint_and_new_captures(qtbot, tmp_path):
    from PySide6.QtWidgets import QDoubleSpinBox

    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    assert window.load_image_path(sample_image(tmp_path))
    window.show_dots_toggle.setChecked(False)
    window.simulation.gravity_strength = 0
    # Choose the source for this origin; the fitted image moves with canvas width.
    field = window.simulation.image_field
    group = field.sample_groups(
        np.array([[100.0, 200.0]]), window.simulation.width, window.simulation.height
    )[0]
    dot = next(dot for dot in window.simulation.dots if dot.palette_index == group)
    dot.x, dot.y = 200, 200

    def capture_from(x):
        window.simulation.add_ball(x, 200, 1200, 0)
        for _ in range(20):
            window.simulation.step(1 / 120)
            if window.simulation.ball_count == 0:
                break
        assert window.simulation.ball_count == 0

    capture_from(100)
    window.refresh()
    original = window.canvas.grab().toImage()
    assert original.pixelColor(99, 200).getRgb()[:3] == dot.color
    size = window.findChild(QDoubleSpinBox, "origin_mark_size")
    assert size is not None
    assert size.value() == 2.5
    size.setValue(1.0)
    smaller = window.canvas.grab().toImage()
    assert smaller.pixelColor(100, 200).getRgb()[:3] == dot.color
    assert smaller.pixelColor(99, 200).name() == "#000000"
    capture_from(120)
    image = window.canvas.grab().toImage()
    assert image.pixelColor(120, 200).getRgb()[:3] == dot.color
    assert image.pixelColor(119, 200).name() == "#000000"
    assert len(window.simulation.specks) == 2
    size.setValue(2.5)
    restored = window.canvas.grab().toImage()
    assert restored.pixelColor(99, 200).getRgb()[:3] == dot.color
    assert restored.pixelColor(119, 200).getRgb()[:3] == dot.color


def test_image_stream_gently_favours_contrast_but_manual_batches_stay_uniform(qtbot):
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, 16:] = 255
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.simulation = Simulation(320, 320, seed=42)
    window.simulation.set_image(ImageField.from_pixels(pixels, 2))
    window.stream_rate.setValue(10000)
    window.stream_toggle.setChecked(True)
    for _ in range(40):
        window.spawn_for_frame(0.1)
    streamed = window.simulation.origins.copy()
    stream_edge_fraction = np.mean((streamed[:, 0] >= 150) & (streamed[:, 0] < 170))
    window.clear_balls()
    window.pending_spawns = 40000
    for _ in range(40):
        window.spawn_for_frame(0)
    manual = window.simulation.origins
    manual_edge_fraction = np.mean((manual[:, 0] >= 150) & (manual[:, 0] < 170))
    assert 1.15 < stream_edge_fraction / manual_edge_fraction < 1.65
    assert stream_edge_fraction < 0.12  # Broad image coverage, not edge-only painting.
    assert window.simulation.ball_count == 40000
    assert (window.simulation.target_groups >= 0).all()


def sample_image(tmp_path):
    image = QImage(4, 4, QImage.Format.Format_RGB32)
    image.fill(QColor(255, 0, 0))
    for y in range(4):
        for x in range(2, 4):
            image.setPixelColor(x, y, QColor(0, 0, 255))
    path = tmp_path / "red-blue.png"
    assert image.save(str(path))
    return path


def test_colours_on_load_accepts_more_than_32_and_creates_palette_dots(qtbot, tmp_path):
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.palette_count.setValue(64)
    assert window.palette_count.value() == 64
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    expected = set()
    for index in range(64):
        color = (index * 4, 0, 255)
        expected.add(color)
        image.setPixelColor(index % 8, index // 8, QColor(*color))
    path = tmp_path / "64-colours.png"
    assert image.save(str(path))
    assert window.load_image_path(path)
    assert len(window.simulation.dots) == 64
    assert {dot.color for dot in window.simulation.dots} == expected


def test_large_palette_allocation_failure_preserves_scene(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    dot = window.simulation.add_dot(50, 50, (255, 0, 0))
    warnings = []
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a))

    def out_of_memory(*args):
        raise MemoryError("Not enough memory for image gravity dots.")

    monkeypatch.setattr(ImageField, "dot_seeds", out_of_memory)
    assert not window.load_image_path(sample_image(tmp_path))
    assert window.simulation.dots == [dot]
    assert window.simulation.image_field is None
    assert warnings and "memory" in warnings[0][-1]


def test_image_stream_keeps_frame_budget_and_manual_queue_accounting(qtbot):
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.simulation.set_image(ImageField.from_pixels(np.zeros((2, 2, 3), dtype=np.uint8), 2))
    window.stream_rate.setValue(100)
    window.stream_toggle.setChecked(True)
    window.pending_spawns = 1500
    window.spawn_for_frame(0.1)
    assert window.simulation.ball_count == 1000
    assert window.pending_spawns == 510
    window.spawn_for_frame(0.1)
    assert window.simulation.ball_count == 1520
    assert window.pending_spawns == 0


def test_hide_dots_changes_only_drawing_not_gravity_or_capture(qtbot):
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    toggle = window.findChild(QCheckBox, "show_gravity_dots")
    assert toggle is not None
    assert toggle.isChecked()
    window.simulation.add_dot(100, 100, (255, 0, 0))
    assert window.canvas.grab().toImage().pixelColor(100, 100).name() == "#ff0000"
    qtbot.mouseClick(toggle, Qt.MouseButton.LeftButton, pos=QPoint(8, toggle.height() // 2))
    assert window.canvas.grab().toImage().pixelColor(100, 100).name() == "#000000"
    window.set_mode("ball")
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(100, 100))
    assert window.simulation.ball_count == 1
    window.simulation.step(1 / 120)
    assert window.simulation.ball_count == 0
    assert window.simulation.captured_count == 1
    assert len(window.simulation.dots) == 1


def test_show_dots_keeps_black_image_sources_visible_with_an_outline(qtbot):
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    window.simulation.add_dot(100, 100, (0, 0, 0))
    assert window.canvas.grab().toImage().pixelColor(112, 100).lightness() > 0
    window.show_dots_toggle.setChecked(False)
    assert window.canvas.grab().toImage().pixelColor(112, 100).name() == "#000000"


def test_load_image_creates_palette_sources_and_routes_streamed_balls(qtbot, tmp_path, monkeypatch):
    path = sample_image(tmp_path)
    window = MainWindow(Simulation(seed=33))
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    button = window.findChild(QPushButton, "load_image")
    assert button is not None
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert window.image_mode_toggle.isChecked()
    assert window.simulation.image_mode
    assert window.simulation.paint_spawn_points
    assert len(window.simulation.dots) == 2
    assert {dot.color for dot in window.simulation.dots} == {(255, 0, 0), (0, 0, 255)}
    window.stream_toggle.setChecked(True)
    window.stream_rate.setValue(100)
    # Inspect spawning before physics can legitimately capture a nearby ball.
    window.spawn_for_frame(0.1)
    assert window.simulation.ball_count == 10
    assert (window.simulation.target_groups >= 0).all()
    origins = window.simulation.origins.copy()
    expected = []
    for i, group in enumerate(window.simulation.target_groups):
        dot = next(dot for dot in window.simulation.dots if dot.palette_index == group)
        expected.append(dot.color)
        window.simulation.positions[i] = (dot.x, dot.y)
    window.simulation.step(1 / 120)
    assert window.simulation.ball_count == 0
    assert [mark.color for mark in window.simulation.specks] == expected
    np.testing.assert_allclose([[mark.x, mark.y] for mark in window.simulation.specks], origins)
    window.image_mode_toggle.setChecked(False)
    assert not window.simulation.image_mode


def test_reset_clears_image_mode_and_restores_dot_visibility(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    assert window.load_image_path(sample_image(tmp_path))
    window.show_dots_toggle.setChecked(False)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    window.request_reset()
    assert window.show_dots_toggle.isChecked()
    assert not window.image_mode_toggle.isChecked()
    assert not window.image_mode_toggle.isEnabled()
    assert window.simulation.image_field is None
    assert window.image_status.text() == "No image loaded"


def test_cancelled_or_invalid_image_import_preserves_scene(qtbot, tmp_path, monkeypatch):
    window = MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.simulation.add_dot(50, 50, (255, 0, 0))
    path = sample_image(tmp_path)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a: (str(path), ""))
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.No)
    button = window.findChild(QPushButton, "load_image")
    assert button is not None
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert window.simulation.image_field is None and len(window.simulation.dots) == 1
    path.write_bytes(b"bad image")
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a: warnings.append(a))
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    assert warnings
    assert len(window.simulation.dots) == 1
