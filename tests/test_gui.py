"""Native Qt integration tests; run with QT_QPA_PLATFORM=offscreen."""

from importlib import import_module

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QPushButton


def test_larger_origin_marks_redraw_existing_paint_and_reset(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    window.paint_toggle.setChecked(True)
    window.simulation.add_dot(200, 200, (240, 112, 125))
    window.simulation.add_ball(80, 80)
    window.simulation.positions[0] = (200, 200)
    window.advance(1 / 120)
    assert window.canvas.grab().toImage().pixelColor(90, 80).name() == "#000000"
    window.origin_mark_size.setValue(100.0)
    assert window.origin_mark_size.value() == 100.0
    large = window.canvas.grab().toImage()
    assert large.pixelColor(120, 80).getRgb()[:3] == (240, 112, 125)
    assert large.pixelColor(131, 80).name() == "#000000"
    window.origin_mark_size.setValue(1.0)
    assert window.canvas.grab().toImage().pixelColor(90, 80).name() == "#000000"
    assert len(window.simulation.specks) == 1
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    window.request_reset()
    assert window.origin_mark_size.value() == 2.5


def test_settings_panel_can_hide_and_restore_without_losing_values(qtbot):
    from PySide6.QtWidgets import QScrollArea

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    sidebar = window.findChild(QScrollArea, "controls_sidebar")
    toggle = window.findChild(QPushButton, "toggle_settings")
    assert toggle is not None
    window.drag_spin.setValue(1.25)
    window.origin_mark_size.setValue(12.0)
    window.simulation.add_dot(200, 200, (255, 0, 0))
    original_width = window.canvas.width()
    qtbot.mouseClick(toggle, Qt.MouseButton.LeftButton)
    qtbot.wait(10)
    assert not sidebar.isVisible()
    assert toggle.isVisible() and toggle.text() == "Show settings"
    assert window.canvas.width() > original_width
    window.advance(1 / 120)
    assert window.simulation.drag == 1.25
    qtbot.mouseClick(toggle, Qt.MouseButton.LeftButton)
    qtbot.wait(10)
    assert sidebar.isVisible() and toggle.text() == "Hide settings"
    assert window.canvas.width() == original_width
    assert window.origin_mark_size.value() == 12.0
    assert len(window.simulation.dots) == 1
    toggle.setFocus()
    qtbot.keyClick(toggle, Qt.Key.Key_Space)
    assert not sidebar.isVisible()
    assert not window.paused
    qtbot.keyClick(toggle, Qt.Key.Key_Space)
    assert sidebar.isVisible()


def test_settings_categories_expand_independently_with_mouse_and_keyboard(qtbot):
    from PySide6.QtWidgets import QToolButton, QWidget

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    window.drag_spin.setValue(1.25)
    window.origin_mark_size.setValue(12.0)
    for key, control in (
        ("place", window.color_button),
        ("physics", window.drag_spin),
        ("painting", window.origin_mark_size),
        ("image_painting", window.palette_count),
        ("random_balls", window.stream_rate),
        ("scene", window.findChild(QPushButton, "clear_paint")),
    ):
        toggle = window.findChild(QToolButton, f"section_{key}_toggle")
        content = window.findChild(QWidget, f"section_{key}_content")
        assert toggle is not None and content is not None
        window.settings_sidebar.ensureWidgetVisible(toggle)
        qtbot.wait(10)
        assert toggle.isChecked() and content.isVisible() and control.isVisible()
        assert toggle.arrowType() == Qt.ArrowType.DownArrow
        qtbot.mouseClick(toggle, Qt.MouseButton.LeftButton)
        assert not content.isVisible() and not control.isVisible()
        assert toggle.arrowType() == Qt.ArrowType.RightArrow
        window.settings_toggle.click()
        window.settings_toggle.click()
        assert not content.isVisible()  # Hiding the whole panel preserves category state.
        toggle.setFocus()
        qtbot.keyClick(toggle, Qt.Key.Key_Space)
        assert toggle.isChecked() and content.isVisible() and control.isVisible()
        assert not window.paused  # Disclosure keyboard input must not trigger global pause.
    assert window.simulation.drag == 1.25
    assert window.origin_mark_size.value() == 12.0
    toggles = window.findChildren(QToolButton)
    for toggle in toggles:
        toggle.setChecked(False)
    qtbot.wait(30)
    assert window.settings_sidebar.verticalScrollBar().maximum() == 0
    for toggle in toggles:
        assert toggle.objectName() and toggle.accessibleName() and toggle.toolTip()
        assert toggle.height() < 50


def test_paint_toggle_has_a_visible_checkmark_when_enabled(qtbot):
    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    window.paint_toggle.setChecked(True)
    image = window.paint_toggle.grab().toImage()
    middle = image.height() // 2
    assert any(
        image.pixelColor(x, y).lightness() < 80
        for x in range(6, 14)
        for y in range(middle - 4, middle + 5)
    )


def test_spawn_allocation_failure_pauses_and_cancels_queue(qtbot, monkeypatch):
    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.queue_random()

    def out_of_memory(count):
        raise MemoryError("Allocation failed")

    monkeypatch.setattr(window.simulation, "spawn_random", out_of_memory)
    window.tick()
    assert window.paused
    assert window.pending_spawns == 0
    assert "memory" in window.error_message.text().lower()
    monkeypatch.undo()
    window.clear_balls()
    window.toggle_pause()
    assert not window.paused
    assert window.error_message.text() == ""


def test_canvas_tools_place_real_entities(qtbot):
    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(150, 150))
    assert len(window.simulation.dots) == 1
    ball_tool = window.findChild(QPushButton, "ball_tool")
    qtbot.mouseClick(ball_tool, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(250, 180))
    assert window.simulation.ball_count == 1
    assert window.simulation.origins[0].tolist() == [250, 180]
    assert "1 balls" in window.counts.text()


def test_dot_selection_drag_and_color_edit(qtbot, monkeypatch):
    from PySide6.QtGui import QColor
    from PySide6.QtWidgets import QColorDialog

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))
    dot = window.simulation.dots[0]
    assert window.selected_dot is dot
    monkeypatch.setattr(QColorDialog, "getColor", lambda *a, **k: QColor("#56b6f7"))
    qtbot.mouseClick(window.findChild(QPushButton, "choose_color"), Qt.MouseButton.LeftButton)
    assert dot.color == (86, 182, 247)
    qtbot.mousePress(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))
    qtbot.mouseMove(window.canvas, QPoint(180, 160))
    qtbot.mouseRelease(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(180, 160))
    assert (dot.x, dot.y) == (180, 160)
    assert len(window.simulation.dots) == 1
    qtbot.mouseClick(window.findChild(QPushButton, "swatch_2"), Qt.MouseButton.LeftButton)
    assert dot.color == window.default_color
    assert dot.color != (86, 182, 247)


def test_physics_controls_update_model_and_synchronize(qtbot):
    from PySide6.QtWidgets import QCheckBox, QDoubleSpinBox, QSlider

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    drag = window.findChild(QDoubleSpinBox, "air_resistance")
    assert drag is not None
    drag.setValue(5.0)
    assert window.simulation.drag == 5.0
    slider = window.findChild(QSlider, "air_resistance_slider")
    assert slider.value() == 500
    slider.setValue(25)
    assert drag.value() == window.simulation.drag == 0.25
    strength = window.findChild(QDoubleSpinBox, "gravity_strength")
    strength.setValue(300000.0)
    assert window.simulation.gravity_strength == 300000.0
    paint = window.findChild(QCheckBox, "paint_origins")
    qtbot.mouseClick(paint, Qt.MouseButton.LeftButton)
    assert window.simulation.paint_spawn_points
    window.show()
    qtbot.wait(20)
    assert window.simulation.width == window.canvas.width()
    assert window.minimumWidth() == 900
    assert window.minimumHeight() == 640


def test_random_spawns_are_chunked_cancellable_and_unlimited(qtbot):
    from PySide6.QtWidgets import QSpinBox

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    batch = window.findChild(QSpinBox, "batch_count")
    assert batch is not None
    assert (batch.minimum(), batch.maximum()) == (1, 1000000)
    batch.setValue(2501)
    qtbot.mouseClick(window.findChild(QPushButton, "spawn_random"), Qt.MouseButton.LeftButton)
    assert window.simulation.ball_count == 0
    assert window.pending_spawns == 2501
    window.advance(0.0)
    assert window.simulation.ball_count == 1000
    assert window.pending_spawns == 1501
    assert "1,501" in window.queue_label.text()
    window.advance(0.0)
    window.advance(0.0)
    assert window.simulation.ball_count == 2501
    assert window.pending_spawns == 0
    qtbot.mouseClick(window.findChild(QPushButton, "spawn_random"), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.findChild(QPushButton, "cancel_spawn"), Qt.MouseButton.LeftButton)
    assert window.pending_spawns == 0
    assert window.simulation.ball_count == 2501
    qtbot.mouseClick(window.findChild(QPushButton, "spawn_random"), Qt.MouseButton.LeftButton)
    qtbot.mouseClick(window.findChild(QPushButton, "clear_balls"), Qt.MouseButton.LeftButton)
    window.advance(0.0)
    assert window.pending_spawns == window.simulation.ball_count == 0


def test_fixed_step_pause_and_keyboard_tools(qtbot):
    import pytest

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    assert window.timer.isActive()
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    window.canvas.setFocus()
    window.simulation.drag = 0
    window.simulation.add_ball(100, 100, 120, 0)
    window.advance(10.0)
    assert window.simulation.positions[0, 0] == pytest.approx(108.0)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Space)
    assert window.paused
    window.advance(10.0)
    assert window.simulation.positions[0, 0] == pytest.approx(108.0)
    qtbot.keyClick(window.canvas, Qt.Key.Key_Space)
    window.advance(1 / 120)
    assert window.simulation.positions[0, 0] == pytest.approx(109.0)
    qtbot.keyClick(window.canvas, Qt.Key.Key_B)
    assert window.mode == "ball"
    qtbot.keyClick(window.canvas, Qt.Key.Key_G)
    assert window.mode == "gravity"
    qtbot.keyClick(window.canvas, Qt.Key.Key_R)
    assert window.pending_spawns == window.batch_count.value()
    window.batch_count.setFocus()
    qtbot.keyClick(window.batch_count, Qt.Key.Key_R)
    assert window.pending_spawns == window.batch_count.value()


def test_captured_origin_paint_is_cached_persistent_and_clearable(qtbot):
    from PySide6.QtWidgets import QCheckBox

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    qtbot.mouseClick(window.findChild(QCheckBox, "paint_origins"), Qt.MouseButton.LeftButton)
    dot = window.simulation.add_dot(200, 200, (240, 112, 125))
    window.simulation.add_ball(80, 80)
    window.simulation.positions[0] = (200, 200)
    window.advance(1 / 120)
    assert window.simulation.ball_count == 0
    assert len(window.simulation.specks) == 1
    rendered = window.canvas.grab().toImage()
    assert rendered.pixelColor(80, 80).getRgb()[:3] == dot.color
    cache_key = window.canvas.paint_cache.cacheKey()
    window.advance(0.0)
    window.canvas.grab()
    assert window.canvas.paint_cache.cacheKey() == cache_key
    window.resize(1100, 740)
    qtbot.wait(10)
    assert window.canvas.grab().toImage().pixelColor(80, 80).getRgb()[:3] == dot.color
    assert "1 marks" in window.counts.text()
    qtbot.mouseClick(window.findChild(QPushButton, "clear_paint"), Qt.MouseButton.LeftButton)
    assert not window.simulation.specks
    assert window.canvas.grab().toImage().pixelColor(80, 80).name() == "#000000"


def test_right_click_erases_dot_before_ball_and_clamps_placement(qtbot):
    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    window.simulation.add_ball(120, 120)
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(120, 120))
    qtbot.mouseClick(window.canvas, Qt.MouseButton.RightButton, pos=QPoint(120, 120))
    assert len(window.simulation.dots) == 0
    assert window.selected_dot is None
    assert window.simulation.ball_count == 1
    qtbot.mouseClick(window.canvas, Qt.MouseButton.RightButton, pos=QPoint(120, 120))
    assert window.simulation.ball_count == 0
    qtbot.mouseClick(window.canvas, Qt.MouseButton.LeftButton, pos=QPoint(-5, -5))
    assert (window.simulation.dots[0].x, window.simulation.dots[0].y) == (0, 0)


def test_reset_confirmation_cancel_then_accept(qtbot):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    reset = window.findChild(QPushButton, "reset_everything")
    assert reset is not None
    qtbot.mouseClick(reset, Qt.MouseButton.LeftButton)  # Empty reset must not open a modal.
    window.simulation.add_dot(100, 100, (255, 150, 50))
    window.queue_random()
    window.drag_spin.setValue(2)

    def respond(answer):
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QMessageBox)
        qtbot.mouseClick(dialog.button(answer), Qt.MouseButton.LeftButton)

    QTimer.singleShot(0, lambda: respond(QMessageBox.StandardButton.No))
    qtbot.mouseClick(reset, Qt.MouseButton.LeftButton)
    assert len(window.simulation.dots) == 1
    assert window.pending_spawns > 0
    QTimer.singleShot(0, lambda: respond(QMessageBox.StandardButton.Yes))
    qtbot.mouseClick(reset, Qt.MouseButton.LeftButton)
    assert window.pending_spawns == window.simulation.ball_count == 0
    assert not window.simulation.dots
    assert not window.simulation.specks
    assert window.selected_dot is None
    assert window.simulation.drag == window.drag_spin.value() == 0.3
    assert window.simulation.gravity_strength == 180000.0


def test_dark_scrollable_accessible_shell_and_explicit_demo(qtbot):
    from PySide6.QtWidgets import QAbstractSpinBox, QCheckBox, QScrollArea, QSlider

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    sidebar = window.findChild(QScrollArea, "controls_sidebar")
    assert sidebar is not None
    assert sidebar.width() == 340
    assert window.simulation.ball_count == len(window.simulation.dots) == 0
    assert window.canvas.empty_hint.isVisible()
    assert window.canvas.grab().toImage().pixelColor(20, 20).name() == "#000000"
    assert window.findChild(QPushButton, "gravity_tool").isChecked()
    for widget_type in (QPushButton, QAbstractSpinBox, QCheckBox, QSlider):
        for control in window.findChildren(widget_type):
            assert control.objectName(), type(control).__name__
            assert control.accessibleName(), control.objectName()
            assert control.toolTip(), control.objectName()
    demo = window.findChild(QPushButton, "demo_scene")
    assert demo is not None
    qtbot.mouseClick(demo, Qt.MouseButton.LeftButton)
    assert len(window.simulation.dots) == 4
    assert window.pending_spawns == 3000
    assert window.simulation.paint_spawn_points
    assert not window.canvas.empty_hint.isVisible()
    window.resize(900, 640)
    qtbot.wait(10)
    assert sidebar.verticalScrollBar().maximum() > 0
    assert window.canvas.width() > 500
    # Check the actual label and editor space, including the largest palette value.
    from PySide6.QtWidgets import QLabel, QLineEdit

    for width, height in ((900, 640), (1280, 820)):
        window.resize(width, height)
        window.palette_count.setValue(window.palette_count.maximum())
        qtbot.wait(10)
        assert sidebar.horizontalScrollBar().maximum() == 0
        for label in sidebar.findChildren(QLabel):
            if label.buddy() is not None:
                assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
        for spin in sidebar.findChildren(QAbstractSpinBox):
            editor = spin.findChild(QLineEdit)
            assert editor.width() >= editor.fontMetrics().horizontalAdvance(editor.text())


def test_module_entry_point_smoke_and_screenshot(tmp_path):
    import os
    import subprocess
    import sys

    from PySide6.QtGui import QImage

    screenshot = tmp_path / "native.png"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "gravity_studio",
            "--smoke-seconds",
            "0.08",
            "--screenshot",
            str(screenshot),
        ],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    image = QImage(str(screenshot))
    assert (image.width(), image.height()) == (1280, 820)
    invalid = subprocess.run(
        [sys.executable, "-m", "gravity_studio", "--smoke-seconds", "-1"],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert invalid.returncode == 2


def test_large_cloud_renders_full_ball_radius_and_clears(qtbot):
    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    window.simulation.spawn_random(100000)
    window.simulation.positions[:] = (80, 80)
    window.refresh()
    image = window.canvas.grab().toImage()
    assert image.pixelColor(80, 80).name() == "#ffffff"
    assert image.pixelColor(82, 80).name() == "#ffffff"
    assert image.pixelColor(84, 80).name() == "#000000"
    window.clear_balls()
    assert window.canvas.grab().toImage().pixelColor(80, 80).name() == "#000000"


def test_space_pauses_even_when_an_action_button_has_focus(qtbot):
    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    spawn = window.findChild(QPushButton, "spawn_random")
    spawn.setFocus()
    qtbot.keyClick(spawn, Qt.Key.Key_Space)
    assert window.paused
    assert window.pending_spawns == 0


def test_reset_modal_freezes_queue_until_user_decides(qtbot):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    window = import_module("gravity_studio.ui").MainWindow()
    window.settings_view.setCurrentText("All settings")
    qtbot.addWidget(window)
    window.show()
    window.batch_count.setValue(2501)
    window.queue_random()
    observed = []

    def decline():
        observed.append(window.pending_spawns)
        dialog = QApplication.activeModalWidget()
        assert isinstance(dialog, QMessageBox)
        qtbot.mouseClick(dialog.button(QMessageBox.StandardButton.No), Qt.MouseButton.LeftButton)

    QTimer.singleShot(65, decline)
    window.request_reset()
    window.timer.stop()
    assert observed == [2501]
    assert window.pending_spawns == 2501


def test_workflows_show_relevant_settings_and_preserve_scene_and_disclosures(qtbot):
    from gravity_studio.ui import CollapsibleSection, MainWindow

    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    qtbot.wait(10)
    assert window.settings_view.currentText() == "Build a scene"
    sections = {
        section.toggle.text(): section for section in window.findChildren(CollapsibleSection)
    }
    assert {title for title, section in sections.items() if section.isVisible()} == {
        "Place",
        "Random balls",
    }
    window.drag_spin.setValue(1.25)
    window.batch_count.setValue(20)
    window.queue_random()
    dot = window.simulation.add_dot(200, 200, (255, 0, 0))
    sections["Place"].toggle.setChecked(False)
    for view, expected in (
        ("Paint with gravity", {"Painting", "Image painting"}),
        ("Adjust physics", {"Physics"}),
        ("Manage scene", {"Scene"}),
        ("All settings", set(sections)),
        ("Build a scene", {"Place", "Random balls"}),
    ):
        window.settings_view.setCurrentText(view)
        qtbot.wait(10)
        assert {title for title, section in sections.items() if section.isVisible()} == expected
        assert window.workflow_hint.text()
        assert window.simulation.drag == window.drag_spin.value() == 1.25
        assert window.simulation.dots == [dot]
        assert window.pending_spawns == 20
        assert not sections["Place"].toggle.isChecked()
    window.settings_toggle.click()
    assert not window.settings_view.isVisible()
    window.settings_toggle.click()
    assert window.settings_view.isVisible()
    window.settings_view.setFocus()
    qtbot.keyClick(window.settings_view, Qt.Key.Key_Down)
    assert window.settings_view.currentText() == "Paint with gravity"
    qtbot.keyClick(window.settings_view, Qt.Key.Key_R)
    qtbot.keyClick(window.settings_view, Qt.Key.Key_Space)
    assert window.pending_spawns == 20
    assert not window.paused
