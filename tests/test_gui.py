"""Drive the sandbox through its native Qt controls and rendered canvas."""

import os
import subprocess
import sys

import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QFontDatabase, QImage
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QDialog,
    QMessageBox,
    QPushButton,
    QToolButton,
)

from gravity_studio.model import Simulation
from gravity_studio.ui import MainWindow

LEFT = Qt.MouseButton.LeftButton
RIGHT = Qt.MouseButton.RightButton


def open_window(qtbot, qapp):
    editor = MainWindow(Simulation(seed=921))
    qtbot.addWidget(editor)
    editor.timer.stop()
    editor.show()
    qapp.processEvents()
    return editor


@pytest.fixture
def window(qtbot, qapp):
    return open_window(qtbot, qapp)


def click_control(qtbot, window, control, *, pos=None):
    if window.settings_sidebar.widget().isAncestorOf(control):
        window.settings_sidebar.ensureWidgetVisible(control)
    if pos is None:
        qtbot.mouseClick(control, LEFT)
    else:
        qtbot.mouseClick(control, LEFT, pos=pos)


def button(window, name):
    return window.findChild(QPushButton, name)


def select_workflow(qtbot, window, name):
    selector = window.settings_view
    workflow = selector.findText(name)
    assert workflow >= 0, name
    selector.setFocus()
    qtbot.keyClick(selector, Qt.Key.Key_Home)
    for _ in range(workflow):
        qtbot.keyClick(selector, Qt.Key.Key_Down)
    QApplication.processEvents()


def type_number(qtbot, window, control, value):
    window.settings_sidebar.ensureWidgetVisible(control)
    editor = control.lineEdit()
    qtbot.mouseClick(editor, LEFT)
    qtbot.keyClick(editor, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(editor, str(value))
    qtbot.keyClick(editor, Qt.Key.Key_Return)


def interact_with_modal(trigger, dialog_type, interact):
    seen = []

    def dismiss_dialog():
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QDialog):
            dialog.reject()

    def on_open():
        dialog = QApplication.activeModalWidget()
        seen.append(dialog)
        if isinstance(dialog, dialog_type):
            try:
                interact(dialog)
            finally:
                if dialog.isVisible():
                    dialog.reject()
        elif isinstance(dialog, QDialog):
            dialog.reject()

    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(dismiss_dialog)
    watchdog.start(1000)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(on_open)
    timer.start(0)
    try:
        trigger()
    finally:
        timer.stop()
        watchdog.stop()
    assert len(seen) == 1 and isinstance(seen[0], dialog_type)


def test_places_drags_recolors_and_erases_canvas_entities(qtbot, window):
    canvas = window.canvas
    location = QPoint(120, 120)
    moved = QPoint(180, 160)
    qtbot.mouseClick(canvas, LEFT, pos=location)
    qtbot.mousePress(canvas, LEFT, pos=location)
    qtbot.mouseMove(canvas, moved)
    qtbot.mouseRelease(canvas, LEFT, pos=moved)

    color = QColor(86, 182, 247)

    def choose_color(dialog):
        dialog.setCurrentColor(color)
        dialog.accept()

    interact_with_modal(
        lambda: click_control(qtbot, window, window.color_button),
        QColorDialog,
        choose_color,
    )
    assert [(dot.x, dot.y) for dot in window.simulation.dots] == [(moved.x(), moved.y())]
    assert canvas.grab().toImage().pixelColor(moved) == color

    click_control(qtbot, window, window.tool_buttons["ball"])
    ball_origin = QPoint(200, 160)
    qtbot.mouseClick(canvas, LEFT, pos=ball_origin)
    assert window.simulation.origins.tolist() == [[ball_origin.x(), ball_origin.y()]]
    qtbot.keyClick(canvas, Qt.Key.Key_G)
    qtbot.mouseClick(canvas, LEFT, pos=location)
    assert canvas.grab().toImage().pixelColor(location) == color
    click_control(qtbot, window, button(window, "swatch_2"))
    image = canvas.grab().toImage()
    assert image.pixelColor(location) != color
    assert image.pixelColor(moved) == color

    # One click falls inside both erase hit areas. The dot takes priority.
    erase = QPoint(192, 160)
    qtbot.mouseClick(canvas, RIGHT, pos=erase)
    assert [(dot.x, dot.y) for dot in window.simulation.dots] == [(location.x(), location.y())]
    assert window.simulation.ball_count == 1
    qtbot.mouseClick(canvas, RIGHT, pos=erase)
    assert window.simulation.ball_count == 0
    qtbot.mousePress(canvas, LEFT, pos=location)
    qtbot.mouseMove(canvas, QPoint(-20, -20))
    qtbot.mouseRelease(canvas, LEFT, pos=QPoint(-20, -20))
    assert [(dot.x, dot.y) for dot in window.simulation.dots] == [(0, 0)]


def test_captures_paint_origins_and_clear_actions_are_independent(qtbot, window):
    select_workflow(qtbot, window, "Paint with gravity")
    click_control(
        qtbot, window, window.paint_toggle, pos=QPoint(8, window.paint_toggle.height() // 2)
    )
    canvas = window.canvas
    origin = QPoint(100, 100)
    target = QPoint(150, 100)
    fringe = QPoint(120, 100)
    qtbot.mouseClick(canvas, LEFT, pos=target)
    capturing_color = canvas.grab().toImage().pixelColor(target)
    background = canvas.grab().toImage().pixelColor(fringe)
    assert capturing_color != background
    qtbot.keyClick(canvas, Qt.Key.Key_B)
    qtbot.mouseClick(canvas, LEFT, pos=origin)
    assert window.simulation.ball_count == 1

    window.timer.start()
    try:
        qtbot.waitUntil(lambda: window.simulation.ball_count == 0, timeout=5000)
    finally:
        window.timer.stop()
    frame = canvas.grab().toImage()
    assert frame.pixelColor(origin) == capturing_color
    assert frame.pixelColor(fringe) == background

    type_number(qtbot, window, window.origin_mark_size, 60)
    assert canvas.grab().toImage().pixelColor(fringe) == capturing_color
    second_origin = QPoint(200, 100)
    second_fringe = QPoint(220, 100)
    assert canvas.grab().toImage().pixelColor(second_fringe) == background
    qtbot.mouseClick(canvas, LEFT, pos=second_origin)
    for _ in range(120):
        window.advance(1 / 60)
    assert window.simulation.ball_count == 0
    assert canvas.grab().toImage().pixelColor(second_fringe) == capturing_color
    window.resize(window.width() - 100, window.height() - 60)
    assert canvas.grab().toImage().pixelColor(fringe) == capturing_color
    type_number(qtbot, window, window.origin_mark_size, 1)
    assert canvas.grab().toImage().pixelColor(fringe) == background
    assert canvas.grab().toImage().pixelColor(second_fringe) == background
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(280, 100))

    select_workflow(qtbot, window, "Manage scene")
    click_control(qtbot, window, button(window, "clear_balls"))
    assert window.simulation.ball_count == 0
    assert canvas.grab().toImage().pixelColor(origin) == capturing_color
    assert len(window.simulation.dots) == 1
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(280, 100))
    click_control(qtbot, window, button(window, "clear_paint"))
    assert canvas.grab().toImage().pixelColor(origin) == background
    assert len(window.simulation.dots) == window.simulation.ball_count == 1


def test_workflow_navigation_preserves_scene_settings_and_folded_controls(qtbot, window):
    canvas = window.canvas
    assert window.color_button.isVisible()
    assert window.batch_count.isVisible()
    assert not window.paint_toggle.isVisible()
    assert not window.drag_spin.isVisible()
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(140, 100))
    qtbot.keyClick(canvas, Qt.Key.Key_R)
    dot = window.simulation.dots[0]
    queued = window.pending_spawns
    place = window.findChild(QToolButton, "section_place_toggle")
    click_control(qtbot, window, place)
    assert not window.color_button.isVisible()
    assert window.batch_count.isVisible()

    select_workflow(qtbot, window, "Paint with gravity")
    assert window.paint_toggle.isVisible() and window.palette_count.isVisible()
    assert not window.batch_count.isVisible()
    type_number(qtbot, window, window.origin_mark_size, 18)
    select_workflow(qtbot, window, "Adjust physics")
    assert window.drag_spin.isVisible()
    assert not window.paint_toggle.isVisible()
    type_number(qtbot, window, window.drag_spin, 1.25)
    select_workflow(qtbot, window, "Manage scene")
    assert button(window, "clear_balls").isVisible()
    assert not window.drag_spin.isVisible()
    select_workflow(qtbot, window, "All settings")
    assert window.drag_spin.isVisible() and window.paint_toggle.isVisible()
    assert not window.color_button.isVisible()
    assert window.origin_mark_size.value() == 18

    before = canvas.width()
    click_control(qtbot, window, window.settings_toggle)
    assert not window.settings_view.isVisible()
    QApplication.processEvents()
    assert canvas.width() > before
    window.settings_toggle.setFocus()
    qtbot.keyClick(window.settings_toggle, Qt.Key.Key_Space)
    assert window.settings_view.isVisible()
    assert not window.color_button.isVisible()
    window.settings_sidebar.ensureWidgetVisible(place)
    place.setFocus()
    qtbot.keyClick(place, Qt.Key.Key_Space)
    assert window.color_button.isVisible()

    select_workflow(qtbot, window, "Build a scene")
    qtbot.keyClick(window.settings_view, Qt.Key.Key_Down)
    assert window.paint_toggle.isVisible()
    qtbot.keyClick(window.settings_view, Qt.Key.Key_R)
    qtbot.keyClick(window.settings_view, Qt.Key.Key_Space)
    qtbot.keyClick(window.settings_view, Qt.Key.Key_Escape)
    assert window.pending_spawns == queued > 0
    assert window.simulation.dots == [dot]
    assert window.simulation.drag == window.drag_spin.value() == 1.25
    assert not window.paused


def test_full_numeric_values_fit_at_smallest_supported_window(qtbot, qapp):
    original_font = qapp.font()
    qapp.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
    try:
        window = open_window(qtbot, qapp)
    finally:
        qapp.setFont(original_font)
    window.resize(window.minimumSize())
    select_workflow(qtbot, window, "All settings")
    for control in (
        window.drag_spin,
        window.strength_spin,
        window.origin_mark_size,
        window.palette_count,
        window.stream_rate,
        window.batch_count,
    ):
        type_number(qtbot, window, control, control.maximum())
        assert control.value() == control.maximum()
        editor = control.lineEdit()
        margins = editor.textMargins()
        available = editor.contentsRect().width() - margins.left() - margins.right()
        required = editor.fontMetrics().horizontalAdvance(editor.displayText())
        assert available >= required, control.accessibleName()


def test_adjusting_physics_controls_changes_ball_motion(qtbot, window):
    select_workflow(qtbot, window, "Adjust physics")
    type_number(qtbot, window, window.strength_spin, 0)
    type_number(qtbot, window, window.drag_spin, 0)
    assert window.strength_slider.value() == window.strength_slider.minimum()
    assert window.drag_slider.value() == window.drag_slider.minimum()
    canvas = window.canvas
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(200, 120))
    qtbot.keyClick(canvas, Qt.Key.Key_B)
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(100, 120))
    window.advance(1 / 30)
    assert window.simulation.positions[0].tolist() == [100, 120]

    window.strength_slider.setFocus()
    qtbot.keyClick(window.strength_slider, Qt.Key.Key_End)
    assert window.strength_spin.value() == window.strength_spin.maximum()
    window.advance(1 / 30)
    assert window.simulation.positions[0, 0] > 100
    moving_speed = window.simulation.velocities[0, 0]
    type_number(qtbot, window, window.strength_spin, 0)
    window.advance(1 / 30)
    assert window.simulation.velocities[0, 0] == pytest.approx(moving_speed)

    window.drag_slider.setFocus()
    qtbot.keyClick(window.drag_slider, Qt.Key.Key_End)
    assert window.drag_spin.value() == window.drag_spin.maximum()
    window.advance(1 / 30)
    assert 0 < window.simulation.velocities[0, 0] < moving_speed


def test_keyboard_pause_freezes_motion_even_from_an_action_button(qtbot, window):
    canvas = window.canvas
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(180, 100))
    qtbot.keyClick(canvas, Qt.Key.Key_B)
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(100, 100))
    window.advance(10)
    before = window.simulation.positions.copy()
    assert 100 < before[0, 0] < 160
    spawn = button(window, "spawn_random")
    window.settings_sidebar.ensureWidgetVisible(spawn)
    spawn.setFocus()
    qtbot.keyClick(spawn, Qt.Key.Key_Space)
    window.advance(10)
    assert window.pending_spawns == 0
    assert window.simulation.positions.tolist() == before.tolist()
    click_control(qtbot, window, window.pause_button)
    window.advance(1 / 120)
    assert window.simulation.positions[0, 0] > before[0, 0]


def test_manual_batches_spawn_in_chunks_and_cancel_without_removing_balls(qtbot, window):
    type_number(qtbot, window, window.batch_count, 2501)
    qtbot.keyClick(window.batch_count.lineEdit(), Qt.Key.Key_R)
    assert window.pending_spawns == 0
    spawn = button(window, "spawn_random")
    cancel = button(window, "cancel_spawn")
    click_control(qtbot, window, spawn)
    assert cancel.isEnabled()
    window.advance(0)
    assert 0 < window.simulation.ball_count <= 1000
    click_control(qtbot, window, cancel)
    partial = window.simulation.ball_count
    window.advance(0)
    assert window.simulation.ball_count == partial
    assert not cancel.isEnabled()

    # Global spawning remains available from the canvas while physics is paused.
    qtbot.keyClick(window.canvas, Qt.Key.Key_Space)
    qtbot.keyClick(window.canvas, Qt.Key.Key_R)
    for _ in range(3):
        window.advance(0)
    assert window.simulation.ball_count == partial + 2501
    assert not cancel.isEnabled()


def test_large_random_batch_is_drawn_on_canvas_and_clears(qtbot, window):
    canvas = window.canvas
    blank = canvas.grab().toImage()
    type_number(qtbot, window, window.batch_count, 100_000)
    click_control(qtbot, window, button(window, "spawn_random"))
    for _ in range(100):
        window.advance(0)
    assert window.simulation.ball_count == 100_000
    rendered = canvas.grab().toImage()
    centers = [QPoint(round(x), round(y)) for x, y in window.simulation.positions[:20]]
    for center in centers:
        assert rendered.pixelColor(center) != blank.pixelColor(center)
        assert rendered.pixelColor(center) == rendered.pixelColor(center + QPoint(2, 0))
    click_control(qtbot, window, button(window, "spawn_random"))
    select_workflow(qtbot, window, "Manage scene")
    click_control(qtbot, window, button(window, "clear_balls"))
    window.advance(0)
    assert window.pending_spawns == window.simulation.ball_count == 0
    cleared = canvas.grab().toImage()
    assert all(cleared.pixelColor(center) == blank.pixelColor(center) for center in centers)


def test_reset_confirmation_preserves_scene_until_accepted(qtbot, window):
    canvas = window.canvas
    select_workflow(qtbot, window, "Paint with gravity")
    click_control(
        qtbot, window, window.paint_toggle, pos=QPoint(8, window.paint_toggle.height() // 2)
    )
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(150, 100))
    qtbot.keyClick(canvas, Qt.Key.Key_B)
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(110, 100))
    for _ in range(90):
        window.advance(1 / 60)
    assert len(window.simulation.specks) == 1
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(270, 100))
    qtbot.keyClick(canvas, Qt.Key.Key_R)
    queued = window.pending_spawns
    type_number(qtbot, window, window.origin_mark_size, 30)
    select_workflow(qtbot, window, "Adjust physics")
    type_number(qtbot, window, window.drag_spin, 2)
    select_workflow(qtbot, window, "Manage scene")
    observed = []

    def respond(answer):
        def answer_dialog(dialog):
            observed.append((window.timer.isActive(), window.pending_spawns))
            qtbot.mouseClick(dialog.button(answer), LEFT)

        interact_with_modal(
            lambda: click_control(qtbot, window, button(window, "reset_everything")),
            QMessageBox,
            answer_dialog,
        )

    window.timer.start()
    try:
        respond(QMessageBox.StandardButton.No)
        assert window.timer.isActive()
        assert observed == [(False, queued)]
        assert window.pending_spawns == queued > 0
        assert len(window.simulation.dots) == window.simulation.ball_count == 1
        assert len(window.simulation.specks) == 1
        assert window.simulation.drag == 2
        assert window.origin_mark_size.value() == 30
        respond(QMessageBox.StandardButton.Yes)
    finally:
        window.timer.stop()
    assert window.pending_spawns == window.simulation.ball_count == 0
    assert not window.simulation.dots
    assert not window.simulation.specks
    assert window.simulation.drag == window.drag_spin.value() == 0.3
    assert window.origin_mark_size.value() == 2.5
    qtbot.mouseClick(canvas, LEFT, pos=QPoint(150, 100))
    assert len(window.simulation.dots) == 1


def test_demo_starts_only_on_request_and_populates_the_scene(qtbot, window):
    select_workflow(qtbot, window, "Manage scene")
    assert window.simulation.ball_count == window.pending_spawns == 0
    assert not window.simulation.dots
    click_control(qtbot, window, button(window, "demo_scene"))
    assert len(window.simulation.dots) == 4
    assert len({dot.color for dot in window.simulation.dots}) == 4
    assert window.simulation.paint_spawn_points
    for _ in range(3):
        window.advance(0)
    assert window.simulation.ball_count == 3000
    assert window.pending_spawns == 0


def test_allocation_failure_stops_spawns_and_user_can_recover(qtbot, window, monkeypatch):
    click_control(
        qtbot,
        window,
        window.stream_toggle,
        pos=QPoint(8, window.stream_toggle.height() // 2),
    )
    assert window.stream_toggle.isChecked()
    click_control(qtbot, window, button(window, "spawn_random"))

    def allocation_fails(additional):
        raise MemoryError

    with monkeypatch.context() as fault:
        fault.setattr(window.simulation, "_reserve", allocation_fails)
        window.tick()
    assert window.paused
    assert not window.stream_toggle.isChecked()
    assert window.pending_spawns == 0
    assert window.error_message.isVisible()

    select_workflow(qtbot, window, "Manage scene")
    click_control(qtbot, window, button(window, "clear_balls"))
    click_control(qtbot, window, window.pause_button)
    select_workflow(qtbot, window, "Build a scene")
    type_number(qtbot, window, window.batch_count, 17)
    click_control(qtbot, window, button(window, "spawn_random"))
    window.advance(0)
    assert window.simulation.ball_count == 17
    assert not window.paused
    assert not window.error_message.isVisible()


def test_native_module_launch_writes_a_readable_screenshot(tmp_path):
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
    assert not QImage(str(screenshot)).isNull()
