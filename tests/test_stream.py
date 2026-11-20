"""Exercise stream controls and frame scheduling through the running Qt window."""

import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox, QPushButton

from gravity_studio import ui

LEFT_CLICK = Qt.MouseButton.LeftButton


@pytest.fixture
def stream_scene(qtbot, monkeypatch):
    now = 0.0
    monkeypatch.setattr(ui, "perf_counter", lambda: now)
    window = ui.MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()

    def frame(duration, *, deliver=True):
        nonlocal now
        now += duration
        if deliver:
            window.timer.timeout.emit()

    return window, frame


def edit_count(qtbot, spin, count):
    qtbot.mouseClick(spin, LEFT_CLICK)
    qtbot.keyClick(spin, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(spin, str(count))
    qtbot.keyClick(spin, Qt.Key.Key_Return)


def toggle_stream(qtbot, window):
    checkbox = window.stream_toggle
    window.settings_sidebar.ensureWidgetVisible(checkbox)
    qtbot.mouseClick(checkbox, LEFT_CLICK, pos=QPoint(8, checkbox.height() // 2))


def click_action(qtbot, window, name):
    button = window.findChild(QPushButton, name)
    window.settings_sidebar.ensureWidgetVisible(button)
    qtbot.mouseClick(button, LEFT_CLICK)


def test_stream_runs_on_the_qt_timer_and_stops_when_disabled(qtbot):
    window = ui.MainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.simulation.ball_count == 0

    toggle_stream(qtbot, window)
    qtbot.waitUntil(lambda: window.simulation.ball_count > 0)
    toggle_stream(qtbot, window)
    stopped_at = window.simulation.ball_count

    with qtbot.waitSignal(window.timer.timeout):
        pass
    assert window.simulation.ball_count == stopped_at


def test_stream_counts_fractional_spawns_and_handles_rate_changes_and_pause(qtbot, stream_scene):
    window, frame = stream_scene
    edit_count(qtbot, window.stream_rate, 7)
    frame(60, deliver=False)
    toggle_stream(qtbot, window)
    frame(0)
    assert window.simulation.ball_count == 0

    # One second split into unequal frames, many shorter than the spawn interval.
    for duration in (0.03125, 0.09375, 0.0625, 0.1875, 0.15625, 0.21875, 0.25):
        frame(duration)
    assert window.simulation.ball_count == 7

    frame(0.125)
    edit_count(qtbot, window.stream_rate, 8)
    frame(0.0625)
    assert window.simulation.ball_count == 7
    frame(0.0625)
    assert window.simulation.ball_count == 8

    qtbot.mouseClick(window.pause_button, LEFT_CLICK)
    frame(60)
    assert window.simulation.ball_count == 8
    frame(60, deliver=False)
    qtbot.mouseClick(window.pause_button, LEFT_CLICK)
    frame(0)
    assert window.simulation.ball_count == 8
    frame(0.125)
    assert window.simulation.ball_count == 9


def test_stalled_stream_drops_backlog_and_batch_uses_remaining_frame_budget(qtbot, stream_scene):
    window, frame = stream_scene
    edit_count(qtbot, window.stream_rate, 800)
    edit_count(qtbot, window.batch_count, 900)
    toggle_stream(qtbot, window)
    click_action(qtbot, window, "spawn_random")

    frame(60)
    assert window.simulation.ball_count == 1000
    frame(0)
    assert window.simulation.ball_count == 1100

    edit_count(qtbot, window.stream_rate, 10_000)
    frame(60)
    assert window.simulation.ball_count == 2100
    frame(0)
    assert window.simulation.ball_count == 2100


def test_clear_disables_stream_and_confirmed_reset_restores_rate(qtbot, stream_scene):
    window, frame = stream_scene
    edit_count(qtbot, window.stream_rate, 250)
    toggle_stream(qtbot, window)
    frame(0.125)
    assert window.simulation.ball_count == 31
    click_action(qtbot, window, "spawn_random")

    window.settings_view.setCurrentText("Manage scene")
    click_action(qtbot, window, "clear_balls")
    frame(1)
    assert window.simulation.ball_count == 0
    assert not window.stream_toggle.isChecked()

    window.settings_view.setCurrentText("Build a scene")
    toggle_stream(qtbot, window)
    frame(0.125)
    assert window.simulation.ball_count == 31
    window.settings_view.setCurrentText("Manage scene")

    confirmed = False

    def confirm_reset():
        nonlocal confirmed
        dialog = QApplication.activeModalWidget()
        if isinstance(dialog, QMessageBox):
            yes = dialog.button(QMessageBox.StandardButton.Yes)
            if yes is not None:
                confirmed = True
                yes.click()
                return
        if dialog is not None:
            dialog.hide()

    def dismiss_modal():
        dialog = QApplication.activeModalWidget()
        if dialog is not None:
            dialog.hide()

    watchdog = QTimer(window)
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(dismiss_modal)
    watchdog.start(1000)
    QTimer.singleShot(0, confirm_reset)
    try:
        click_action(qtbot, window, "reset_everything")
    finally:
        watchdog.stop()
    assert confirmed
    frame(1)
    assert window.simulation.ball_count == 0
    assert not window.stream_toggle.isChecked()

    window.settings_view.setCurrentText("Build a scene")
    toggle_stream(qtbot, window)
    frame(0.125)
    assert window.simulation.ball_count == 12
