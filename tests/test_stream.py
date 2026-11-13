"""Deterministic real-Qt integration checks for continuous random spawning."""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QSpinBox

from gravity_studio.ui import MainWindow


@pytest.mark.parametrize("fps", [30, 60, 120])
def test_continuous_stream_spawns_the_selected_number_per_second(qtbot, fps):
    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.show()
    toggle = window.findChild(QCheckBox, "continuous_stream")
    rate = window.findChild(QSpinBox, "stream_rate")
    assert toggle is not None, "Continuous spawning needs a visible on/off control"
    assert rate is not None, "The spawn rate needs an editable balls/second control"
    assert not toggle.isChecked()
    rate.setValue(7)
    qtbot.mouseClick(toggle, Qt.MouseButton.LeftButton)
    for _ in range(fps):
        window.advance(1 / fps)
    assert window.simulation.ball_count == 7
    assert window.pending_spawns == 0
    assert window.simulation.positions[:, 0].max() > window.simulation.positions[:, 0].min()


def test_stream_stops_immediately_and_pause_does_not_accumulate_time(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.stream_rate.setValue(10)
    window.stream_toggle.setChecked(True)
    window.advance(0.1)
    assert window.simulation.ball_count == 1
    window.toggle_pause()
    window.advance(10)
    assert window.simulation.ball_count == 1
    window.toggle_pause()
    window.advance(0)
    assert window.simulation.ball_count == 1
    window.advance(0.1)
    assert window.simulation.ball_count == 2
    window.stream_toggle.setChecked(False)
    window.advance(1)
    assert window.simulation.ball_count == 2


def test_clear_balls_stops_stream_and_reset_restores_defaults(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.stream_rate.setValue(250)
    window.stream_toggle.setChecked(True)
    window.advance(0.1)
    window.clear_balls()
    assert not window.stream_toggle.isChecked()
    window.advance(1)
    assert window.simulation.ball_count == 0
    window.stream_toggle.setChecked(True)
    monkeypatch.setattr(QMessageBox, "question", lambda *a: QMessageBox.StandardButton.Yes)
    window.request_reset()
    assert not window.stream_toggle.isChecked()
    assert window.stream_rate.value() == 100
    window.advance(0.1)
    assert window.simulation.ball_count == 0


def test_stalled_stream_has_bounded_work_and_no_backlog(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.stream_rate.setValue(10000)
    window.stream_toggle.setChecked(True)
    window.advance(60)
    assert window.simulation.ball_count == 1000
    window.advance(0)
    assert window.simulation.ball_count == 1000
    assert window.pending_spawns == 0


def test_manual_batch_still_works_alongside_stream(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.stream_rate.setValue(10)
    window.stream_toggle.setChecked(True)
    window.batch_count.setValue(20)
    window.queue_random()
    window.advance(0.1)
    assert window.simulation.ball_count == 21
    assert window.pending_spawns == 0


def test_enabling_stream_resets_clock_and_changing_rate_is_live(qtbot, monkeypatch):
    from gravity_studio import ui

    now = [100.0]
    monkeypatch.setattr(ui, "perf_counter", lambda: now[0])
    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.stream_rate.setValue(10)
    now[0] = 1000.0
    window.stream_toggle.setChecked(True)
    now[0] += 0.1
    window.tick()
    assert window.simulation.ball_count == 1
    window.stream_rate.setValue(20)
    now[0] += 0.1
    window.tick()
    assert window.simulation.ball_count == 3
    assert "20 balls/s" in window.queue_label.text()


def test_allocation_failure_also_turns_off_stream(qtbot, monkeypatch):
    window = MainWindow()
    qtbot.addWidget(window)
    window.timer.stop()
    window.stream_toggle.setChecked(True)

    def out_of_memory(count):
        raise MemoryError

    monkeypatch.setattr(window.simulation, "spawn_random", out_of_memory)
    window._last_frame -= 0.1
    window.tick()
    assert window.paused
    assert not window.stream_toggle.isChecked()
