import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication, QCheckBox, QFileDialog, QMessageBox, QPushButton

from gravity_studio.image_field import ImageField
from gravity_studio.model import Simulation
from gravity_studio.ui import MainWindow

LEFT = Qt.MouseButton.LeftButton
RED = (255, 0, 0)
BLUE = (0, 0, 255)


@pytest.fixture
def window(qtbot):
    window = MainWindow(Simulation(seed=33))
    qtbot.addWidget(window)
    window.show()
    window.timer.stop()  # Drive the real frame handler with elapsed simulation time.
    return window


def click_control(qtbot, window, control):
    if isinstance(control, str):
        control = window.findChild(QPushButton, control)
    assert control.isVisible()
    window.settings_sidebar.ensureWidgetVisible(control)
    if isinstance(control, QCheckBox):
        qtbot.mouseClick(control, LEFT, pos=QPoint(8, control.height() // 2))
    else:
        qtbot.mouseClick(control, LEFT)


def click_with_dialogs(qtbot, window, control, *, choose_file=False, path=None, answers=()):
    seen = []
    icons = []
    errors = []
    pending_answers = list(answers)
    file_chosen = False

    def interact():
        nonlocal file_chosen
        dialog = QApplication.activeModalWidget()
        if dialog is None or dialog in seen:
            return
        seen.append(dialog)
        if choose_file and not file_chosen and isinstance(dialog, QFileDialog):
            file_chosen = True
            if path is None:
                qtbot.keyClick(dialog, Qt.Key.Key_Escape)
            else:
                dialog.selectFile(str(path))
                qtbot.keyClick(dialog, Qt.Key.Key_Return)
        elif isinstance(dialog, QMessageBox) and pending_answers:
            icons.append(dialog.icon())
            button = dialog.button(pending_answers.pop(0))
            if button is None:
                errors.append("The expected dialog button is missing.")
                dialog.reject()
            else:
                qtbot.mouseClick(button, LEFT)
        else:
            errors.append(f"Unexpected modal {type(dialog).__name__}")
            dialog.reject()

    def time_out():
        errors.append("A dialog failed to close.")
        dialog = QApplication.activeModalWidget()
        if dialog is not None:
            dialog.reject()

    driver = QTimer()
    driver.timeout.connect(interact)
    watchdog = QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(time_out)
    driver.start(1)
    watchdog.start(3000)
    try:
        click_control(qtbot, window, control)
    finally:
        driver.stop()
        watchdog.stop()
    assert not errors, errors
    assert file_chosen == choose_file
    assert not pending_answers
    return icons


def import_image(qtbot, window, path, *answers):
    return click_with_dialogs(
        qtbot, window, "load_image", choose_file=True, path=path, answers=answers
    )


def split_image(tmp_path, *, size=4):
    image = QImage(size, size, QImage.Format.Format_RGB32)
    image.fill(QColor(*RED))
    for y in range(size):
        for x in range(size // 2, size):
            image.setPixelColor(x, y, QColor(*BLUE))
    path = tmp_path / "red-blue.png"
    assert image.save(str(path))
    return path


def test_import_batch_and_stream_paint_original_image_colours(qtbot, window, tmp_path):
    window.settings_view.setCurrentText("Paint with gravity")
    import_image(qtbot, window, split_image(tmp_path))
    sim = window.simulation
    assert {dot.color for dot in sim.dots} == {RED, BLUE}
    assert sim.image_mode and sim.paint_spawn_points

    window.settings_view.setCurrentText("Build a scene")
    window.batch_count.setValue(24)
    click_control(qtbot, window, "spawn_random")
    window.advance(0)
    click_control(qtbot, window, window.stream_toggle)
    window.advance(0.08)
    click_control(qtbot, window, window.stream_toggle)
    initial_origins = {tuple(origin) for origin in sim.origins}
    initial_origins.update((mark.x, mark.y) for mark in sim.specks)
    assert len(initial_origins) == 32

    # Hide the source markers so the only visible colour after capture comes from paint.
    click_control(qtbot, window, window.show_dots_toggle)
    for _ in range(600):
        if not sim.ball_count:
            break
        window.advance(1 / 15)
    assert not sim.ball_count
    assert len(sim.specks) == 32
    assert {(mark.x, mark.y) for mark in sim.specks} == initial_origins
    midpoint = sim.width / 2
    assert {mark.color for mark in sim.specks} == {RED, BLUE}
    assert all(mark.color == (RED if mark.x < midpoint else BLUE) for mark in sim.specks)
    paint = window.canvas.grab().toImage()
    for color in (RED, BLUE):
        mark = next(mark for mark in sim.specks if mark.color == color)
        assert paint.pixelColor(int(mark.x), int(mark.y)).getRgb()[:3] == color


def test_image_mode_and_hidden_sources_change_canvas_behavior(qtbot, window, tmp_path):
    window.settings_view.setCurrentText("All settings")
    import_image(qtbot, window, split_image(tmp_path))
    sim = window.simulation
    red = next(dot for dot in sim.dots if dot.color == RED)
    blue = next(dot for dot in sim.dots if dot.color == BLUE)

    # Drag a blue source onto the red half. A red-origin ball cannot use it until
    # ordinary gravity is selected.
    start = QPoint(round(blue.x), round(blue.y))
    target = QPoint(round(red.x), round(red.y) + 80)
    qtbot.mousePress(window.canvas, LEFT, pos=start)
    qtbot.mouseMove(window.canvas, target)
    qtbot.mouseRelease(window.canvas, LEFT, pos=target)
    assert window.canvas.grab().toImage().pixelColor(target).getRgb()[:3] == BLUE
    click_control(qtbot, window, window.show_dots_toggle)
    assert window.canvas.grab().toImage().pixelColor(target).value() == 0
    qtbot.mouseClick(window.canvas, Qt.MouseButton.RightButton, pos=target)
    assert blue in sim.dots  # Hidden sources cannot be erased from the canvas.

    qtbot.keyClick(window.canvas, Qt.Key.Key_B)
    qtbot.mouseClick(window.canvas, LEFT, pos=target)
    window.advance(1 / 120)
    assert sim.ball_count == 1
    click_control(qtbot, window, window.image_mode_toggle)
    window.advance(1 / 120)
    assert sim.ball_count == 0
    assert [(mark.x, mark.y, mark.color) for mark in sim.specks] == [(target.x(), target.y(), BLUE)]
    click_control(qtbot, window, window.image_mode_toggle)
    assert sim.image_mode
    assert len(sim.specks) == 1


def test_recolouring_an_image_source_preserves_existing_paint(qtbot, window, tmp_path):
    window.settings_view.setCurrentText("All settings")
    import_image(qtbot, window, split_image(tmp_path))
    sim = window.simulation
    blue = next(dot for dot in sim.dots if dot.color == BLUE)
    target = QPoint(round(blue.x), round(blue.y))

    click_control(qtbot, window, window.show_dots_toggle)
    qtbot.keyClick(window.canvas, Qt.Key.Key_B)
    qtbot.mouseClick(window.canvas, LEFT, pos=target)
    window.advance(1 / 120)
    assert [mark.color for mark in sim.specks] == [BLUE]

    click_control(qtbot, window, window.show_dots_toggle)
    qtbot.mouseClick(window.canvas, LEFT, pos=target)
    click_control(qtbot, window, "swatch_2")
    click_control(qtbot, window, window.show_dots_toggle)
    next_origin = target + QPoint(6, 0)
    qtbot.mouseClick(window.canvas, LEFT, pos=next_origin)
    window.advance(1 / 120)
    assert blue.color != BLUE
    paint = window.canvas.grab().toImage()
    assert paint.pixelColor(target).getRgb()[:3] == BLUE
    assert paint.pixelColor(next_origin).getRgb()[:3] == blue.color


def test_palette_applies_on_reload_and_reset_discards_image(qtbot, window, tmp_path):
    window.settings_view.setCurrentText("All settings")
    image = QImage(8, 8, QImage.Format.Format_RGB32)
    colors = {(index * 4, 0, 255) for index in range(64)}
    for index in range(64):
        image.setPixelColor(index % 8, index // 8, QColor(index * 4, 0, 255))
    path = tmp_path / "64-colours.png"
    assert image.save(str(path))

    import_image(qtbot, window, path)
    original_dots = window.simulation.dots.copy()
    window.settings_sidebar.ensureWidgetVisible(window.palette_count)
    window.palette_count.setFocus()
    qtbot.keyClick(window.palette_count, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClicks(window.palette_count, "64")
    qtbot.keyClick(window.palette_count, Qt.Key.Key_Return)
    assert window.simulation.dots == original_dots
    assert import_image(qtbot, window, path, QMessageBox.StandardButton.Yes) == [
        QMessageBox.Icon.Question
    ]
    assert {dot.color for dot in window.simulation.dots} == colors

    click_control(qtbot, window, window.show_dots_toggle)
    assert click_with_dialogs(
        qtbot, window, "reset_everything", answers=[QMessageBox.StandardButton.Yes]
    ) == [QMessageBox.Icon.Question]
    assert window.show_dots_toggle.isChecked()
    assert not window.image_mode_toggle.isEnabled()
    assert window.simulation.image_field is None
    assert not window.simulation.dots


def test_stream_favours_image_edges_while_manual_batches_cover_the_field(qtbot, window, tmp_path):
    window.settings_view.setCurrentText("All settings")
    import_image(qtbot, window, split_image(tmp_path, size=32))
    window.stream_rate.setValue(10000)
    window.batch_count.setValue(40000)
    click_control(qtbot, window, window.stream_toggle)
    click_control(qtbot, window, "spawn_random")
    window.spawn_for_frame(0.1)
    assert window.simulation.ball_count == 1000
    assert window.pending_spawns == 40000
    for _ in range(39):
        window.spawn_for_frame(0.1)
    streamed = window.simulation.origins.copy()
    assert len(streamed) == 40000

    click_control(qtbot, window, window.stream_toggle)
    for _ in range(40):
        window.spawn_for_frame(0)
    manual = window.simulation.origins[len(streamed) :]
    assert len(manual) == len(streamed)
    assert not window.pending_spawns

    sim = window.simulation
    midpoint = sim.width / 2
    edge_width = min(sim.width, sim.height) / 32

    def edge_fraction(origins):
        return np.mean(np.abs(origins[:, 0] - midpoint) < edge_width)

    assert edge_fraction(manual) == pytest.approx(2 / 32, abs=0.007)
    assert edge_fraction(streamed) > edge_fraction(manual) + 0.01
    assert edge_fraction(streamed) < 0.12


def test_cancelled_or_invalid_image_import_preserves_scene(qtbot, window, tmp_path):
    qtbot.mouseClick(window.canvas, LEFT, pos=QPoint(100, 100))
    qtbot.keyClick(window.canvas, Qt.Key.Key_B)
    qtbot.mouseClick(window.canvas, LEFT, pos=QPoint(220, 100))
    sim = window.simulation
    original_dots = [(dot.x, dot.y, dot.color) for dot in sim.dots]
    original_balls = sim.positions.copy()
    path = split_image(tmp_path)
    window.settings_view.setCurrentText("Paint with gravity")

    def assert_scene_preserved():
        assert sim.image_field is None
        assert [(dot.x, dot.y, dot.color) for dot in sim.dots] == original_dots
        np.testing.assert_array_equal(sim.positions, original_balls)

    assert not import_image(qtbot, window, None)
    assert_scene_preserved()
    assert import_image(qtbot, window, path, QMessageBox.StandardButton.No) == [
        QMessageBox.Icon.Question
    ]
    assert_scene_preserved()
    path.write_bytes(b"bad image")
    assert import_image(qtbot, window, path, QMessageBox.StandardButton.Ok) == [
        QMessageBox.Icon.Warning
    ]
    assert_scene_preserved()


def test_palette_allocation_failure_preserves_scene(qtbot, window, tmp_path, monkeypatch):
    qtbot.mouseClick(window.canvas, LEFT, pos=QPoint(100, 100))
    original = window.simulation.dots.copy()
    window.settings_view.setCurrentText("Paint with gravity")

    def out_of_memory(*args):
        raise MemoryError

    # The allocator is the fault boundary, and the file, scene, and dialog stay real.
    monkeypatch.setattr(ImageField, "dot_seeds", out_of_memory)
    assert import_image(
        qtbot,
        window,
        split_image(tmp_path),
        QMessageBox.StandardButton.Yes,
        QMessageBox.StandardButton.Ok,
    ) == [QMessageBox.Icon.Question, QMessageBox.Icon.Warning]
    assert window.simulation.image_field is None
    assert window.simulation.dots == original
