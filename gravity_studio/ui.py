"""Native, single-threaded Gravity Studio editor and particle canvas."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

# Essentials ships Qt stubs but this distribution omits the PEP 561 marker.
from PySide6.QtCore import QEvent, QObject, QPointF, Qt, QTimer  # type: ignore[import-untyped]
from PySide6.QtGui import (  # type: ignore[import-untyped]
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QResizeEvent,
)
from PySide6.QtWidgets import (  # type: ignore[import-untyped]
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gravity_studio.image_field import ImageField
from gravity_studio.image_io import load_image_pixels
from gravity_studio.model import BALL_RADIUS, GravityDot, Simulation


class GravityCanvas(QWidget):
    """Canvas coordinates are logical pixels, shared with the simulation."""

    def __init__(self, window: MainWindow) -> None:
        super().__init__()
        self.editor = window
        self._dragging_dot: GravityDot | None = None
        self._drag_offset = QPointF()
        self.paint_cache = QImage()
        self._ball_layer = QImage()
        self._paint_generation = -1
        self._painted_count = 0
        self._paint_mark_size = 0.0
        self.setObjectName("gravity_canvas")
        self.setAccessibleName("Gravity canvas")
        self.setToolTip("Click to place a gravity dot or ball.")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(300, 300)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.empty_hint = QLabel(
            "Start with a gravity dot\n\n"
            "Click anywhere to place an attractor.\n"
            "Switch to Ball, or spawn a random batch.\n\n"
            "Drag dots to reshape the field. Right-click to erase.",
            self,
        )
        self.empty_hint.setObjectName("empty_hint")
        self.empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_hint.setWordWrap(True)
        self.empty_hint.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        hint_layout = QVBoxLayout(self)
        hint_layout.setContentsMargins(40, 40, 40, 40)
        hint_layout.addWidget(self.empty_hint)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.setFocus()
        point = QPointF(
            max(0.0, min(self.editor.simulation.width, event.position().x())),
            max(0.0, min(self.editor.simulation.height, event.position().y())),
        )
        if event.button() == Qt.MouseButton.RightButton:
            dot = self.dot_near(point)
            if dot is not None:
                self.editor.simulation.remove_dot(dot)
                if self.editor.selected_dot is dot:
                    self.editor.selected_dot = None
                self._dragging_dot = None
            else:
                self.editor.simulation.erase_ball_near(point.x(), point.y())
            self.editor.refresh()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            dot = self.dot_near(point)
            if dot is not None:
                self.editor.selected_dot = dot
                self._dragging_dot = dot
                self._drag_offset = QPointF(dot.x, dot.y) - point
                self.editor.apply_color(dot.color)
            elif self.editor.mode == "gravity":
                self.editor.selected_dot = self.editor.simulation.add_dot(
                    point.x(), point.y(), self.editor.default_color
                )
            else:
                self.editor.selected_dot = None
                self.editor.simulation.add_ball(point.x(), point.y())
            self.editor.refresh()

    def dot_near(self, point: QPointF) -> GravityDot | None:
        if not self.editor.show_dots_toggle.isChecked():
            return None
        candidates = [
            dot
            for dot in self.editor.simulation.dots
            if (dot.x - point.x()) ** 2 + (dot.y - point.y()) ** 2 <= (dot.radius + 6) ** 2
        ]
        return min(
            candidates,
            key=lambda dot: (dot.x - point.x()) ** 2 + (dot.y - point.y()) ** 2,
            default=None,
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging_dot is not None:
            point = event.position() + self._drag_offset
            self._dragging_dot.x = max(0.0, min(float(self.width()), point.x()))
            self._dragging_dot.y = max(0.0, min(float(self.height()), point.y()))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self.mouseMoveEvent(event)
        self._dragging_dot = None

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.editor.simulation.resize(max(6, self.width()), max(6, self.height()))
        super().resizeEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        self.update_paint_cache()
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        painter.drawImage(0, 0, self.paint_cache)
        self.draw_balls(painter)
        if not self.editor.show_dots_toggle.isChecked():
            painter.end()
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        for dot in self.editor.simulation.dots:
            painter.setBrush(QColor(*dot.color))
            painter.setPen(QPen(QColor("#949494"), 1) if max(dot.color) < 32 else Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(dot.x, dot.y), dot.radius, dot.radius)
            if dot is self.editor.selected_dot:
                painter.setPen(QPen(QColor("#ffffff"), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(QPointF(dot.x, dot.y), dot.radius + 5, dot.radius + 5)

    def draw_balls(self, painter: QPainter) -> None:
        """Rasterize the array in bulk, without one Python/Qt call per ball."""
        positions = self.editor.simulation.positions
        if not len(positions):
            return
        if self._ball_layer.size() != self.size():
            self._ball_layer = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        self._ball_layer.fill(Qt.GlobalColor.transparent)
        pixels = np.frombuffer(self._ball_layer.bits(), dtype=np.uint32).reshape(
            self.height(), self.width()
        )
        centers = np.rint(positions).astype(np.intp)
        # A radius-2.5 disk on the logical-pixel raster; work scales in NumPy,
        # while the number of Python iterations is independent of ball count.
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dx * dx + dy * dy > BALL_RADIUS * BALL_RADIUS:
                    continue
                x = centers[:, 0] + dx
                y = centers[:, 1] + dy
                visible = (x >= 0) & (x < self.width()) & (y >= 0) & (y < self.height())
                pixels[y[visible], x[visible]] = 0xFFFFFFFF
        painter.drawImage(0, 0, self._ball_layer)

    def update_paint_cache(self) -> None:
        """Append marks once; scene, size and mark-width changes rebuild the raster."""
        simulation = self.editor.simulation
        mark_size = self.editor.origin_mark_size.value()
        if (
            self.paint_cache.size() != self.size()
            or self._paint_generation != simulation.paint_generation
            or self._painted_count > len(simulation.specks)
            or self._paint_mark_size != mark_size
        ):
            self.paint_cache = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
            self.paint_cache.fill(Qt.GlobalColor.transparent)
            self._paint_generation = simulation.paint_generation
            self._paint_mark_size = mark_size
            self._painted_count = 0
        if self._painted_count == len(simulation.specks):
            return
        painter = QPainter(self.paint_cache)
        for index in range(self._painted_count, len(simulation.specks)):
            speck = simulation.specks[index]
            painter.setPen(QPen(QColor(*speck.color), mark_size))
            painter.drawPoint(QPointF(speck.x, speck.y))
        painter.end()
        self._painted_count = len(simulation.specks)


class CollapsibleSection(QWidget):
    """A keyboard-accessible disclosure that keeps its controls alive when folded."""

    def __init__(self, title: str) -> None:
        super().__init__()
        name = f"section_{title.lower().replace(' ', '_')}"
        self.setObjectName(name)
        self.setProperty("role", "settings_section")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.toggle = QToolButton()
        self.toggle.setObjectName(f"{name}_toggle")
        self.toggle.setProperty("role", "section")
        self.toggle.setText(title)
        self.toggle.setAccessibleName(f"Expand or collapse {title.lower()} settings")
        self.toggle.setToolTip(f"Expand or collapse {title.lower()} without changing settings.")
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        layout.addWidget(self.toggle)
        self.content = QWidget()
        self.content.setObjectName(f"{name}_content")
        self.content.setProperty("role", "settings_content")
        self.controls = QVBoxLayout(self.content)
        self.controls.setContentsMargins(0, 0, 0, 0)
        self.controls.setSpacing(8)
        layout.addWidget(self.content)
        self.toggle.toggled.connect(self.set_expanded)
        self.set_expanded(True)

    def set_expanded(self, expanded: bool) -> None:
        self.content.setVisible(expanded)
        self.toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self.toggle.setAccessibleDescription("Expanded" if expanded else "Collapsed")


class MainWindow(QMainWindow):
    def __init__(self, simulation: Simulation | None = None) -> None:
        super().__init__()
        self.simulation = simulation if simulation is not None else Simulation()
        self.mode = "gravity"
        self.selected_dot: GravityDot | None = None
        self.default_color = (242, 179, 61)
        self.image_name = ""
        self.pending_spawns = 0
        self._stream_fraction = 0.0
        self.paused = False
        self._accumulator = 0.0
        self._last_frame = perf_counter()
        self.canvas: GravityCanvas
        self.setWindowTitle("Gravity Studio")
        self.resize(1280, 820)
        self.setMinimumSize(900, 640)
        families = QFontDatabase.families()
        self.setFont(QFont("Inter" if "Inter" in families else "sans-serif", 10))
        stylesheet = Path(__file__).with_name("style.qss").read_text(encoding="utf-8")
        check_mark = Path(__file__).with_name("check.svg").as_posix()
        self.setStyleSheet(stylesheet.replace("CHECK_MARK_PATH", check_mark))
        root = QWidget()
        self.setCentralWidget(root)
        frame = QVBoxLayout(root)
        frame.setContentsMargins(16, 12, 16, 12)
        frame.setSpacing(12)
        header = QHBoxLayout()
        title = QLabel("Gravity Studio")
        title.setObjectName("app_title")
        header.addWidget(title)
        self.settings_toggle = QPushButton("Hide settings")
        self.settings_toggle.setObjectName("toggle_settings")
        self.settings_toggle.setAccessibleName("Show or hide settings panel")
        self.settings_toggle.setToolTip("Hide settings to enlarge the canvas, or show them again.")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setChecked(True)
        self.settings_toggle.toggled.connect(self.set_settings_visible)
        header.addWidget(self.settings_toggle)
        self.run_state = QLabel()
        self.run_state.setProperty("role", "muted")
        header.addWidget(self.run_state)
        header.addStretch()
        self.counts = QLabel()
        self.counts.setMinimumWidth(280)
        self.counts.setProperty("role", "muted")
        self.counts.setAlignment(Qt.AlignmentFlag.AlignRight)
        header.addWidget(self.counts)
        self.pause_button = QPushButton("Pause · Space")
        self.pause_button.setObjectName("pause")
        self.pause_button.setAccessibleName("Pause or resume simulation")
        self.pause_button.setToolTip(
            "Pause physics; placement and queued spawning remain available."
        )
        self.pause_button.clicked.connect(self.toggle_pause)
        header.addWidget(self.pause_button)
        frame.addLayout(header)
        # 16 + 270 + 12 + flexible canvas + 16 closes the horizontal frame.
        layout = QHBoxLayout()
        layout.setSpacing(12)
        frame.addLayout(layout, 1)
        sidebar = QWidget()
        sidebar.setObjectName("controls_content")
        categories = QVBoxLayout(sidebar)
        categories.setContentsMargins(16, 16, 16, 16)
        categories.setSpacing(16)
        controls = self.add_section(categories, "Place")
        self.tool_buttons: dict[str, QPushButton] = {}
        tool_group = QButtonGroup(self)
        tool_group.setExclusive(True)
        for mode, label in (("gravity", "Gravity dot · G"), ("ball", "Ball · B")):
            button = QPushButton(label)
            button.setObjectName(f"{mode}_tool")
            button.setAccessibleName(label)
            button.setToolTip(f"Click the canvas to place a {mode}.")
            button.setCheckable(True)
            tool_group.addButton(button)
            self.tool_buttons[mode] = button
            button.setChecked(mode == self.mode)
            button.clicked.connect(lambda checked=False, tool=mode: self.set_mode(tool))
            controls.addWidget(button)
        controls.addSpacing(8)
        self.show_dots_toggle = QCheckBox("Show gravity dots")
        self.show_dots_toggle.setObjectName("show_gravity_dots")
        self.show_dots_toggle.setAccessibleName("Show gravity dots")
        self.show_dots_toggle.setToolTip(
            "Hide dot markers without changing gravity or capture. "
            "Show them again to select, drag or erase them."
        )
        self.show_dots_toggle.setChecked(True)
        self.show_dots_toggle.toggled.connect(lambda checked: self.canvas.update())
        controls.addWidget(self.show_dots_toggle)
        self.color_label = QLabel("New dot · #f2b33d")
        self.color_label.setProperty("role", "muted")
        controls.addWidget(self.color_label)
        self.color_button = QPushButton("Choose colour…")
        self.color_button.setObjectName("choose_color")
        self.color_button.setAccessibleName("Choose gravity dot colour")
        self.color_button.setToolTip(
            "Edit the selected dot and new dots. Recolouring an image dot "
            "changes its whole palette class; old paint is unchanged."
        )
        self.color_button.clicked.connect(self.choose_color)
        controls.addWidget(self.color_button)
        swatches = QHBoxLayout()
        for index, color in enumerate(
            ((242, 179, 61), (86, 182, 247), (240, 112, 125), (117, 209, 165))
        ):
            swatch = QPushButton()
            swatch.setObjectName(f"swatch_{index}")
            swatch.setAccessibleName(f"Use colour {QColor(*color).name()}")
            swatch.setToolTip(f"Set dot colour to {QColor(*color).name()}")
            swatch.setFixedSize(30, 30)
            swatch.setStyleSheet(
                f"QPushButton {{background: {QColor(*color).name()}; "
                "border: 2px solid transparent;}"
                "QPushButton:focus {border-color: #ffffff;}"
            )
            swatch.clicked.connect(lambda checked=False, value=color: self.apply_color(value))
            swatches.addWidget(swatch)
        controls.addLayout(swatches)
        controls = self.add_section(categories, "Physics")
        self.drag_spin = QDoubleSpinBox()
        self.drag_spin.setObjectName("air_resistance")
        self.drag_spin.setAccessibleName("Air resistance per second")
        self.drag_spin.setToolTip("Exponential velocity damping per second. Zero is frictionless.")
        self.drag_spin.setRange(0.0, 5.0)
        self.drag_spin.setDecimals(2)
        self.drag_spin.setSingleStep(0.05)
        self.drag_spin.setSuffix(" /s")
        self.drag_spin.setValue(self.simulation.drag)
        drag_label = QLabel("Air resistance")
        drag_label.setBuddy(self.drag_spin)
        self.drag_spin.setFixedWidth(118)
        drag_row = QHBoxLayout()
        drag_row.addWidget(drag_label)
        drag_row.addWidget(self.drag_spin)
        controls.addLayout(drag_row)
        self.drag_slider = QSlider(Qt.Orientation.Horizontal)
        self.drag_slider.setObjectName("air_resistance_slider")
        self.drag_slider.setAccessibleName("Air resistance slider")
        self.drag_slider.setToolTip(self.drag_spin.toolTip())
        self.drag_slider.setRange(0, 500)
        self.drag_slider.setValue(round(self.simulation.drag * 100))
        self.drag_spin.valueChanged.connect(self.set_drag)
        self.drag_slider.valueChanged.connect(lambda value: self.drag_spin.setValue(value / 100))
        controls.addWidget(self.drag_slider)
        self.strength_spin = QDoubleSpinBox()
        self.strength_spin.setObjectName("gravity_strength")
        self.strength_spin.setAccessibleName("Global gravity strength")
        self.strength_spin.setToolTip("Attraction strength shared by every gravity dot.")
        self.strength_spin.setRange(0, 2000000)
        self.strength_spin.setDecimals(0)
        self.strength_spin.setSingleStep(10000)
        self.strength_spin.setValue(self.simulation.gravity_strength)
        strength_label = QLabel("Gravity strength")
        strength_label.setBuddy(self.strength_spin)
        self.strength_spin.setFixedWidth(118)
        self.strength_spin.setGroupSeparatorShown(True)
        strength_row = QHBoxLayout()
        strength_row.addWidget(strength_label)
        strength_row.addWidget(self.strength_spin)
        controls.addLayout(strength_row)
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setObjectName("gravity_strength_slider")
        self.strength_slider.setAccessibleName("Gravity strength slider")
        self.strength_slider.setToolTip(self.strength_spin.toolTip())
        self.strength_slider.setRange(0, 2000)
        self.strength_slider.setValue(round(self.simulation.gravity_strength / 1000))
        self.strength_spin.valueChanged.connect(self.set_strength)
        self.strength_slider.valueChanged.connect(
            lambda value: self.strength_spin.setValue(value * 1000)
        )
        controls.addWidget(self.strength_slider)
        controls = self.add_section(categories, "Painting")
        self.paint_toggle = QCheckBox("Paint spawn points")
        self.paint_toggle.setObjectName("paint_origins")
        self.paint_toggle.setAccessibleName("Paint captured spawn points")
        self.paint_toggle.setToolTip(
            "A captured ball leaves its origin painted in the capturing dot's colour."
        )
        self.paint_toggle.setChecked(self.simulation.paint_spawn_points)
        self.paint_toggle.toggled.connect(self.set_paint)
        controls.addWidget(self.paint_toggle)
        paint_note = QLabel(
            "Captured balls paint their origins in the dot’s colour. Marks never fade."
        )
        paint_note.setWordWrap(True)
        paint_note.setProperty("role", "muted")
        controls.addWidget(paint_note)
        self.origin_mark_size = QDoubleSpinBox()
        self.origin_mark_size.setObjectName("origin_mark_size")
        self.origin_mark_size.setAccessibleName("Origin paint mark size in pixels")
        self.origin_mark_size.setToolTip(
            "Size of coloured marks left at captured balls' origins. "
            "Use 1 px for finer image detail. Updates existing and future paint; "
            "does not change balls or gravity dots."
        )
        self.origin_mark_size.setRange(1.0, 100.0)
        self.origin_mark_size.setDecimals(1)
        self.origin_mark_size.setSingleStep(0.5)
        self.origin_mark_size.setSuffix(" px")
        self.origin_mark_size.setValue(2.5)
        self.origin_mark_size.setFixedWidth(118)
        self.origin_mark_size.valueChanged.connect(lambda value: self.canvas.update())
        mark_size_label = QLabel("Origin mark size")
        mark_size_label.setBuddy(self.origin_mark_size)
        mark_size_row = QHBoxLayout()
        mark_size_row.addWidget(mark_size_label)
        mark_size_row.addWidget(self.origin_mark_size)
        controls.addLayout(mark_size_row)
        controls = self.add_section(categories, "Image painting")
        self.palette_count = QSpinBox()
        self.palette_count.setObjectName("palette_colors")
        self.palette_count.setAccessibleName("Image palette colour count")
        self.palette_count.setToolTip(
            "Colour count for the next image load. "
            "Changing this does not alter an already loaded image. "
            "Large palettes take longer to load and simulate; "
            "the image's distinct colours determine how many can be created."
        )
        self.palette_count.setRange(2, 2_147_483_647)
        self.palette_count.setGroupSeparatorShown(True)
        self.palette_count.setValue(8)
        self.palette_count.setFixedWidth(118)
        palette_label = QLabel("Colours on load")
        palette_label.setBuddy(self.palette_count)
        palette_row = QHBoxLayout()
        palette_row.addWidget(palette_label)
        palette_row.addWidget(self.palette_count)
        controls.addLayout(palette_row)
        image_button = QPushButton("Load image…")
        image_button.setObjectName("load_image")
        image_button.setAccessibleName("Load image for colour-directed gravity")
        image_button.setToolTip(
            "Extract a palette and create gravity dots. "
            "Replacing a nonempty scene asks for confirmation."
        )
        image_button.clicked.connect(self.load_image)
        controls.addWidget(image_button)
        self.image_mode_toggle = QCheckBox("Image painting mode")
        self.image_mode_toggle.setObjectName("image_mode")
        self.image_mode_toggle.setAccessibleName("Colour-directed image painting mode")
        self.image_mode_toggle.setToolTip(
            "Load an image first. Each ball then targets only "
            "the palette class sampled at its spawn position."
        )
        self.image_mode_toggle.setEnabled(False)
        self.image_mode_toggle.toggled.connect(self.set_image_mode)
        controls.addWidget(self.image_mode_toggle)
        self.image_status = QLabel("No image loaded")
        self.image_status.setWordWrap(True)
        self.image_status.setProperty("role", "muted")
        controls.addWidget(self.image_status)
        controls = self.add_section(categories, "Random balls")
        self.stream_toggle = QCheckBox("Continuous stream")
        self.stream_toggle.setObjectName("continuous_stream")
        self.stream_toggle.setAccessibleName("Continuous random ball stream")
        self.stream_toggle.setToolTip("Spawn random balls continuously while the simulation runs.")
        self.stream_toggle.toggled.connect(lambda enabled: self.reset_stream_clock())
        controls.addWidget(self.stream_toggle)
        self.stream_rate = QSpinBox()
        self.stream_rate.setObjectName("stream_rate")
        self.stream_rate.setAccessibleName("Continuous stream balls per second")
        self.stream_rate.setToolTip(
            "Target balls per second while running. Pausing stops the stream. "
            "Missed spawns after stalls are skipped; at most 1,000 balls are added per frame."
        )
        self.stream_rate.setRange(1, 10000)
        self.stream_rate.setGroupSeparatorShown(True)
        self.stream_rate.setValue(100)
        self.stream_rate.setFixedWidth(118)
        self.stream_rate.valueChanged.connect(lambda value: self.reset_stream_clock())
        rate_label = QLabel("Balls per second")
        rate_label.setBuddy(self.stream_rate)
        rate_row = QHBoxLayout()
        rate_row.addWidget(rate_label)
        rate_row.addWidget(self.stream_rate)
        controls.addLayout(rate_row)
        controls.addSpacing(8)
        self.batch_count = QSpinBox()
        self.batch_count.setObjectName("batch_count")
        self.batch_count.setAccessibleName("Random batch count")
        self.batch_count.setToolTip(
            "Balls per batch, added up to 1,000 per frame. No scene count limit."
        )
        self.batch_count.setRange(1, 1000000)
        self.batch_count.setGroupSeparatorShown(True)
        self.batch_count.setValue(1000)
        batch_label = QLabel("Batch count")
        batch_label.setBuddy(self.batch_count)
        controls.addWidget(batch_label)
        controls.addWidget(self.batch_count)
        self.queue_label = QLabel()
        self.queue_label.setProperty("role", "muted")
        for name, label, action in (
            ("spawn_random", "Spawn random · R", self.queue_random),
            ("cancel_spawn", "Cancel queued balls", self.cancel_spawns),
        ):
            button = QPushButton(label)
            button.setObjectName(name)
            button.setAccessibleName(label)
            button.setToolTip(label)
            button.clicked.connect(action)
            controls.addWidget(button)
        controls.addWidget(self.queue_label)
        controls = self.add_section(categories, "Scene")
        for name, label, action, tooltip in (
            (
                "clear_balls",
                "Clear balls",
                self.clear_balls,
                "Stop streaming, remove balls and cancel queued spawns; keep dots and paint.",
            ),
            (
                "clear_paint",
                "Clear paint",
                self.clear_paint,
                "Remove all origin marks; keep dots and balls.",
            ),
            (
                "demo_scene",
                "Add demo scene",
                self.add_demo,
                "Add four coloured attractors and queue 3,000 balls with painting enabled.",
            ),
            (
                "reset_everything",
                "Reset everything…",
                self.request_reset,
                "Remove all scene content and restore defaults. Asks first if nonempty.",
            ),
        ):
            button = QPushButton(label)
            button.setObjectName(name)
            button.setAccessibleName(label)
            button.setToolTip(tooltip)
            button.clicked.connect(action)
            controls.addWidget(button)
        categories.addStretch()
        scroll = QScrollArea()
        scroll.setObjectName("controls_sidebar")
        scroll.setAccessibleName("Gravity Studio controls")
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(270)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(sidebar)
        self.settings_sidebar = scroll
        layout.addWidget(scroll)
        self.canvas = GravityCanvas(self)
        layout.addWidget(self.canvas, 1)
        self.tool_hint = QLabel()
        self.tool_hint.setProperty("role", "muted")
        frame.addWidget(self.tool_hint)
        self.error_message = QLabel()
        self.error_message.setObjectName("simulation_error")
        self.error_message.setAccessibleName("Simulation error")
        self.error_message.setProperty("role", "error")
        self.error_message.setWordWrap(True)
        self.error_message.hide()
        frame.addWidget(self.error_message)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.refresh()
        self.timer.start()

    def set_settings_visible(self, visible: bool) -> None:
        self.settings_sidebar.setVisible(visible)
        self.settings_toggle.setText("Hide settings" if visible else "Show settings")

    def load_image(self) -> None:
        timer_was_active = self.timer.isActive()
        self.timer.stop()
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Choose an image",
                "",
                "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*)",
            )
            if filename:
                self.load_image_path(Path(filename))
        finally:
            self._last_frame = perf_counter()
            self._accumulator = 0.0
            if timer_was_active:
                self.timer.start()

    def load_image_path(self, path: Path) -> bool:
        try:
            pixels = load_image_pixels(path)
            field = ImageField.from_pixels(pixels, self.palette_count.value())
        except (OSError, ValueError, MemoryError) as error:
            QMessageBox.warning(self, "Could not load image", str(error) or "Not enough memory.")
            return False
        if (
            self.simulation.dots
            or self.simulation.ball_count
            or self.simulation.specks
            or self.pending_spawns
            or self.stream_toggle.isChecked()
        ):
            answer = QMessageBox.question(
                self,
                "Replace the scene?",
                "Loading an image replaces all dots, balls and paint. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        try:
            self.simulation.set_image(field)
        except (ValueError, MemoryError) as error:
            QMessageBox.warning(self, "Could not load image", str(error) or "Not enough memory.")
            return False
        self.selected_dot = None
        self.canvas._dragging_dot = None
        self.pending_spawns = 0
        self.stream_toggle.setChecked(False)
        self.image_name = path.name
        self.image_mode_toggle.setEnabled(True)
        self.image_mode_toggle.setChecked(True)
        self.paint_toggle.setChecked(True)
        self.refresh()
        return True

    def set_image_mode(self, enabled: bool) -> None:
        self.simulation.image_mode = enabled and self.simulation.image_field is not None
        if self.simulation.image_mode:
            self.paint_toggle.setChecked(True)
        self.refresh()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.tool_buttons[mode].setChecked(True)
        self.refresh()

    @staticmethod
    def add_section(layout: QVBoxLayout, title: str) -> QVBoxLayout:
        section = CollapsibleSection(title)
        layout.addWidget(section)
        return section.controls

    def add_demo(self) -> None:
        colors = ((242, 179, 61), (86, 182, 247), (240, 112, 125), (117, 209, 165))
        points = ((0.33, 0.35), (0.67, 0.35), (0.33, 0.65), (0.67, 0.65))
        for (x, y), color in zip(points, colors, strict=True):
            self.simulation.add_dot(self.simulation.width * x, self.simulation.height * y, color)
        self.paint_toggle.setChecked(True)
        self.pending_spawns += 3000
        self.refresh()

    def reset_stream_clock(self) -> None:
        """Do not retroactively spawn when enabling the stream or changing its rate."""
        self._stream_fraction = 0.0
        self._last_frame = perf_counter()
        self.refresh()

    def spawn_for_frame(self, elapsed_seconds: float) -> None:
        stream_count = 0
        if self.stream_toggle.isChecked() and not self.paused:
            credit = self._stream_fraction + min(max(0.0, elapsed_seconds), 0.25) * (
                self.stream_rate.value()
            )
            due = int(credit + 1e-9)
            self._stream_fraction = max(0.0, credit - due)
            # Keep the fractional ball, but never accumulate a stalled stream backlog.
            stream_count = min(1000, due)
        batch_count = min(1000 - stream_count, self.pending_spawns)
        if stream_count and self.simulation.image_mode:
            self.simulation.spawn_random(stream_count, favor_contrast=True)
            stream_count = 0
        if stream_count + batch_count:
            self.simulation.spawn_random(stream_count + batch_count)
            self.pending_spawns -= batch_count

    def queue_random(self) -> None:
        self.pending_spawns += self.batch_count.value()
        self.refresh()

    def cancel_spawns(self) -> None:
        self.pending_spawns = 0
        self.refresh()

    def clear_balls(self) -> None:
        self.stream_toggle.setChecked(False)
        self._stream_fraction = 0.0
        self.pending_spawns = 0
        self.simulation.clear_balls()
        self.refresh()

    def clear_paint(self) -> None:
        self.simulation.clear_paint()
        self.refresh()

    def request_reset(self) -> None:
        nonempty = (
            self.simulation.ball_count
            or self.simulation.dots
            or self.simulation.specks
            or self.pending_spawns
            or self.stream_toggle.isChecked()
        )
        if nonempty:
            timer_was_active = self.timer.isActive()
            self.timer.stop()
            try:
                answer = QMessageBox.question(
                    self,
                    "Reset everything?",
                    "Remove all dots, balls and paint, cancel queued balls, and restore defaults?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
            finally:
                self._last_frame = perf_counter()
                self._accumulator = 0.0
                if timer_was_active:
                    self.timer.start()
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.stream_toggle.setChecked(False)
        self.stream_rate.setValue(100)
        self._stream_fraction = 0.0
        self.simulation.reset()
        self.image_name = ""
        self.image_mode_toggle.setChecked(False)
        self.image_mode_toggle.setEnabled(False)
        self.palette_count.setValue(8)
        self.show_dots_toggle.setChecked(True)
        self.pending_spawns = 0
        self.selected_dot = None
        self.canvas._dragging_dot = None
        self.default_color = (242, 179, 61)
        self.drag_spin.setValue(0.3)
        self.strength_spin.setValue(180000.0)
        self.paint_toggle.setChecked(False)
        self.origin_mark_size.setValue(2.5)
        self.batch_count.setValue(1000)
        self.set_mode("gravity")
        if self.paused:
            self.toggle_pause()
        self._accumulator = 0.0
        self._last_frame = perf_counter()
        self.refresh()

    def advance(self, elapsed_seconds: float) -> None:
        """Run one UI frame; queued allocations are bounded independently of time."""
        self.spawn_for_frame(elapsed_seconds)
        if self.paused:
            self._accumulator = 0.0
        else:
            step_seconds = 1 / 120
            self._accumulator = min(self._accumulator + max(0.0, elapsed_seconds), 8 * step_seconds)
            for _ in range(8):
                if self._accumulator + 1e-12 < step_seconds:
                    break
                self.simulation.step(step_seconds)
                self._accumulator = max(0.0, self._accumulator - step_seconds)
        self.refresh()

    def tick(self) -> None:
        now = perf_counter()
        elapsed = now - self._last_frame
        self._last_frame = now
        try:
            self.advance(elapsed)
        except (MemoryError, ValueError, FloatingPointError) as error:
            self.stream_toggle.setChecked(False)
            self.pending_spawns = 0
            self.paused = True
            self._accumulator = 0.0
            self.pause_button.setText("Resume · Space")
            detail = "Not enough memory." if isinstance(error, MemoryError) else str(error)
            self.error_message.setText(
                f"Simulation paused: {detail} Clear balls or reduce the batch size."
            )
            self.error_message.show()
            self.refresh()

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        if not self.paused:
            self.error_message.clear()
            self.error_message.hide()
        self._accumulator = 0.0
        self._last_frame = perf_counter()
        self.pause_button.setText("Resume · Space" if self.paused else "Pause · Space")
        self.refresh()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            not isinstance(event, QKeyEvent)
            or event.type() != QEvent.Type.KeyPress
            or not isinstance(watched, QWidget)
            or watched.window() is not self
            or isinstance(watched, (QAbstractSpinBox, QLineEdit))
            or isinstance(QApplication.focusWidget(), (QAbstractSpinBox, QLineEdit))
            or (
                event.key() == Qt.Key.Key_Space
                and (isinstance(watched, QToolButton) or watched is self.settings_toggle)
            )
            or event.modifiers() != Qt.KeyboardModifier.NoModifier
        ):
            return super().eventFilter(watched, event)
        actions = {
            Qt.Key.Key_G: lambda: self.set_mode("gravity"),
            Qt.Key.Key_B: lambda: self.set_mode("ball"),
            Qt.Key.Key_Space: self.toggle_pause,
            Qt.Key.Key_R: self.queue_random,
        }
        action = actions.get(event.key())
        if action is not None:
            if not event.isAutoRepeat():
                action()
            return True
        return super().eventFilter(watched, event)

    def set_drag(self, value: float) -> None:
        self.simulation.drag = value
        self.drag_slider.blockSignals(True)
        self.drag_slider.setValue(round(value * 100))
        self.drag_slider.blockSignals(False)

    def set_strength(self, value: float) -> None:
        self.simulation.gravity_strength = value
        self.strength_slider.blockSignals(True)
        self.strength_slider.setValue(round(value / 1000))
        self.strength_slider.blockSignals(False)

    def set_paint(self, enabled: bool) -> None:
        self.simulation.paint_spawn_points = enabled

    def choose_color(self) -> None:
        color = QColorDialog.getColor(QColor(*self.default_color), self, "Gravity dot colour")
        if color.isValid():
            self.apply_color((color.red(), color.green(), color.blue()))

    def apply_color(self, color: tuple[int, int, int]) -> None:
        self.default_color = color
        if self.selected_dot is not None:
            self.simulation.recolor_dot(self.selected_dot, color)
        self.refresh()

    def refresh(self) -> None:
        self.run_state.setText("Paused" if self.paused else "Running")
        field = self.simulation.image_field
        self.image_status.setText(
            f"{self.image_name}\n{len(field.colors)} palette colours · "
            f"{'image mode' if self.simulation.image_mode else 'sandbox mode'}"
            if field is not None
            else "No image loaded"
        )
        self.color_label.setText(
            f"{'Selected' if self.selected_dot is not None else 'New'} dot · "
            f"{QColor(*self.default_color).name()}"
        )
        self.canvas.empty_hint.setVisible(
            not (self.simulation.dots or self.simulation.ball_count or self.simulation.specks)
        )
        tool = "Gravity dot" if self.mode == "gravity" else "Ball"
        self.tool_hint.setText(
            f"{tool} tool · Click to place   /   Drag dots to move   /   Right-click to erase"
        )
        stream_status = (
            f"Stream on · {self.stream_rate.value():,} balls/s"
            if self.stream_toggle.isChecked()
            else "Stream off"
        )
        self.queue_label.setText(f"{self.pending_spawns:,} balls queued\n{stream_status}")
        cancel = self.findChild(QPushButton, "cancel_spawn")
        if cancel is not None:
            cancel.setEnabled(self.pending_spawns > 0)
        self.counts.setText(
            f"{self.simulation.ball_count:,} balls · {len(self.simulation.dots):,} gravity dots\n"
            f"{self.simulation.captured_count:,} captured · {len(self.simulation.specks):,} marks"
        )
        self.canvas.update()


def main(argv: list[str] | None = None) -> int:
    """Launch the native application, with bounded smoke/capture options for QA."""
    parser = argparse.ArgumentParser(
        description="Gravity Studio — native gravity and paint sandbox"
    )
    parser.add_argument(
        "--smoke-seconds", type=float, help="Exit after this many seconds (0–3600)."
    )
    parser.add_argument("--screenshot", type=Path, help="Save a PNG of the native window.")
    args = parser.parse_args(argv)
    if args.smoke_seconds is not None and (
        not math.isfinite(args.smoke_seconds) or not 0 < args.smoke_seconds <= 3600
    ):
        parser.error("--smoke-seconds must be finite, greater than zero, and at most 3600")
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Gravity Studio")
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()

    def capture() -> bool:
        if args.screenshot is not None and not window.grab().save(str(args.screenshot), "PNG"):
            print(f"Could not save screenshot: {args.screenshot}", file=sys.stderr)
            app.exit(1)
            return False
        return True

    def finish_smoke() -> None:
        if capture():
            app.quit()

    if args.smoke_seconds is not None:
        QTimer.singleShot(max(1, round(args.smoke_seconds * 1000)), finish_smoke)
    elif args.screenshot is not None:
        QTimer.singleShot(250, capture)
    return app.exec()
