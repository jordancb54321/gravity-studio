"""Vectorized point-attractor physics with no dependency on Qt."""

from dataclasses import dataclass
from math import ceil, exp, isfinite, sqrt

import numpy as np
from numpy.typing import NDArray

from gravity_studio.image_field import ImageField

Color = tuple[int, int, int]
FloatArray = NDArray[np.float64]
BALL_RADIUS = 2.5


@dataclass(eq=False)
class GravityDot:
    x: float
    y: float
    color: Color
    radius: float = 12.0
    palette_index: int | None = None


@dataclass(frozen=True)
class Speck:
    x: float
    y: float
    color: Color


class Simulation:
    """Fixed attractors pull independent balls; balls do not attract each other."""

    def __init__(self, width: float = 1000, height: float = 700, seed: int | None = None) -> None:
        self._validate_bounds(width, height)
        self.width = width
        self.height = height
        self.gravity_strength = 180_000.0
        self.drag = 0.3
        self.paint_spawn_points = False
        self.image_field: ImageField | None = None
        self.image_mode = False
        self._image_canvas_size = (width, height)
        self.dots: list[GravityDot] = []
        self.specks: list[Speck] = []
        self.paint_generation = 0
        self.captured_count = 0
        self._rng = np.random.default_rng(seed)
        # Each row: current x/y, velocity x/y, immutable spawn x/y, palette class (-1 = sandbox).
        self._balls: FloatArray = np.empty((0, 7), dtype=np.float64)
        self._count = 0

    @property
    def positions(self) -> FloatArray:
        return self._balls[: self._count, :2]

    @property
    def velocities(self) -> FloatArray:
        return self._balls[: self._count, 2:4]

    @property
    def origins(self) -> FloatArray:
        return self._balls[: self._count, 4:6]

    @property
    def ball_count(self) -> int:
        return self._count

    @staticmethod
    def _validate_bounds(width: float, height: float) -> None:
        if not all(isfinite(value) and 6 <= value <= 1_000_000 for value in (width, height)):
            raise ValueError("Canvas dimensions must be finite and between 6 and 1,000,000 pixels.")

    def _validate_position(self, x: float, y: float) -> None:
        if not (isfinite(x) and isfinite(y) and 0 <= x <= self.width and 0 <= y <= self.height):
            raise ValueError("Positions must be finite and inside the canvas.")

    @property
    def target_groups(self) -> NDArray[np.int32]:
        return self._balls[: self._count, 6].astype(np.int32)

    def _assign_groups(self, positions: FloatArray) -> NDArray[np.int32]:
        if self.image_field is None:
            return np.full(len(positions), -1, dtype=np.int32)
        return self.image_field.sample_groups(positions, *self._image_canvas_size)

    def set_image(self, field: ImageField) -> None:
        """Replace the scene only after a complete, valid field has been prepared."""
        colors = field.colors
        dots = [
            GravityDot(x, y, colors[group], 12.0, group)
            for x, y, group in field.dot_seeds(self.width, self.height)
        ]
        self.reset()
        self.image_field = field
        self._image_canvas_size = (self.width, self.height)
        self.image_mode = True
        self.paint_spawn_points = True
        self.dots.extend(dots)

    def _eligible(self, dot: GravityDot) -> NDArray[np.bool_]:
        groups = self._balls[: self._count, 6]
        if not self.image_mode:
            return np.ones(self._count, dtype=np.bool_)
        return (groups < 0) | (groups == dot.palette_index)

    def _reserve(self, additional: int) -> None:
        required = self._count + additional
        if required > len(self._balls):
            capacity = max(required, 256, len(self._balls) * 2)
            grown = np.empty((capacity, 7), dtype=np.float64)
            grown[: self._count] = self._balls[: self._count]
            self._balls = grown

    def add_dot(self, x: float, y: float, color: Color, radius: float = 12) -> GravityDot:
        self._validate_position(x, y)
        if len(color) != 3 or any(
            type(channel) is not int or not 0 <= channel <= 255 for channel in color
        ):
            raise ValueError("Colours must contain three integer RGB channels from 0 to 255.")
        if not isfinite(radius) or not 0 < radius <= 1_000_000:
            raise ValueError("Dot radius must be finite, positive, and at most 1,000,000 pixels.")
        group = None
        if self.image_field is not None and color in self.image_field.colors:
            group = self.image_field.colors.index(color)
        dot = GravityDot(x, y, color, radius, group)
        self.dots.append(dot)
        return dot

    def add_ball(self, x: float, y: float, vx: float = 0, vy: float = 0) -> None:
        self._validate_position(x, y)
        if not all(isfinite(value) and abs(value) <= 1_000_000 for value in (vx, vy)):
            raise ValueError(
                "Velocity must be finite and at most 1,000,000 pixels/second per axis."
            )
        self._reserve(1)
        inset_x = min(self.width - BALL_RADIUS, max(BALL_RADIUS, x))
        inset_y = min(self.height - BALL_RADIUS, max(BALL_RADIUS, y))
        group = int(self._assign_groups(np.array([[x, y]], dtype=np.float64))[0])
        self._balls[self._count] = (inset_x, inset_y, vx, vy, x, y, group)
        self._count += 1

    def step(self, dt_seconds: float) -> None:
        if not isfinite(dt_seconds) or not 0 <= dt_seconds <= 0.05:
            raise ValueError("Physics steps must be finite and between 0 and 0.05 seconds.")
        if not all(
            isfinite(value) and 0 <= value <= 1_000_000_000
            for value in (self.drag, self.gravity_strength)
        ):
            raise ValueError("Air resistance and gravity must be finite and between 0 and 1e9.")
        if dt_seconds == 0 or self._count == 0:
            return
        acceleration = np.zeros_like(self.positions)
        for dot in self.dots:
            delta = np.array((dot.x, dot.y)) - self.positions
            softened_squared = np.einsum("ij,ij->i", delta, delta) + max(dot.radius, 0.5) ** 2
            acceleration += (
                delta
                * (self.gravity_strength / softened_squared**1.5)[:, None]
                * (self._eligible(dot)[:, None])
            )
        self.velocities[:] += acceleration * dt_seconds
        self.velocities[:] *= exp(-self.drag * dt_seconds)
        # Bound numerical path lengths, not entity counts; ordinary motion is far below this.
        np.clip(self.velocities, -1_000_000, 1_000_000, out=self.velocities)
        previous = self.positions.copy()
        self.positions[:] += self.velocities * dt_seconds
        self._capture(previous, dt_seconds)
        self._reflect_walls()

    def _capture(self, previous: FloatArray, dt_seconds: float) -> None:
        """Use swept circles, splitting the few wall-crossing trajectories exactly."""
        if not self.dots:
            return
        motion = self.positions - previous
        length_squared = np.einsum("ij,ij->i", motion, motion)
        closest = np.full(self._count, np.inf)
        owners = np.full(self._count, -1, dtype=np.int64)
        for index, dot in enumerate(self.dots):
            offset = previous - (dot.x, dot.y)
            b = 2 * np.einsum("ij,ij->i", offset, motion)
            c = np.einsum("ij,ij->i", offset, offset) - (dot.radius + BALL_RADIUS) ** 2
            discriminant = b * b - 4 * length_squared * c
            denominator = 2 * np.maximum(length_squared, np.finfo(float).tiny)
            t = (-b - np.sqrt(np.maximum(discriminant, 0))) / denominator
            t[c <= 0] = 0
            crossed = (c <= 0) | ((length_squared > 0) & (discriminant >= 0) & (t >= 0) & (t <= 1))
            nearer = crossed & self._eligible(dot) & (t < closest)
            closest[nearer] = t[nearer]
            owners[nearer] = index
        crossed_wall = np.any(
            (self.positions < BALL_RADIUS)
            | (self.positions > (self.width - BALL_RADIUS, self.height - BALL_RADIUS)),
            axis=1,
        )
        for i in np.flatnonzero(crossed_wall):
            owners[i] = self._bounced_owner(
                previous[i], self.velocities[i], dt_seconds, int(self._balls[i, 6])
            )
        captured = owners >= 0
        captured_indices = np.flatnonzero(captured)
        if not len(captured_indices):
            return
        if self.paint_spawn_points:
            self.specks.extend(
                Speck(
                    float(self.origins[i, 0]),
                    float(self.origins[i, 1]),
                    self.dots[int(owners[i])].color,
                )
                for i in captured_indices
            )
        self.captured_count += len(captured_indices)
        survivors = self._balls[: self._count][~captured]
        self._count = len(survivors)
        self._balls[: self._count] = survivors

    def _segment_owner(self, x: float, y: float, dx: float, dy: float, group: int = -1) -> int:
        length_squared = dx * dx + dy * dy
        nearest = float("inf")
        owner = -1
        for index, dot in enumerate(self.dots):
            if self.image_mode and group >= 0 and dot.palette_index != group:
                continue
            ox, oy = x - dot.x, y - dot.y
            c = ox * ox + oy * oy - (dot.radius + BALL_RADIUS) ** 2
            if c <= 0:
                t = 0.0
            elif length_squared == 0:
                continue
            else:
                b = 2 * (ox * dx + oy * dy)
                discriminant = b * b - 4 * length_squared * c
                if discriminant < 0:
                    continue
                t = (-b - sqrt(discriminant)) / (2 * length_squared)
            if 0 <= t <= 1 and t < nearest:
                nearest, owner = t, index
        return owner

    def _bounced_owner(
        self, start: FloatArray, velocity: FloatArray, dt: float, group: int = -1
    ) -> int:
        x, y = float(start[0]), float(start[1])
        vx, vy = float(velocity[0]), float(velocity[1])
        remaining = dt
        spans = (self.width - 2 * BALL_RADIUS, self.height - 2 * BALL_RADIUS)
        max_segments = ceil(abs(vx) * dt / spans[0] + abs(vy) * dt / spans[1]) + 3
        for _ in range(max_segments):
            if remaining <= 0:
                break
            edge_x = self.width - BALL_RADIUS if vx > 0 else BALL_RADIUS
            edge_y = self.height - BALL_RADIUS if vy > 0 else BALL_RADIUS
            tx = max(0.0, (edge_x - x) / vx) if vx else float("inf")
            ty = max(0.0, (edge_y - y) / vy) if vy else float("inf")
            duration = min(remaining, tx, ty)
            dx, dy = vx * duration, vy * duration
            owner = self._segment_owner(x, y, dx, dy, group)
            if owner >= 0:
                return owner
            x += dx
            y += dy
            remaining -= duration
            if tx <= duration:
                x = edge_x
                vx = -vx
            if ty <= duration:
                y = edge_y
                vy = -vy
        return -1

    def spawn_random(self, count: int, *, favor_contrast: bool = False) -> None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Ball count must be a nonnegative integer.")
        low = (BALL_RADIUS, BALL_RADIUS)
        high = (self.width - BALL_RADIUS, self.height - BALL_RADIUS)
        if self.image_mode and self.image_field is not None:
            left, top, width, height = self.image_field.fit_rect(*self._image_canvas_size)
            low = (max(low[0], left), max(low[1], top))
            high = (min(high[0], left + width), min(high[1], top + height))
            if high[0] < low[0] or high[1] < low[1]:
                raise ValueError("The image is outside the resized canvas; enlarge or reload it.")
        positions = self._rng.uniform(low, high, size=(count, 2))
        if (
            favor_contrast
            and self.image_mode
            and self.image_field is not None
            and self.image_field.contrast is not None
        ):
            field = self.image_field
            left, top, width, height = field.fit_rect(*self._image_canvas_size)
            rows, columns = field.labels.shape
            alternatives = self._rng.uniform(low, high, size=(count, 2))
            candidates = np.stack((positions, alternatives), axis=1)
            x = np.clip(
                ((candidates[:, :, 0] - left) / width * columns).astype(int), 0, columns - 1
            )
            y = np.clip(((candidates[:, :, 1] - top) / height * rows).astype(int), 0, rows - 1)
            # Choose between two uniform candidates: a gentle bias with fixed work per ball.
            # Absolute contrast avoids amplifying tiny noise; flat regions keep full coverage.
            weights = 1.0 + 0.75 * self.image_field.contrast[y, x]
            use_alternative = self._rng.random(count) < weights[:, 1] / weights.sum(axis=1)
            positions[use_alternative] = alternatives[use_alternative]
        groups = self._assign_groups(positions)
        self._reserve(count)
        target = self._balls[self._count : self._count + count]
        target[:, :2] = positions
        target[:, 2:4] = 0
        target[:, 4:6] = positions
        target[:, 6] = groups
        self._count += count

    def _reflect_walls(self) -> None:
        for axis, size in enumerate((self.width, self.height)):
            span = size - 2 * BALL_RADIUS
            phase = (self.positions[:, axis] - BALL_RADIUS) % (2 * span)
            reflected = phase > span
            self.positions[:, axis] = BALL_RADIUS + np.where(reflected, 2 * span - phase, phase)
            self.velocities[:, axis] *= np.where(reflected, -1, 1)

    def resize(self, width: float, height: float) -> None:
        self._validate_bounds(width, height)
        self.width = width
        self.height = height
        np.clip(
            self.positions,
            BALL_RADIUS,
            (width - BALL_RADIUS, height - BALL_RADIUS),
            out=self.positions,
        )
        for dot in self.dots:
            dot.x = min(width, max(0, dot.x))
            dot.y = min(height, max(0, dot.y))

    def recolor_dot(self, dot: GravityDot, color: Color) -> None:
        if len(color) != 3 or any(
            type(value) is not int or not 0 <= value <= 255 for value in color
        ):
            raise ValueError("Colours require three integer RGB channels from 0 to 255.")
        if dot.palette_index is not None and self.image_field is not None:
            group = dot.palette_index
            self.image_field.palette[group] = color
            for source in self.dots:
                if source.palette_index == group:
                    source.color = color
        else:
            dot.color = color

    def remove_dot(self, dot: GravityDot) -> None:
        if dot in self.dots:
            self.dots.remove(dot)

    def erase_ball_near(self, x: float, y: float, radius: float = 12) -> bool:
        if not self._count:
            return False
        delta = self.positions - (x, y)
        distances = np.einsum("ij,ij->i", delta, delta)
        index = int(np.argmin(distances))
        if distances[index] > radius**2:
            return False
        self._balls[index] = self._balls[self._count - 1]
        self._count -= 1
        return True

    def clear_balls(self) -> None:
        self._balls = np.empty((0, 7), dtype=np.float64)
        self._count = 0

    def clear_paint(self) -> None:
        self.specks.clear()
        self.paint_generation += 1

    def reset(self) -> None:
        self.clear_balls()
        self.clear_paint()
        self.dots.clear()
        self.captured_count = 0
        self.image_field = None
        self.image_mode = False
