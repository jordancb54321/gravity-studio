"""Palette extraction and spatial colour classes, independent of the GUI."""

from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
from numpy.typing import NDArray


@dataclass
class ImageField:
    palette: NDArray[np.uint8]
    # Signed int32 covers all 2**24 RGB classes and the outside-image -1 sentinel.
    labels: NDArray[np.int32]
    contrast: NDArray[np.float64] | None = None

    @property
    def colors(self) -> list[tuple[int, int, int]]:
        return [(int(row[0]), int(row[1]), int(row[2])) for row in self.palette]

    @classmethod
    def from_pixels(cls, pixels: NDArray[np.uint8], colors: int = 8) -> "ImageField":
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("Image pixels must be an RGB uint8 array.")
        if not pixels.shape[0] or not pixels.shape[1]:
            raise ValueError("Image must contain at least one pixel.")
        if type(colors) is not int or colors < 2:
            raise ValueError("Choose at least 2 palette colours.")
        # Packed RGB keys sort in the same lexicographic order as RGB rows, but
        # avoid the expensive structured-row sort on every full-resolution pixel.
        flat = pixels.reshape(-1, 3)
        packed = flat[:, 0].astype(np.uint32) << 16
        packed |= flat[:, 1].astype(np.uint32) << 8
        packed |= flat[:, 2]
        keys, inverse, counts = np.unique(packed, return_inverse=True, return_counts=True)
        unique = np.column_stack((keys >> 16, (keys >> 8) & 255, keys & 255)).astype(np.uint8)
        if colors >= len(unique):
            palette = unique
            labels = inverse.astype(np.int32).reshape(pixels.shape[:2])
        else:
            palette = cls._median_cut(unique, counts, colors)
            # Repeated RGB values always classify identically; search each only once.
            groups = cls._nearest_groups(unique, palette)
            labels = groups[inverse].reshape(pixels.shape[:2])
        del packed, inverse, counts
        # Detect real RGB changes before palette reduction, including equal-brightness edges.
        contrast = np.zeros(pixels.shape[:2], dtype=np.float64)
        rows, columns = pixels.shape[:2]
        block_rows = max(1, 65536 // columns)
        for start in range(0, rows, block_rows):
            stop = min(rows, start + block_rows)
            # One halo row carries vertical edges across block boundaries.
            rgb = pixels[start : stop + 1].astype(np.float64) / 255.0
            current = rgb[: stop - start]
            horizontal = np.max(np.abs(current[:, 1:] - current[:, :-1]), axis=2)
            contrast[start:stop, 1:] = np.maximum(contrast[start:stop, 1:], horizontal)
            contrast[start:stop, :-1] = np.maximum(contrast[start:stop, :-1], horizontal)
            vertical = np.max(np.abs(rgb[1:] - rgb[:-1]), axis=2)
            end = start + len(vertical)
            contrast[start + 1 : end + 1] = np.maximum(contrast[start + 1 : end + 1], vertical)
            contrast[start:end] = np.maximum(contrast[start:end], vertical)
        return cls(palette, labels, contrast)

    @staticmethod
    def _nearest_groups(pixels: NDArray[np.uint8], palette: NDArray[np.uint8]) -> NDArray[np.int32]:
        """Exact squared RGB distance, with first-index ties and bounded temporaries."""
        groups = np.empty(len(pixels), dtype=np.int32)
        # Both dimensions are tiled: lifting the palette limit must not grow a
        # pixels-by-entire-palette temporary. Each distance matrix is at most 1 MiB.
        for start in range(0, len(pixels), 256):
            chunk = pixels[start : start + 256].astype(np.int32)
            best = np.full(len(chunk), 3 * 255**2 + 1, dtype=np.int32)
            chosen = np.zeros(len(chunk), dtype=np.int32)
            for offset in range(0, len(palette), 1024):
                colors = palette[offset : offset + 1024].astype(np.int32)
                distances = np.zeros((len(chunk), len(colors)), dtype=np.int32)
                for channel in range(3):
                    delta = chunk[:, None, channel] - colors[None, :, channel]
                    np.square(delta, out=delta)
                    distances += delta
                nearest = np.argmin(distances, axis=1)
                values = distances[np.arange(len(chunk)), nearest]
                closer = values < best  # Strict comparison preserves earlier ties across tiles.
                best[closer] = values[closer]
                chosen[closer] = offset + nearest[closer]
            groups[start : start + len(chunk)] = chosen
        return groups

    @staticmethod
    def _median_cut(
        unique: NDArray[np.uint8], counts: NDArray[np.int64], colors: int
    ) -> NDArray[np.uint8]:
        def entry(box: NDArray[np.int64], order: int) -> tuple[int, int, NDArray[np.int64]]:
            score = int(np.ptp(unique[box], axis=0).max()) * int(counts[box].sum())
            return -score, order, box

        # Cache split priorities instead of rescanning every live box each time.
        # Insertion order breaks ties exactly as the original stable list did.
        boxes = [entry(np.arange(len(unique)), 0)]
        order = 1
        while len(boxes) < min(colors, len(unique)):
            _, _, box = heappop(boxes)
            channel = int(np.argmax(np.ptp(unique[box], axis=0)))
            ordered = box[np.argsort(unique[box, channel], kind="stable")]
            middle = int(np.searchsorted(np.cumsum(counts[ordered]), counts[ordered].sum() / 2)) + 1
            middle = max(1, min(len(ordered) - 1, middle))
            heappush(boxes, entry(ordered[:middle], order))
            heappush(boxes, entry(ordered[middle:], order + 1))
            order += 2
        palette = np.array(
            [np.average(unique[box], axis=0, weights=counts[box]) for _, _, box in boxes]
        )
        return np.unique(np.rint(palette).astype(np.uint8), axis=0)

    def fit_rect(self, width: float, height: float) -> tuple[float, float, float, float]:
        rows, columns = self.labels.shape
        scale = min(width / columns, height / rows)
        fitted_width, fitted_height = columns * scale, rows * scale
        return (
            (width - fitted_width) / 2,
            (height - fitted_height) / 2,
            fitted_width,
            fitted_height,
        )

    def sample_groups(
        self, positions: NDArray[np.float64], width: float, height: float
    ) -> NDArray[np.int32]:
        left, top, fitted_width, fitted_height = self.fit_rect(width, height)
        rows, columns = self.labels.shape
        inside = (
            (positions[:, 0] >= left)
            & (positions[:, 0] <= left + fitted_width)
            & (positions[:, 1] >= top)
            & (positions[:, 1] <= top + fitted_height)
        )
        groups = np.full(len(positions), -1, dtype=np.int32)
        x = np.clip(
            ((positions[inside, 0] - left) / fitted_width * columns).astype(int), 0, columns - 1
        )
        y = np.clip(((positions[inside, 1] - top) / fitted_height * rows).astype(int), 0, rows - 1)
        groups[inside] = self.labels[y, x]
        return groups

    def dot_seeds(self, width: float, height: float) -> list[tuple[float, float, int]]:
        left, top, fitted_width, fitted_height = self.fit_rect(width, height)
        rows, columns = self.labels.shape
        flat = self.labels.ravel()
        counts = np.bincount(flat, minlength=len(self.palette))
        x_sums = np.zeros(len(counts), dtype=np.float64)
        y_sums = np.zeros(len(counts), dtype=np.float64)
        # Accumulate each pixel once, rather than scanning the image once per
        # palette class. Chunk coordinates so centroid scratch space stays small.
        for start in range(0, len(flat), 65536):
            groups = flat[start : start + 65536]
            indices = np.arange(start, start + len(groups))
            np.add.at(x_sums, groups, indices % columns)
            np.add.at(y_sums, groups, indices // columns)
        return [
            (
                left + (x_sums[group] / counts[group] + 0.5) / columns * fitted_width,
                top + (y_sums[group] / counts[group] + 0.5) / rows * fitted_height,
                int(group),
            )
            for group in np.flatnonzero(counts)
        ]
