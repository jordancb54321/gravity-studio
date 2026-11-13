"""Decode local images at original resolution; composite alpha onto black."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt  # type: ignore[import-untyped]
from PySide6.QtGui import QImage, QImageReader, QPainter  # type: ignore[import-untyped]


def load_image_pixels(path: Path) -> NDArray[np.uint8]:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Choose an image smaller than 32 MiB.")
    reader = QImageReader(str(path))
    reader.setAutoTransform(True)
    size = reader.size()
    if not size.isValid():
        raise ValueError(f"Cannot read image: {reader.errorString()}")

    image = reader.read()
    if image.isNull():
        raise ValueError(f"Cannot decode image: {reader.errorString()}")
    rgb = QImage(image.size(), QImage.Format.Format_RGB888)
    if rgb.isNull():
        raise MemoryError("Not enough memory to prepare the image's RGB pixels.")
    rgb.fill(Qt.GlobalColor.black)
    painter = QPainter(rgb)
    painter.drawImage(0, 0, image)
    painter.end()
    data = np.frombuffer(rgb.constBits(), dtype=np.uint8).reshape(rgb.height(), rgb.bytesPerLine())
    return data[:, : rgb.width() * 3].reshape(rgb.height(), rgb.width(), 3).copy()
