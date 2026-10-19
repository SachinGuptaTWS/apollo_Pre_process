from __future__ import annotations

import cv2
import numpy as np

_ROTATE = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    height, current = image.shape[:2]
    if current == width or current == 0:
        return image
    new_h = max(1, int(round(height * (width / current))))
    interp = cv2.INTER_AREA if width < current else cv2.INTER_CUBIC
    return cv2.resize(image, (width, new_h), interpolation=interp)


def rotate_rgb(image: np.ndarray, degrees: int) -> np.ndarray:
    degrees = int(degrees) % 360
    if degrees == 0:
        return image
    return cv2.rotate(image, _ROTATE[degrees])
