"""Bubble detection — decide whether a text region sits in a normal speech bubble."""

import cv2
import numpy as np


def check_color(image: np.ndarray) -> bool:
    """Return True if the image contains non-greyscale colour pixels."""
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])[..., np.newaxis]
    dist = np.sum((image.astype(np.float32) - gray) ** 2, axis=-1)
    return int(np.sum(dist > 100)) > 10


def is_ignore(region_img: np.ndarray, ignore_bubble: int = 0) -> bool:
    """Return True if the region should be skipped (not a normal bubble).

    *ignore_bubble* range 1-50.  Recommended value: 10.
    """
    if ignore_bubble < 1 or ignore_bubble > 50:
        return False

    gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if region_img.ndim == 3 else region_img
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape[:2]

    total = 0
    black = 0

    for sl in [
        binary[0:2, 0:w],
        binary[h - 2 : h, 0:w],
        binary[2 : h - 2, 0:2],
        binary[2 : h - 2, w - 2 : w],
    ]:
        black += int(np.sum(sl == 0))
        total += sl.size

    if total == 0:
        return False

    ratio = round(black / total, 6) * 100
    if ignore_bubble <= ratio <= (100 - ignore_bubble):
        return True
    if check_color(region_img):
        return True
    return False
