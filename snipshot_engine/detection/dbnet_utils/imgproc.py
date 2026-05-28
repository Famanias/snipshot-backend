"""Image processing utilities for detection (from CRAFT)."""

import numpy as np
import cv2


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape
    target_size = mag_ratio * square_size
    ratio = target_size / max(height, width)
    target_h, target_w = int(round(height * ratio)), int(round(width * ratio))
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    MULT = 256
    pad_h = (MULT - target_h % MULT) % MULT
    pad_w = (MULT - target_w % MULT) % MULT
    target_h32 = target_h + pad_h
    target_w32 = target_w + pad_w
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.uint8)
    resized[:target_h, :target_w, :] = proc

    size_heatmap = (target_w32 // 2, target_h32 // 2)
    return resized, ratio, size_heatmap, pad_w, pad_h
