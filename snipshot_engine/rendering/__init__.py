# Rendering dispatch — places translated text back onto the image.

import cv2
import numpy as np
from typing import List

from . import text_render
from ..utils import TextBlock, color_difference, get_logger, rotate_polygons

logger = get_logger("render")


def _fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg


def _render_region(img, region: TextBlock, dst_points, hyphenate, line_spacing, disable_font_border):
    fg, bg = region.get_font_colors()
    fg, bg = _fg_bg_compare(fg, bg)
    if disable_font_border:
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)

    forced_dir = region._direction if hasattr(region, "_direction") else region.direction
    if forced_dir != "auto":
        render_h = forced_dir in ("horizontal", "h")
    else:
        render_h = region.horizontal

    if render_h:
        temp_box = text_render.put_text_horizontal(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_h[0]), round(norm_v[0]),
            region.alignment,
            region.direction == 'hl',
            fg, bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size,
            region.get_translation_for_rendering(),
            round(norm_v[0]),
            region.alignment,
            fg, bg,
            line_spacing,
        )

    if temp_box is None:
        return img

    h, w, _ = temp_box.shape
    r_temp = w / h
    r_orig = np.mean(norm_h / norm_v)

    # Pad temp_box to match original region aspect ratio
    if render_h:
        if r_temp > r_orig:
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0
            if h_ext >= 0:
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
                box[h_ext:h_ext + h, :w] = temp_box
            else:
                box = temp_box.copy()
        else:
            w_ext = int((h * r_orig - w) // 2)
            if w_ext >= 0:
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
                box[:h, :w] = temp_box
            else:
                box = temp_box.copy()
    else:
        if r_temp > r_orig:
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0
            if h_ext >= 0:
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
                box[:h, :w] = temp_box
            else:
                box = temp_box.copy()
        else:
            w_ext = int((h * r_orig - w) / 2)
            if w_ext >= 0:
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
                box[:h, w_ext:w_ext + w] = temp_box
            else:
                box = temp_box.copy()

    src_pts = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)
    M, _ = cv2.findHomography(src_pts, dst_points, cv2.RANSAC, 5.0)
    rgba = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    x, y, rw, rh = cv2.boundingRect(dst_points.astype(np.int32))
    canvas = rgba[y:y + rh, x:x + rw, :3]
    mask = rgba[y:y + rh, x:x + rw, 3:4].astype(np.float32) / 255.0
    img[y:y + rh, x:x + rw] = np.clip(
        img[y:y + rh, x:x + rw].astype(np.float32) * (1 - mask) + canvas.astype(np.float32) * mask,
        0, 255,
    ).astype(np.uint8)
    return img


async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    line_spacing: int = None,
    disable_font_border: bool = False,
) -> np.ndarray:
    text_render.set_font(font_path)
    text_regions = [r for r in text_regions if r.translation]

    for region in text_regions:
        dst_points = region.min_rect
        img = _render_region(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
    return img
