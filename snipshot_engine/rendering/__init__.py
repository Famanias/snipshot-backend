# Rendering dispatch — places translated text back onto the image.

import cv2
import numpy as np
from typing import List, Optional
from shapely import affinity
from shapely.geometry import Polygon

from . import text_render
from .bubble import detect_bubbles
from ..utils import TextBlock, color_difference, get_logger, rotate_polygons

logger = get_logger("render")


def _fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg


def _count_text_length(text: str) -> float:
    half_width_chars = 'っッぁぃぅぇぉ'
    length = 0.0
    for char in text.strip():
        if char in half_width_chars:
            length += 0.5
        else:
            length += 1.0
    return length


def _resize_regions_to_font_size(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_size_offset: int,
    font_size_minimum: int,
):
    """Expand text bounding boxes when translated text is longer than the original."""
    if font_size_minimum == -1:
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 200)
    font_size_minimum = max(1, font_size_minimum)

    dst_points_list = []
    for region in text_regions:
        original_fs = region.font_size
        if original_fs <= 0:
            original_fs = font_size_minimum

        target_fs = original_fs + font_size_offset
        target_fs = max(target_fs, font_size_minimum, 1)

        single_axis_expanded = False
        dst_points = None

        # Try single-axis expansion based on how many rows/cols the translation needs
        if region.horizontal:
            used_rows = max(len(region.texts), 1)
            line_text_list, _ = text_render.calc_horizontal(
                region.font_size,
                region.translation,
                max_width=region.unrotated_size[0],
                max_height=region.unrotated_size[1],
                language=getattr(region, "target_lang", "en_US"),
            )
            needed_rows = len(line_text_list)
            if needed_rows > used_rows:
                scale_x = ((needed_rows - used_rows) / used_rows) + 1
                try:
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    poly = affinity.scale(poly, xfact=scale_x, yfact=1.0, origin=(minx, miny))
                    pts = np.array(poly.exterior.coords[:4])
                    dst_points = rotate_polygons(
                        region.center, pts.reshape(1, -1), -region.angle, to_int=False
                    ).reshape(-1, 4, 2).astype(np.int64)
                    single_axis_expanded = True
                except Exception:
                    pass

        elif region.vertical:
            used_cols = max(len(region.texts), 1)
            line_text_list, _ = text_render.calc_vertical(
                region.font_size,
                region.translation,
                max_height=region.unrotated_size[1],
            )
            needed_cols = len(line_text_list)
            if needed_cols > used_cols:
                scale_x = ((needed_cols - used_cols) / used_cols) + 1
                try:
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    poly = affinity.scale(poly, xfact=1.0, yfact=scale_x, origin=(minx, miny))
                    pts = np.array(poly.exterior.coords[:4])
                    dst_points = rotate_polygons(
                        region.center, pts.reshape(1, -1), -region.angle, to_int=False
                    ).reshape(-1, 4, 2).astype(np.int64)
                    single_axis_expanded = True
                except Exception:
                    pass

        # Fallback: general scaling based on text length difference
        if not single_axis_expanded:
            orig_text = getattr(region, "text_raw", region.text)
            char_count_orig = _count_text_length(orig_text)
            char_count_trans = _count_text_length(region.translation.strip())

            target_scale = 1.0
            if char_count_orig > 0 and char_count_trans > char_count_orig:
                increase_pct = (char_count_trans - char_count_orig) / char_count_orig
                font_increase_ratio = min(1.5, max(1.0, 1 + increase_pct * 0.3))
                target_fs = int(target_fs * font_increase_ratio)
                target_scale = max(1, min(1 + increase_pct * 0.3, 2))

            font_size_scale = (
                (((target_fs - original_fs) / original_fs) * 0.4 + 1)
                if original_fs > 0 else 1.0
            )
            final_scale = max(font_size_scale, target_scale)
            final_scale = max(1, min(final_scale, 1.1))

            if final_scale > 1.001:
                try:
                    poly = Polygon(region.unrotated_min_rect[0])
                    poly = affinity.scale(poly, xfact=final_scale, yfact=final_scale, origin='center')
                    scaled_pts = np.array(poly.exterior.coords[:4])
                    dst_points = rotate_polygons(
                        region.center, scaled_pts.reshape(1, -1), -region.angle, to_int=False
                    ).reshape(-1, 4, 2).astype(np.int64)
                except Exception:
                    dst_points = region.min_rect
            else:
                dst_points = region.min_rect

        if dst_points is None:
            dst_points = region.min_rect

        dst_points_list.append(dst_points)
        region.font_size = int(target_fs)

    return dst_points_list


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Reorder 4 corner points to [TL, TR, BR, BL] regardless of input ordering."""
    pts = pts.reshape(4, 2).astype(np.float64)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()  # y - x
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl])


def _render_region(img, region: TextBlock, dst_points, hyphenate, line_spacing, disable_font_border):
    fg, bg = region.get_font_colors()
    fg, bg = _fg_bg_compare(fg, bg)
    if disable_font_border:
        bg = None

    # Reorder dst_points to [TL, TR, BR, BL] so the homography maps correctly
    dst_points = _order_points(dst_points[0])[np.newaxis].astype(dst_points.dtype)

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
    box = None
    if render_h:
        if r_temp > r_orig:
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0
            if h_ext >= 0:
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
                box[h_ext:h_ext + h, 0:w] = temp_box
            else:
                box = temp_box.copy()
        else:
            w_ext = int((h * r_orig - w) // 2)
            if w_ext >= 0:
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
                box[0:h, 0:w] = temp_box
            else:
                box = temp_box.copy()
    else:
        if r_temp > r_orig:
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0
            if h_ext >= 0:
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
                box[0:h, 0:w] = temp_box
            else:
                box = temp_box.copy()
        else:
            w_ext = int((h * r_orig - w) / 2)
            if w_ext >= 0:
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
                box[0:h, w_ext:w_ext + w] = temp_box
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


def _find_optimal_font_size(text, box_w, box_h, initial_fs, lang, min_fs=10):
    """Binary-search for the largest font size that fits *text* inside *box_w* × *box_h*."""
    # Upper bound: at most half the box width (so calc_horizontal won't auto-expand),
    # a fraction of box height, and never absurdly large.
    max_fs = min(int(box_w / 2), int(box_h * 0.4), initial_fs * 3, 80)
    lo, hi = min_fs, max(min_fs, max_fs)
    best = min_fs

    for _ in range(12):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        if mid < 1:
            break

        lines, _ = text_render.calc_horizontal(mid, text, box_w, box_h, lang)
        # Match put_text_horizontal canvas height:
        #   canvas_h = fs * n_lines + spacing*(n-1) + (fs + bg_size)*2
        bg_size = int(max(mid * 0.07, 1))
        total_h = mid * len(lines) + (mid + bg_size) * 2

        if total_h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


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

    # ── 1. Detect speech bubbles ─────────────────────────────────────
    bubble_rects = detect_bubbles(img, text_regions)

    dst_points_list: list = [None] * len(text_regions)
    non_bubble_indices: list[int] = []
    non_bubble_regions: list[TextBlock] = []

    for i, (region, bubble_rect) in enumerate(zip(text_regions, bubble_rects)):
        if bubble_rect is not None:
            bw = int(bubble_rect[0, 1, 0] - bubble_rect[0, 0, 0])
            bh = int(bubble_rect[0, 2, 1] - bubble_rect[0, 0, 1])
            optimal_fs = _find_optimal_font_size(
                region.get_translation_for_rendering(),
                bw, bh,
                region.font_size,
                getattr(region, "target_lang", "en_US"),
            )
            region.font_size = optimal_fs
            dst_points_list[i] = bubble_rect
        else:
            non_bubble_indices.append(i)
            non_bubble_regions.append(region)

    # ── 2. Fallback: expand textline boxes for non-bubble regions ────
    if non_bubble_regions:
        fallback = _resize_regions_to_font_size(
            img, non_bubble_regions, font_size_offset, font_size_minimum,
        )
        for idx, pts in zip(non_bubble_indices, fallback):
            dst_points_list[idx] = pts

    # ── 3. Render ────────────────────────────────────────────────────
    for region, dst_points in zip(text_regions, dst_points_list):
        img = _render_region(img, region, dst_points, hyphenate, line_spacing, disable_font_border)
    return img
