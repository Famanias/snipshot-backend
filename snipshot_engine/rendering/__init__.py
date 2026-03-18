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
            base_scale = max(font_size_scale, target_scale)
            over_x, over_y = _estimate_overflow_scales(region, target_fs)
            final_scale_x = max(1.0, min(max(base_scale, over_x), 1.15))
            final_scale_y = max(1.0, min(max(base_scale, over_y), 1.15))

            if final_scale_x > 1.001 or final_scale_y > 1.001:
                try:
                    poly = Polygon(region.unrotated_min_rect[0])
                    poly = affinity.scale(poly, xfact=final_scale_x, yfact=final_scale_y, origin='center')
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


def _compute_inner_margin(target_w: int, target_h: int, text_len: int) -> int:
    """Compute smooth, text-aware inner margin for bubble rendering."""
    min_dim = max(1, min(target_w, target_h))
    base = 0.06 * min_dim + 2.0
    density_factor = max(0.75, min(1.15, 1.05 - 0.0035 * text_len))
    margin = int(round(base * density_factor))
    return max(2, min(20, margin))


def _center_text_in_box(temp_box: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center rendered text in a target box, shrinking proportionally if needed."""
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))

    h, w, _ = temp_box.shape
    content = temp_box

    if h > target_h or w > target_w:
        sx = target_w / max(w, 1)
        sy = target_h / max(h, 1)
        scale = max(0.1, min(sx, sy))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        content = cv2.resize(temp_box, (new_w, new_h), interpolation=interp)
        h, w = new_h, new_w

    out = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    x0 = max(0, (target_w - w) // 2)
    y0 = max(0, (target_h - h) // 2)
    x1 = min(target_w, x0 + w)
    y1 = min(target_h, y0 + h)
    out[y0:y1, x0:x1] = content[: y1 - y0, : x1 - x0]
    return out


def _estimate_overflow_scales(region: TextBlock, target_fs: int) -> tuple[float, float]:
    """Estimate anisotropic expansion scales for non-bubble overflow handling."""
    translation = region.get_translation_for_rendering()
    lang = getattr(region, "target_lang", "en_US")

    if region.horizontal:
        lines, widths = text_render.calc_horizontal(
            target_fs,
            translation,
            max_width=region.unrotated_size[0],
            max_height=region.unrotated_size[1],
            language=lang,
        )
        used_rows = max(len(region.texts), 1)
        needed_rows = max(len(lines), 1)
        row_overflow = max(1.0, needed_rows / used_rows)
        width_overflow = max(1.0, (max(widths) if widths else 0) / max(region.unrotated_size[0], 1))
        scale_x = 1.0 + 0.30 * (width_overflow - 1.0) + 0.20 * (row_overflow - 1.0)
        scale_y = 1.0 + 0.55 * (row_overflow - 1.0)
    else:
        cols, col_heights = text_render.calc_vertical(
            target_fs,
            translation,
            max_height=region.unrotated_size[1],
        )
        used_cols = max(len(region.texts), 1)
        needed_cols = max(len(cols), 1)
        col_overflow = max(1.0, needed_cols / used_cols)
        height_overflow = max(1.0, (max(col_heights) if col_heights else 0) / max(region.unrotated_size[1], 1))
        scale_x = 1.0 + 0.55 * (col_overflow - 1.0)
        scale_y = 1.0 + 0.30 * (height_overflow - 1.0) + 0.20 * (col_overflow - 1.0)

    return min(max(scale_x, 1.0), 1.15), min(max(scale_y, 1.0), 1.15)


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

    target_w = max(1, int(round(norm_h[0])))
    target_h = max(1, int(round(norm_v[0])))
    margin = _compute_inner_margin(target_w, target_h, len(region.get_translation_for_rendering()))

    inner_w = max(1, target_w - margin * 2)
    inner_h = max(1, target_h - margin * 2)
    centered_inner = _center_text_in_box(temp_box, inner_w, inner_h)

    box = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    ox = max(0, (target_w - inner_w) // 2)
    oy = max(0, (target_h - inner_h) // 2)
    box[oy:oy + inner_h, ox:ox + inner_w] = centered_inner

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
    """
    Binary-search for the largest font size that fits text inside box.
    
    PHASE 1 IMPROVEMENTS:
    - Smooth formula based on min(box_w, box_h) instead of discrete classes
    - Accounts for padding and line spacing (15% instead of 1%)
    - More iterations for precision (14 instead of 12)
    - Adaptive margin calculation
    """
    # Calculate adaptive safe margin (5-12% of smallest dimension)
    min_dim = min(box_w, box_h)
    margin_ratio = max(0.05, min(0.12, 8 / min_dim))  # Smooth curve
    margin = max(8, int(min_dim * margin_ratio))
    
    safe_w = max(margin * 2, box_w - margin * 2)
    safe_h = max(margin * 2, box_h - margin * 2)
    
    # Smooth formula for max font size based on smallest dimension
    # Small bubbles (< 80px): Conservative to ensure readability
    # Medium bubbles (80-200px): Linear growth
    # Large bubbles (> 200px): Capped but generous
    if min_dim < 80:
        max_fs = min(int(min_dim * 0.35), 28)
    elif min_dim < 200:
        # Linear interpolation: 28px at 80px -> 48px at 200px
        max_fs = int(28 + (min_dim - 80) * 0.167)
    else:
        max_fs = min(int(min_dim * 0.28), 72)
    
    # Don't exceed reasonable bounds
    max_fs = min(max_fs, initial_fs * 2.5)  # Max 2.5x original
    max_fs = max(max_fs, min_fs + 4)         # Ensure room for growth
    
    lo, hi = min_fs, max_fs
    best = min_fs
    
    # More iterations for precision
    for _ in range(14):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        if mid < 1:
            break
        
        lines, widths = text_render.calc_horizontal(mid, text, safe_w, safe_h, lang)
        
        # Account for improved line spacing (15% instead of 1%)
        bg_size = int(max(mid * 0.07, 1))
        line_spacing_px = max(int(mid * 0.15), 3)  # New improved spacing
        total_h = mid * len(lines) + line_spacing_px * max(0, len(lines) - 1) + margin * 2
        
        max_line_w = max(widths) if widths else 0
        
        # Check if text fits with margin buffer
        if total_h <= box_h and max_line_w + margin * 2 <= box_w:
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
