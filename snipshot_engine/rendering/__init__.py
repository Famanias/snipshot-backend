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


def _fallback_scale_cap(region: TextBlock, severity: float) -> float:
    """Single dynamic cap for fallback expansion, confidence-driven."""
    bubble_conf = float(getattr(region, "_bubble_confidence", 0.0) or 0.0)

    if bubble_conf < 0.25:
        cap = 2.5
    elif bubble_conf < 0.45:
        cap = 2.2
    elif bubble_conf < 0.65:
        cap = 1.9
    elif bubble_conf < 0.80:
        cap = 1.7
    else:
        cap = 1.5

    if severity > 2.0:
        cap = min(2.5, cap + 0.1)
    return cap


def _early_fallback_bias(region: TextBlock) -> float:
    """Small optional bias for very low-confidence fallback only."""
    bubble_conf = float(getattr(region, "_bubble_confidence", 0.0) or 0.0)
    ratio = _translation_length_ratio(region)

    if bubble_conf < 0.20 and ratio > 1.2:
        return 1.2
    if bubble_conf < 0.30 and ratio > 1.6:
        return 1.1
    return 1.0


def _translation_length_ratio(region: TextBlock) -> float:
    orig_text = getattr(region, "text_raw", region.text)
    char_count_orig = _count_text_length(orig_text)
    char_count_trans = _count_text_length((region.translation or "").strip())
    if char_count_orig <= 0:
        return 1.0
    return max(0.5, char_count_trans / char_count_orig)


def _rect_to_quad(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype=np.int64)


def _quad_from_inpaint_bbox(region: TextBlock) -> Optional[np.ndarray]:
    """Try to read inpaint-aligned bbox if available on the region."""
    bbox = getattr(region, "inpaint_bbox", None)
    if bbox is None:
        return None

    try:
        # dict style: {x, y, w, h}
        if isinstance(bbox, dict):
            x, y = int(bbox["x"]), int(bbox["y"])
            w, h = int(bbox["w"]), int(bbox["h"])
            if w > 2 and h > 2:
                return _rect_to_quad(x, y, w, h)

        # tuple/list style: (x, y, w, h)
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            if w > 2 and h > 2:
                return _rect_to_quad(x, y, w, h)

        # 4-point quad style: [[x,y], ...]
        arr = np.array(bbox, dtype=np.int64)
        if arr.ndim == 2 and arr.shape == (4, 2):
            return arr.reshape(1, 4, 2)
        if arr.ndim == 3 and arr.shape[1:] == (4, 2):
            return arr[:1]
    except Exception:
        return None

    return None


def _get_region_base_quad(region: TextBlock) -> np.ndarray:
    """Preferred placement quad: inpaint-aligned bbox, then region min rect."""
    inpaint_quad = _quad_from_inpaint_bbox(region)
    if inpaint_quad is not None:
        return inpaint_quad.astype(np.int64)
    return region.min_rect.astype(np.int64)


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
        base_quad = _get_region_base_quad(region)
        _, _, base_w, base_h = cv2.boundingRect(base_quad[0].astype(np.int32))
        base_w = max(1, int(base_w))
        base_h = max(1, int(base_h))

        original_fs = region.font_size
        if original_fs <= 0:
            original_fs = font_size_minimum

        target_fs = original_fs + font_size_offset
        target_fs = max(target_fs, font_size_minimum, 1)

        # Keep font-size nudging mild; geometry scaling is driven by overflow estimates.
        ratio = _translation_length_ratio(region)
        if ratio > 1.0:
            target_fs = int(round(target_fs * min(1.35, 1.0 + 0.18 * (ratio - 1.0))))
            target_fs = max(target_fs, font_size_minimum, 1)

        # Single overflow-driven scaling decision on the true base region.
        base_scale = 1.0
        if original_fs > 0:
            fs_growth = max(0.0, (target_fs - original_fs) / float(original_fs))
            base_scale = 1.0 + 0.35 * fs_growth

        over_x, over_y = _estimate_overflow_scales(
            region,
            target_fs,
            avail_w=base_w,
            avail_h=base_h,
        )
        severity = max(base_scale, over_x, over_y)
        cap = _fallback_scale_cap(region, severity)

        final_scale_x = min(max(max(base_scale, over_x), 1.0), cap)
        final_scale_y = min(max(max(base_scale, over_y), 1.0), cap)

        if final_scale_x > 1.001 or final_scale_y > 1.001:
            try:
                poly = Polygon(base_quad[0])
                poly = affinity.scale(poly, xfact=final_scale_x, yfact=final_scale_y, origin='center')
                scaled_pts = np.array(poly.exterior.coords[:4])
                dst_points = scaled_pts.reshape(-1, 4, 2).astype(np.int64)
            except Exception:
                dst_points = base_quad
        else:
            dst_points = base_quad

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

    # Tiny boxes: keep margins minimal to preserve usable layout space.
    if min_dim < 70:
        return max(1, int(round(min_dim * 0.02)))

    if min_dim < 120:
        return max(2, int(round(min_dim * 0.03)))

    base = 0.040 * min_dim + 1.0
    density_factor = max(0.86, min(1.08, 1.02 - 0.0022 * text_len))
    margin = int(round(base * density_factor))
    return max(2, min(10, margin))


def _center_text_in_box(temp_box: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center rendered text in a target box without additional raster shrink."""
    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))

    h, w, _ = temp_box.shape
    content = temp_box

    out = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    draw_w = min(w, target_w)
    draw_h = min(h, target_h)

    x0 = max(0, (target_w - draw_w) // 2)
    y0 = max(0, (target_h - draw_h) // 2)

    src_x0 = max(0, (w - draw_w) // 2)
    src_y0 = max(0, (h - draw_h) // 2)
    out[y0:y0 + draw_h, x0:x0 + draw_w] = content[src_y0:src_y0 + draw_h, src_x0:src_x0 + draw_w]
    return out


def _render_temp_text_box(
    region: TextBlock,
    font_size: int,
    width: int,
    height: int,
    fg,
    bg,
    hyphenate,
    line_spacing,
    render_h: bool,
):
    if render_h:
        return text_render.put_text_horizontal(
            font_size,
            region.get_translation_for_rendering(),
            width,
            height,
            region.alignment,
            region.direction == 'hl',
            fg,
            bg,
            region.target_lang,
            hyphenate,
            line_spacing,
        )
    return text_render.put_text_vertical(
        font_size,
        region.get_translation_for_rendering(),
        height,
        region.alignment,
        fg,
        bg,
        line_spacing,
    )


def _sanitize_dst_quad(dst_points: np.ndarray, region: TextBlock) -> np.ndarray:
    """Sanitize destination quadrilateral to reduce homography placement errors."""
    pts = _order_points(dst_points[0]).astype(np.float32)

    # Use region min rect if the incoming quad is degenerate.
    def _region_fallback():
        return _order_points(_get_region_base_quad(region)[0]).astype(np.float32)

    area = abs(cv2.contourArea(pts.astype(np.int32)))
    if area < 20:
        pts = _region_fallback()

    if not cv2.isContourConvex(pts.astype(np.int32)):
        pts = _region_fallback()

    edges = [
        np.linalg.norm(pts[1] - pts[0]),
        np.linalg.norm(pts[2] - pts[1]),
        np.linalg.norm(pts[3] - pts[2]),
        np.linalg.norm(pts[0] - pts[3]),
    ]
    if min(edges) < 4.0:
        pts = _region_fallback()

    # If destination is near-axis aligned but region angle is not, nudge orientation.
    vec = pts[1] - pts[0]
    quad_angle = np.degrees(np.arctan2(float(vec[1]), float(vec[0])))
    region_angle = float(getattr(region, "angle", 0.0) or 0.0)
    if abs(region_angle) > 10.0 and abs(quad_angle) < 3.0 and abs(region_angle) < 45.0:
        rot_deg = float(np.clip(region_angle * 0.35, -10.0, 10.0))
        theta = np.deg2rad(rot_deg)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        center = np.mean(pts, axis=0, keepdims=True)
        pts = (pts - center) @ R.T + center

    return pts[np.newaxis].astype(np.float32)


def _estimate_overflow_scales(
    region: TextBlock,
    target_fs: int,
    avail_w: Optional[float] = None,
    avail_h: Optional[float] = None,
) -> tuple[float, float]:
    """Estimate overflow-driven expansion scales from the current base region size."""
    translation = region.get_translation_for_rendering()
    lang = getattr(region, "target_lang", "en_US")

    base_w = float(avail_w if avail_w is not None else region.unrotated_size[0])
    base_h = float(avail_h if avail_h is not None else region.unrotated_size[1])
    base_w = max(base_w, 1.0)
    base_h = max(base_h, 1.0)

    if region.horizontal:
        lines, widths = text_render.calc_horizontal(
            target_fs,
            translation,
            max_width=base_w,
            max_height=base_h,
            language=lang,
        )
        used_rows = max(len(region.texts), 1)
        needed_rows = max(len(lines), 1)
        row_overflow = max(1.0, needed_rows / used_rows)
        width_overflow = max(1.0, (max(widths) if widths else 0) / base_w)
        scale_x = max(width_overflow, 1.0 + 0.35 * (row_overflow - 1.0))
        scale_y = max(1.0, row_overflow)
    else:
        cols, col_heights = text_render.calc_vertical(
            target_fs,
            translation,
            max_height=base_h,
        )
        used_cols = max(len(region.texts), 1)
        needed_cols = max(len(cols), 1)
        col_overflow = max(1.0, needed_cols / used_cols)
        height_overflow = max(1.0, (max(col_heights) if col_heights else 0) / base_h)
        scale_x = max(1.0, col_overflow)
        scale_y = max(height_overflow, 1.0 + 0.35 * (col_overflow - 1.0))

    return min(max(scale_x, 1.0), 2.5), min(max(scale_y, 1.0), 2.5)


def _render_region(img, region: TextBlock, dst_points, hyphenate, line_spacing, disable_font_border):
    fg, bg = region.get_font_colors()
    fg, bg = _fg_bg_compare(fg, bg)
    if disable_font_border:
        bg = None

    # Sanitize destination points so homography does not amplify bad geometry.
    dst_points = _sanitize_dst_quad(dst_points, region)

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)

    forced_dir = region._direction if hasattr(region, "_direction") else region.direction
    if forced_dir != "auto":
        render_h = forced_dir in ("horizontal", "h")
    else:
        render_h = region.horizontal

    target_w = max(1, int(round(norm_h[0])))
    target_h = max(1, int(round(norm_v[0])))

    temp_box = _render_temp_text_box(
        region,
        region.font_size,
        target_w,
        target_h,
        fg,
        bg,
        hyphenate,
        line_spacing,
        render_h,
    )

    if temp_box is None:
        return img

    margin = _compute_inner_margin(target_w, target_h, len(region.get_translation_for_rendering()))

    inner_w = max(1, target_w - margin * 2)
    inner_h = max(1, target_h - margin * 2)

    # Fit by font-size adjustment only. Avoid additional raster downscaling.
    bubble_conf = float(getattr(region, "_bubble_confidence", 0.0) or 0.0)
    shrink_step_ratio = 0.03 if bubble_conf < 0.45 else 0.05
    max_fit_steps = 4 if bubble_conf < 0.45 else 6

    fit_steps = 0
    while (
        (temp_box.shape[1] - inner_w > 2 or temp_box.shape[0] - inner_h > 2)
        and fit_steps < max_fit_steps
    ):
        fs0 = int(region.font_size)
        trial_fs = max(8, fs0 - max(1, int(round(fs0 * shrink_step_ratio))))
        if trial_fs >= fs0:
            break
        prev_over = max(
            max(temp_box.shape[1] - inner_w, 0) / max(inner_w, 1),
            max(temp_box.shape[0] - inner_h, 0) / max(inner_h, 1),
        )
        trial_box = _render_temp_text_box(
            region,
            trial_fs,
            target_w,
            target_h,
            fg,
            bg,
            hyphenate,
            line_spacing,
            render_h,
        )
        if trial_box is None:
            break
        next_over = max(
            max(trial_box.shape[1] - inner_w, 0) / max(inner_w, 1),
            max(trial_box.shape[0] - inner_h, 0) / max(inner_h, 1),
        )
        if next_over > prev_over - 0.01:
            break
        temp_box = trial_box
        region.font_size = trial_fs
        fit_steps += 1

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


def _find_optimal_font_size(
    text,
    box_w,
    box_h,
    initial_fs,
    lang,
    min_fs=10,
    render_h: bool = True,
    line_spacing: Optional[float] = None,
    region: Optional[TextBlock] = None,
):
    """
    Binary-search for the largest font size that fits text inside box.
    
    PHASE 1 IMPROVEMENTS:
    - Smooth formula based on min(box_w, box_h) instead of discrete classes
    - Accounts for padding and line spacing (15% instead of 1%)
    - More iterations for precision (14 instead of 12)
    - Adaptive margin calculation
    """
    bubble_conf = float(getattr(region, "_bubble_confidence", 0.0) if region is not None else 0.0)
    text_len = _count_text_length(text)

    # Calculate adaptive safe margin (looser for low-confidence fallback regions)
    min_dim = max(1, min(box_w, box_h))
    # Higher text density in the same region is treated as a complex case and
    # gets a slightly more conservative fit target.
    density = text_len / float(max(min_dim, 1))
    complexity = max(0.0, min(1.0, (density - 0.22) / 0.35))

    margin_ratio = max(0.02, min(0.10, 7 / min_dim))
    if bubble_conf < 0.45:
        margin_ratio *= 0.85
    margin_ratio *= 1.0 + 0.20 * complexity
    margin = max(4, int(min_dim * margin_ratio))

    horizontal_spacing = float(line_spacing) if line_spacing is not None else 0.15
    vertical_spacing = float(line_spacing) if line_spacing is not None else 0.10
    
    safe_w = max(margin * 2, box_w - margin * 2)
    safe_h = max(margin * 2, box_h - margin * 2)
    
    # Smooth formula for max font size based on smallest dimension
    if min_dim < 80:
        max_fs = min(int(min_dim * 0.56), 40)
    elif min_dim < 200:
        max_fs = int(40 + (min_dim - 80) * 0.30)
    else:
        max_fs = min(int(min_dim * 0.44), 168)

    # Tone down the upper bound further for dense/complex bubbles.
    max_fs = int(round(max_fs * (1.0 - 0.12 * complexity)))
    
    # Relax growth caps, especially for low-confidence fallback regions.
    growth_mult = 3.2 if bubble_conf < 0.45 else 2.6
    max_fs = min(max_fs, int(max(initial_fs * growth_mult, min_fs + 8)))
    max_fs = max(max_fs, min_fs + 6)

    # Tighten fit slack when complexity is high to avoid visually oversized text.
    slack_tighten = 0.03 * complexity
    
    lo, hi = min_fs, max_fs
    best = min_fs
    
    # More iterations for precision
    for _ in range(14):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        if mid < 1:
            break
        
        # Layout-aware fit check (horizontal and vertical text behave differently).
        if render_h:
            lines, widths = text_render.calc_horizontal(mid, text, safe_w, safe_h, lang)
            line_spacing_px = max(int(mid * horizontal_spacing), 3)
            total_h = mid * len(lines) + line_spacing_px * max(0, len(lines) - 1) + margin * 2
            max_line_w = max(widths) if widths else 0
            h_slack = (1.04 if bubble_conf < 0.45 else 1.01) - slack_tighten
            w_slack = (1.05 if bubble_conf < 0.45 else 1.02) - slack_tighten
            h_slack = max(1.0, h_slack)
            w_slack = max(1.0, w_slack)
            fits = total_h <= box_h * h_slack and (max_line_w + margin * 2) <= box_w * w_slack
        else:
            cols, col_heights = text_render.calc_vertical(mid, text, safe_h)
            col_spacing = max(int(mid * vertical_spacing), 2)
            total_w = mid * len(cols) + col_spacing * max(0, len(cols) - 1) + margin * 2
            max_col_h = max(col_heights) if col_heights else 0
            w_slack = (1.04 if bubble_conf < 0.45 else 1.01) - slack_tighten
            h_slack = (1.05 if bubble_conf < 0.45 else 1.02) - slack_tighten
            w_slack = max(1.0, w_slack)
            h_slack = max(1.0, h_slack)
            fits = total_w <= box_w * w_slack and (max_col_h + margin * 2) <= box_h * h_slack

        if fits:
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

            forced_dir = region._direction if hasattr(region, "_direction") else region.direction
            render_h = (forced_dir in ("horizontal", "h")) if forced_dir != "auto" else region.horizontal

            optimal_fs = _find_optimal_font_size(
                region.get_translation_for_rendering(),
                bw, bh,
                region.font_size,
                getattr(region, "target_lang", "en_US"),
                render_h=render_h,
                line_spacing=line_spacing,
                region=region,
            )
            region.font_size = optimal_fs
            dst_points_list[i] = bubble_rect
        else:
            region._bubble_confidence = float(getattr(region, "_bubble_confidence", 0.0) or 0.0)
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