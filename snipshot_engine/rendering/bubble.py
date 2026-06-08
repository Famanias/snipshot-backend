"""Speech bubble detection — find bubble boundaries for accurate text placement.

Uses flood fill on the inpainted image (text already erased) to find
the enclosing speech bubble for each text region. The bubble interior
rectangle is used as the rendering target instead of the tight textline
bounding box, giving translated text much more room to breathe.
"""

import cv2
import numpy as np
from typing import List, Optional

from ..utils import TextBlock, get_logger

logger = get_logger("bubble")


def detect_bubbles(
    inpainted_img: np.ndarray,
    text_regions: List[TextBlock],
    min_bubble_area: int = 800,
    max_bubble_ratio: float = 0.3,
    padding: int = 12,
) -> List[Optional[np.ndarray]]:
    """
    For each text region, detect the enclosing speech bubble.

    Returns a list (one per region) of dst_points ``(1, 4, 2)`` int64 arrays
    representing the bubble interior rectangle, or ``None`` when no clear
    bubble is found (falls back to textline bounding box in the caller).
    """
    if inpainted_img.ndim == 3:
        gray = cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = inpainted_img.copy()

    # Slight blur reduces pixel-level noise that can stop flood fill prematurely
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    img_area = h * w

    results: List[Optional[np.ndarray]] = []
    for idx, region in enumerate(text_regions):
        cx = max(0, min(int(region.center[0]), w - 1))
        cy = max(0, min(int(region.center[1]), h - 1))

        detected = _detect_single_bubble(
            gray, cx, cy, h, w, img_area, region,
            min_bubble_area, max_bubble_ratio, padding,
        )
        rect = None
        conf = 0.0
        if detected is not None:
            rect, conf = detected
        region._bubble_confidence = conf

        if rect is not None:
            bw = int(rect[0, 1, 0] - rect[0, 0, 0])
            bh = int(rect[0, 2, 1] - rect[0, 0, 1])
            logger.debug(
                "Region %d: BUBBLE %dx%d at (%d,%d), conf=%.2f",
                idx,
                bw,
                bh,
                int(rect[0, 0, 0]),
                int(rect[0, 0, 1]),
                conf,
            )
        else:
            logger.debug("Region %d: no bubble detected", idx)
        results.append(rect)

    # If multiple regions share the same bubble, split the space.
    _resolve_overlaps(text_regions, results)
    return results


# ── internals ────────────────────────────────────────────────────────


def find_largest_inscribed_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Finds the largest axis-aligned rectangle fully contained within the white pixels (255) of a binary mask.
    Returns (x, y, w, h). If no rectangle is found, returns (0, 0, 0, 0).
    """
    if mask is None or mask.size == 0:
        return 0, 0, 0, 0
    
    H, W = mask.shape
    heights = np.zeros(W, dtype=np.int32)
    max_area = 0
    best_x, best_y, best_w, best_h = 0, 0, 0, 0
    
    for r in range(H):
        row_vals = mask[r]
        heights = np.where(row_vals > 0, heights + 1, 0)
        
        stack = []
        for c in range(W + 1):
            h = heights[c] if c < W else 0
            start_c = c
            while stack and stack[-1][1] > h:
                pos_c, height = stack.pop()
                width = c - pos_c
                area = width * height
                if area > max_area:
                    max_area = area
                    best_x = pos_c
                    best_y = r - height + 1
                    best_w = width
                    best_h = height
                start_c = pos_c
            stack.append((start_c, h))
            
    return int(best_x), int(best_y), int(best_w), int(best_h)


def _detect_single_bubble(
    gray, cx, cy, h, w, img_area, region,
    min_area, max_ratio, padding,
) -> Optional[tuple[np.ndarray, float]]:
    """Flood-fill from the text center to find the enclosing bubble."""
    tx, ty, tw, th = cv2.boundingRect(region.min_rect[0].astype(np.int32))
    text_area = max(1, tw * th)
    text_len = len(getattr(region, "translation", "") or region.text or "")

    min_area_dyn = max(min_area, int(text_area * 0.9), 300)
    max_area_dyn = int(img_area * max_ratio)

    best_rect: Optional[np.ndarray] = None
    best_score = -1.0
    best_component = None
    best_bbox = None

    for sx, sy in _candidate_seed_points(cx, cy, tw, th, w, h):
        # ── 1. Flood fill from candidate seed ───────────────────────
        bubble_mask, flood_area = _flood_fill(gray, sx, sy)
        if flood_area < min_area_dyn:
            continue
        if flood_area > max_area_dyn:
            continue
        if flood_area < text_area * 1.05:
            continue

        contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        compactness = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area > 0:
                compactness = cv2.contourArea(cnt) / hull_area

        # Relax compactness threshold for dense text to avoid over-rejection.
        density = text_len / max(np.sqrt(max(flood_area, 1)), 1.0)
        compact_thresh = max(0.28, min(0.42, 0.40 - 0.10 * density))
        if compactness > 0 and compactness < compact_thresh:
            continue

        # ── 2. Erode to safe interior while keeping selected seed inside ────
        adaptive_padding = _adaptive_padding(flood_area, region, tw, th, padding)
        eroded = _erode_mask(bubble_mask, sx, sy, adaptive_padding)
        if eroded is None:
            continue

        num_labels, labels = cv2.connectedComponents(eroded)
        label_at_seed = labels[sy, sx]
        if label_at_seed == 0:
            continue

        component = (labels == label_at_seed).astype(np.uint8) * 255
        bx, by, bw, bh = cv2.boundingRect(component)

        min_dim = max(18, int(min(tw, th) * 0.55))
        if bw < min_dim or bh < min_dim:
            continue
        if region.horizontal and bw < tw * 0.7:
            continue

        score = _bubble_confidence_score(flood_area, text_area, compactness, bw, bh, tw, th)
        if score > best_score:
            best_score = score
            best_rect = _rect_to_dst(bx, by, bw, bh)
            best_component = component
            best_bbox = (bx, by, bw, bh)

    if best_rect is None or best_component is None or best_bbox is None:
        return None

    # Calculate Largest Inscribed Rectangle exactly once on the final selected best component
    lir_x, lir_y, lir_w, lir_h = find_largest_inscribed_rectangle(best_component)
    bx, by, bw, bh = best_bbox
    bbox_area = max(1, bw * bh)
    lir_area = lir_w * lir_h

    # Log areas and retention rate
    retention_pct = (lir_area / bbox_area) * 100.0
    logger.info(
        "Bubble LIR Area Calculation - BBox: %d px², LIR: %d px², Retention: %.1f%%",
        bbox_area,
        lir_area,
        retention_pct,
    )

    # 45% Area-Loss Fallback check
    if lir_area < 0.45 * bbox_area or lir_w <= 0 or lir_h <= 0:
        logger.info("LIR Area Retention below threshold (45%%) or invalid LIR. Falling back to component bounding box.")
        final_rect = best_rect
    else:
        # Perform geometric validity assertion: LIR must be subset of component mask
        assert np.all(best_component[lir_y:lir_y+lir_h, lir_x:lir_x+lir_w] == 255), "LIR geometric validation failed: rectangle contains background pixels"

        # Apply safety margin (2% inset, min 2px, max 5px)
        inset_w = max(2, min(5, int(round(lir_w * 0.02))))
        inset_h = max(2, min(5, int(round(lir_h * 0.02))))
        
        # Only apply inset if it leaves enough space (at least 10px in width/height)
        if lir_w - 2 * inset_w >= 10 and lir_h - 2 * inset_h >= 10:
            lir_x += inset_w
            lir_y += inset_h
            lir_w -= 2 * inset_w
            lir_h -= 2 * inset_h

        final_rect = _rect_to_dst(lir_x, lir_y, lir_w, lir_h)

    return final_rect, float(max(0.0, min(1.0, best_score)))


def _candidate_seed_points(cx: int, cy: int, tw: int, th: int, w: int, h: int) -> list[tuple[int, int]]:
    delta = max(2, int(round(min(tw, th) * 0.10)))
    pts = [
        (cx, cy),
        (cx + delta, cy),
        (cx - delta, cy),
        (cx, cy + delta),
        (cx, cy - delta),
    ]
    uniq = set()
    out = []
    for x, y in pts:
        sx = max(0, min(int(x), w - 1))
        sy = max(0, min(int(y), h - 1))
        key = (sx, sy)
        if key not in uniq:
            uniq.add(key)
            out.append(key)
    return out


def _flood_fill(gray: np.ndarray, sx: int, sy: int) -> tuple[np.ndarray, int]:
    h, w = gray.shape
    # Get seed pixel value dynamically
    seed_val = int(gray[sy, sx])
    
    # Binarize: pixels close to seed value (within 35 levels) become 255 (foreground), others become 0 (border).
    # This prevents the flood fill from leaking through smudged/blurry inpainted bubble borders.
    thresh = np.where((gray >= max(0, seed_val - 35)) & (gray <= min(255, seed_val + 35)), 255, 0).astype(np.uint8)
    
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Simple binary flood fill on the thresholded image
    cv2.floodFill(
        thresh,
        flood_mask,
        (int(sx), int(sy)),
        newVal=0,
        loDiff=(10,),
        upDiff=(10,),
        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    bubble_mask = flood_mask[1:-1, 1:-1]
    flood_area = int(np.sum(bubble_mask > 0))
    return bubble_mask, flood_area


def _bubble_confidence_score(
    flood_area: int,
    text_area: int,
    compactness: float,
    bw: int,
    bh: int,
    tw: int,
    th: int,
) -> float:
    area_ratio = flood_area / max(text_area, 1)
    area_term = max(0.0, min(1.0, (area_ratio - 1.0) / 4.0))
    compact_term = max(0.0, min(1.0, compactness))

    width_fit = max(0.0, min(1.0, bw / max(tw, 1)))
    height_fit = max(0.0, min(1.0, bh / max(th, 1)))
    fit_term = 0.5 * width_fit + 0.5 * height_fit

    return 0.45 * area_term + 0.30 * compact_term + 0.25 * fit_term


def _erode_mask(mask, cx, cy, padding):
    """Erode *mask*, reducing padding until *center* is still inside."""
    h, w = mask.shape
    if cy < 0 or cy >= h or cx < 0 or cx >= w:
        return None

    for p in range(padding, 1, -2):
        kernel = np.ones((p * 2 + 1, p * 2 + 1), np.uint8)
        eroded = cv2.erode(mask, kernel)
        if eroded[cy, cx] > 0:
            return eroded

    # Minimal / no erosion
    if mask[cy, cx] > 0:
        return mask
    return None


def _adaptive_padding(
    flood_area: int,
    region: TextBlock,
    text_w: int,
    text_h: int,
    base_padding: int,
) -> int:
    """Compute erosion padding from bubble size and text density.

    Uses a smooth formula rather than fixed area classes:
    - Larger bubbles get more padding.
    - Longer/denser text gets less padding to avoid cramped rendering.
    - Preserves caller-provided ``base_padding`` as a soft prior.
    """
    bubble_dim = max(1.0, float(np.sqrt(max(flood_area, 1))))
    text_len = len(getattr(region, "translation", "") or region.text or "")

    # Size-driven padding component (smooth growth with bubble dimension)
    size_padding = 0.06 * bubble_dim

    # Text density proxy: higher density => less interior erosion
    text_density = text_len / max(bubble_dim, 1.0)
    density_factor = max(0.72, min(1.12, 1.06 - 0.16 * text_density))

    # Blend caller default with adaptive value for backward compatibility
    blended = (0.45 * float(base_padding) + 0.55 * size_padding) * density_factor

    # Prevent erosion from consuming tiny bubbles
    upper_bound = max(3, int(min(text_w, text_h) * 0.22))
    return int(max(3, min(24, min(upper_bound, round(blended)))))


def _resolve_overlaps(text_regions, bubble_rects):
    """Split shared bubbles among multiple text regions."""
    n = len(bubble_rects)
    for i in range(n):
        if bubble_rects[i] is None:
            continue
        for j in range(i + 1, n):
            if bubble_rects[j] is None:
                continue

            r1 = cv2.boundingRect(bubble_rects[i][0].astype(np.int32))
            r2 = cv2.boundingRect(bubble_rects[j][0].astype(np.int32))

            if _rect_iou(r1, r2) < 0.5:
                continue

            _split_shared_bubble(text_regions, bubble_rects, i, j)


def _rect_iou(r1, r2):
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = r1[2] * r1[3] + r2[2] * r2[3] - inter
    return inter / union if union > 0 else 0.0


def _split_shared_bubble(text_regions, bubble_rects, i, j):
    """Split shared bubble along dominant region-center axis."""
    ri = bubble_rects[i][0]
    bx = int(ri[0, 0])
    by = int(ri[0, 1])
    bw = int(ri[1, 0] - bx)
    bh = int(ri[2, 1] - by)

    cy_i = float(text_regions[i].center[1])
    cy_j = float(text_regions[j].center[1])
    cx_i = float(text_regions[i].center[0])
    cx_j = float(text_regions[j].center[0])

    if abs(cx_i - cx_j) > abs(cy_i - cy_j):
        _split_horizontally(text_regions, bubble_rects, i, j, bx, by, bw, bh, cx_i, cx_j)
    else:
        _split_vertically(text_regions, bubble_rects, i, j, bx, by, bw, bh, cy_i, cy_j)


def _split_vertically(text_regions, bubble_rects, i, j, bx, by, bw, bh, cy_i, cy_j):
    """Split a shared bubble between two regions by Y axis."""

    split_y = int((cy_i + cy_j) / 2)
    split_y = max(by + 10, min(split_y, by + bh - 10))

    if cy_i <= cy_j:
        h_top = split_y - by
        h_bot = by + bh - split_y
        bubble_rects[i] = _rect_to_dst(bx, by, bw, h_top)
        bubble_rects[j] = _rect_to_dst(bx, split_y, bw, h_bot)
    else:
        h_top = split_y - by
        h_bot = by + bh - split_y
        bubble_rects[j] = _rect_to_dst(bx, by, bw, h_top)
        bubble_rects[i] = _rect_to_dst(bx, split_y, bw, h_bot)


def _split_horizontally(text_regions, bubble_rects, i, j, bx, by, bw, bh, cx_i, cx_j):
    """Split a shared bubble between two regions by X axis."""
    split_x = int((cx_i + cx_j) / 2)
    split_x = max(bx + 10, min(split_x, bx + bw - 10))

    if cx_i <= cx_j:
        w_left = split_x - bx
        w_right = bx + bw - split_x
        bubble_rects[i] = _rect_to_dst(bx, by, w_left, bh)
        bubble_rects[j] = _rect_to_dst(split_x, by, w_right, bh)
    else:
        w_left = split_x - bx
        w_right = bx + bw - split_x
        bubble_rects[j] = _rect_to_dst(bx, by, w_left, bh)
        bubble_rects[i] = _rect_to_dst(split_x, by, w_right, bh)


def _rect_to_dst(x, y, w, h):
    """Pack (x, y, w, h) into a ``(1, 4, 2)`` int64 dst_points array."""
    return np.array(
        [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]],
        dtype=np.int64,
    )
