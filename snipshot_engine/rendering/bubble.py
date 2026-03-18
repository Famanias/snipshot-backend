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

        rect = _detect_single_bubble(
            gray, cx, cy, h, w, img_area, region,
            min_bubble_area, max_bubble_ratio, padding,
        )
        if rect is not None:
            bw = int(rect[0, 1, 0] - rect[0, 0, 0])
            bh = int(rect[0, 2, 1] - rect[0, 0, 1])
            logger.debug("Region %d: BUBBLE %dx%d at (%d,%d)", idx, bw, bh,
                         int(rect[0, 0, 0]), int(rect[0, 0, 1]))
        else:
            logger.debug("Region %d: no bubble detected", idx)
        results.append(rect)

    # If multiple regions share the same bubble, split the space.
    _resolve_overlaps(text_regions, results)
    return results


# ── internals ────────────────────────────────────────────────────────


def _detect_single_bubble(
    gray, cx, cy, h, w, img_area, region,
    min_area, max_ratio, padding,
) -> Optional[np.ndarray]:
    """Flood-fill from the text center to find the enclosing bubble."""

    # ── 1. Flood fill ────────────────────────────────────────────────
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    gray_copy = gray.copy()
    cv2.floodFill(
        gray_copy, flood_mask, (cx, cy),
        newVal=0,
        loDiff=(35,), upDiff=(35,),
        flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8),
    )
    bubble_mask = flood_mask[1:-1, 1:-1]

    flood_area = int(np.sum(bubble_mask > 0))

    # ── 2. Validate area ─────────────────────────────────────────────
    if flood_area < min_area:
        return None
    if flood_area > img_area * max_ratio:
        return None  # leaked outside bubble

    # Bubble should be meaningfully larger than the textline bounding box.
    tx, ty, tw, th = cv2.boundingRect(region.min_rect[0].astype(np.int32))
    if flood_area < tw * th * 1.2:
        return None

    # Adaptive erosion padding: larger bubbles can tolerate larger padding,
    # while dense/long text gets tighter padding to preserve usable area.
    adaptive_padding = _adaptive_padding(flood_area, region, tw, th, padding)

    # Compactness check: real bubbles are roughly convex, not jagged/leaky.
    contours, _ = cv2.findContours(bubble_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if hull_area > 0 and cv2.contourArea(cnt) / hull_area < 0.4:
            return None  # highly non-convex → not a clean bubble

    # ── 3. Erode to get safe interior ────────────────────────────────
    eroded = _erode_mask(bubble_mask, cx, cy, adaptive_padding)
    if eroded is None:
        return None

    # ── 4. Bounding rect of eroded region at center ──────────────────
    num_labels, labels = cv2.connectedComponents(eroded)
    label_at_center = labels[cy, cx]
    if label_at_center == 0:
        return None

    component = (labels == label_at_center).astype(np.uint8) * 255
    bx, by, bw, bh = cv2.boundingRect(component)

    if bw < 30 or bh < 30:
        return None

    # For horizontal text the bubble should be at least as wide as the text.
    if region.horizontal and bw < tw * 0.8:
        return None

    return _rect_to_dst(bx, by, bw, bh)


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

            # Shared bubble → split vertically between the two regions.
            _split_vertically(text_regions, bubble_rects, i, j)


def _rect_iou(r1, r2):
    x1 = max(r1[0], r2[0])
    y1 = max(r1[1], r2[1])
    x2 = min(r1[0] + r1[2], r2[0] + r2[2])
    y2 = min(r1[1] + r1[3], r2[1] + r2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = r1[2] * r1[3] + r2[2] * r2[3] - inter
    return inter / union if union > 0 else 0.0


def _split_vertically(text_regions, bubble_rects, i, j):
    """Split a shared bubble between two regions based on vertical centre."""
    ri = bubble_rects[i][0]
    bx = int(ri[0, 0])
    by = int(ri[0, 1])
    bw = int(ri[1, 0] - bx)
    bh = int(ri[2, 1] - by)

    cy_i = float(text_regions[i].center[1])
    cy_j = float(text_regions[j].center[1])

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


def _rect_to_dst(x, y, w, h):
    """Pack (x, y, w, h) into a ``(1, 4, 2)`` int64 dst_points array."""
    return np.array(
        [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]],
        dtype=np.int64,
    )
