"""Sort text regions in reading order (with optional panel detection)."""

from typing import List, Tuple

import cv2
import numpy as np

from .textblock import TextBlock


def sort_regions(
    regions: List[TextBlock],
    right_to_left: bool = True,
    img: np.ndarray = None,
    force_simple_sort: bool = False,
) -> List[TextBlock]:
    if not regions:
        return []

    if force_simple_sort:
        return _simple_sort(regions, right_to_left)

    # Try panel-based sorting when image is available
    if img is not None:
        try:
            from manga_translator.utils.panel import get_panels_from_array

            panels_raw = get_panels_from_array(img, rtl=right_to_left)
            panels = [(x, y, x + w, y + h) for x, y, w, h in panels_raw]
            panels = _sort_panels_fill(panels, right_to_left)

            for r in regions:
                cx, cy = r.center
                r.panel_index = -1
                for idx, (x1, y1, x2, y2) in enumerate(panels):
                    if x1 <= cx <= x2 and y1 <= cy <= y2:
                        r.panel_index = idx
                        break
                if r.panel_index < 0:
                    dists = [
                        ((max(x1 - cx, 0, cx - x2)) ** 2 + (max(y1 - cy, 0, cy - y2)) ** 2, i)
                        for i, (x1, y1, x2, y2) in enumerate(panels)
                    ]
                    if dists:
                        r.panel_index = min(dists)[1]

            grouped: dict = {}
            for r in regions:
                grouped.setdefault(r.panel_index, []).append(r)

            sorted_all: List[TextBlock] = []
            for pi in sorted(grouped.keys()):
                sorted_all += sort_regions(grouped[pi], right_to_left, img=None)
            return sorted_all

        except Exception:
            return _simple_sort(regions, right_to_left)

    # Coordinate-based sorting when no image is provided
    xs = [r.center[0] for r in regions]
    ys = [r.center[1] for r in regions]

    if len(regions) > 1:
        x_std = float(np.std(xs))
        y_std = float(np.std(ys))
        is_horizontal = x_std > y_std
    else:
        is_horizontal = False

    sorted_regions: List[TextBlock] = []
    if is_horizontal:
        primary = sorted(regions, key=lambda r: -r.center[0] if right_to_left else r.center[0])
        group: List[TextBlock] = []
        prev = None
        for r in primary:
            cx = r.center[0]
            if prev is not None and abs(cx - prev) > 20:
                group.sort(key=lambda r: r.center[1])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cx
        if group:
            group.sort(key=lambda r: r.center[1])
            sorted_regions += group
    else:
        primary = sorted(regions, key=lambda r: r.center[1])
        group = []
        prev = None
        for r in primary:
            cy = r.center[1]
            if prev is not None and abs(cy - prev) > 15:
                group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
                sorted_regions += group
                group = []
            group.append(r)
            prev = cy
        if group:
            group.sort(key=lambda r: -r.center[0] if right_to_left else r.center[0])
            sorted_regions += group

    return sorted_regions


def _simple_sort(regions: List[TextBlock], right_to_left: bool) -> List[TextBlock]:
    sorted_regions: List[TextBlock] = []
    for region in sorted(regions, key=lambda r: r.center[1]):
        for i, sr in enumerate(sorted_regions):
            if region.center[1] > sr.xyxy[3]:
                continue
            if region.center[1] < sr.xyxy[1]:
                sorted_regions.insert(i, region)
                break
            if right_to_left and region.center[0] > sr.center[0]:
                sorted_regions.insert(i, region)
                break
            if not right_to_left and region.center[0] < sr.center[0]:
                sorted_regions.insert(i, region)
                break
        else:
            sorted_regions.append(region)
    return sorted_regions


def _sort_panels_fill(
    panels: List[Tuple[int, int, int, int]], right_to_left: bool
) -> List[Tuple[int, int, int, int]]:
    if not panels:
        return panels

    remaining = sorted(list(panels), key=lambda p: p[1])
    ordered: List[Tuple[int, int, int, int]] = []

    avg_w = float(np.mean([p[2] - p[0] for p in remaining]))
    avg_h = float(np.mean([p[3] - p[1] for p in remaining]))
    y_thr = max(10, avg_h * 0.3)

    while remaining:
        base_y = remaining[0][1]
        row: list = []
        i = 0
        while i < len(remaining):
            if abs(remaining[i][1] - base_y) <= y_thr:
                row.append(remaining.pop(i))
            else:
                i += 1
        row.sort(key=lambda p: (-p[0] if right_to_left else p[0]))
        ordered.extend(row)

    return ordered
