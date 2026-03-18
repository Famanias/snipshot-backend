# SnipShot Rendering Placement Technical Report

## Objective

This document analyzes why translated text placement is inaccurate even when inpainting quality is good.

The focus is the rendering stack in `snipshot_engine/rendering`.

## Evidence Base

Primary data source:
- `sample_pages/diagnostics_baseline.md`
- `sample_pages/diagnostics_baseline.json`

Observed metrics (latest baseline run):
- Bubble detection rate by complexity:
  - complex: 2.7%
  - medium: 34.8%
  - easy: 100.0%
- Small-font incidence (`<16px`) in complex set: 13.5%
- One difficult sample frequently errors (`test-image-complex3.png`) during broader diagnostics runs.

Interpretation:
- In hard pages, renderer often cannot get a clean bubble target region.
- When bubble targeting fails, fallback placement geometry dominates and causes visible size/position drift.

---

## Rendering Pipeline Overview

Entry point:
- `snipshot_engine/rendering/__init__.py` -> `dispatch(...)`

High-level sequence:
1. Detect bubble rectangle per region (`detect_bubbles`)
2. If bubble exists: compute optimal font size for that bubble
3. If bubble missing: expand original textline rectangle heuristically
4. Render RGBA text patch
5. Warp patch into destination quadrilateral via homography

### Code Path Snippet: Dispatch Flow

```python
# snipshot_engine/rendering/__init__.py
bubble_rects = detect_bubbles(img, text_regions)

for i, (region, bubble_rect) in enumerate(zip(text_regions, bubble_rects)):
    if bubble_rect is not None:
        bw = int(bubble_rect[0, 1, 0] - bubble_rect[0, 0, 0])
        bh = int(bubble_rect[0, 2, 1] - bubble_rect[0, 0, 1])
        optimal_fs = _find_optimal_font_size(...)
        region.font_size = optimal_fs
        dst_points_list[i] = bubble_rect
    else:
        non_bubble_indices.append(i)
        non_bubble_regions.append(region)

if non_bubble_regions:
    fallback = _resize_regions_to_font_size(...)
```

Why it matters:
- A missed bubble immediately changes both target geometry and font sizing policy.

---

## Root Cause Analysis

## 1) Bubble Detection Misses on Complex/Medium Pages

Files:
- `snipshot_engine/rendering/bubble.py`
- `snipshot_engine/rendering/__init__.py`

Diagnostic evidence:
- Very low bubble detection on complex pages (2.7%).

Technical behavior:
- Detection starts from text center and flood-fills with fixed intensity tolerance.
- Multiple hard filters are applied: area bounds, compactness, center-preserving erosion, min width/height.
- Any failure returns `None`, forcing fallback to textline box geometry.

### Code Snippet: Strict Bubble Gate Stack

```python
# snipshot_engine/rendering/bubble.py
cv2.floodFill(..., loDiff=(35,), upDiff=(35,), ...)

if flood_area < min_area:
    return None
if flood_area > img_area * max_ratio:
    return None
if flood_area < tw * th * 1.2:
    return None

if hull_area > 0 and cv2.contourArea(cnt) / hull_area < 0.4:
    return None

if bw < 30 or bh < 30:
    return None
if region.horizontal and bw < tw * 0.8:
    return None
```

Failure mode:
- On textured, grayscale, or irregular panels, this can reject valid bubble-like regions.

Resulting user-visible issue:
- Text appears misplaced or constrained because fallback target is smaller/offset relative to erased zone.

---

## 2) Font Sizing in Bubble Path Is Horizontal-Oriented

File:
- `snipshot_engine/rendering/__init__.py`

Technical behavior:
- `_find_optimal_font_size(...)` always uses horizontal layout fitting (`calc_horizontal`) for its binary search fit check.
- Vertical region cases are not explicitly modeled in this function.

### Code Snippet: Horizontal Fit Assumption

```python
# snipshot_engine/rendering/__init__.py
lines, widths = text_render.calc_horizontal(mid, text, safe_w, safe_h, lang)

line_spacing_px = max(int(mid * 0.15), 3)
total_h = mid * len(lines) + line_spacing_px * max(0, len(lines) - 1) + margin * 2

if total_h <= box_h and max_line_w + margin * 2 <= box_w:
    best = mid
```

Failure mode:
- For vertical text or forced vertical direction, computed fit can diverge from true rendered footprint.

Resulting user-visible issue:
- Inconsistent font size selection: too large in narrow vertical shapes or too small after conservative fitting.

---

## 3) Non-Bubble Fallback Scaling Is Capped Aggressively

File:
- `snipshot_engine/rendering/__init__.py`

Technical behavior:
- When bubble detection fails, region expansion uses anisotropic overflow estimates.
- Final expansion is hard-clamped to 1.15x in each axis.

### Code Snippet: Hard Expansion Cap

```python
# snipshot_engine/rendering/__init__.py
over_x, over_y = _estimate_overflow_scales(region, target_fs)
final_scale_x = max(1.0, min(max(base_scale, over_x), 1.15))
final_scale_y = max(1.0, min(max(base_scale, over_y), 1.15))
```

Failure mode:
- For long translations in small original boxes, 1.15x is often insufficient.

Resulting user-visible issue:
- Text appears too small (because font cannot grow enough) or cramped in fallback regions.

---

## 4) Margin + Spacing + Shrink Pipeline Can Compound Conservative Layout

Files:
- `snipshot_engine/rendering/__init__.py`
- `snipshot_engine/rendering/text_render.py`

Technical behavior:
- Inner margin is text-aware but can be substantial on small boxes.
- Horizontal line spacing defaults to 15%; vertical to 10%.
- If rendered patch does not fit inner box, `_center_text_in_box` shrinks proportionally.

### Code Snippet: Margin and Shrink

```python
# snipshot_engine/rendering/__init__.py
margin = _compute_inner_margin(target_w, target_h, len(region.get_translation_for_rendering()))
inner_w = max(1, target_w - margin * 2)
inner_h = max(1, target_h - margin * 2)
centered_inner = _center_text_in_box(temp_box, inner_w, inner_h)
```

```python
# snipshot_engine/rendering/__init__.py
if h > target_h or w > target_w:
    sx = target_w / max(w, 1)
    sy = target_h / max(h, 1)
    scale = max(0.1, min(sx, sy))
    content = cv2.resize(temp_box, (new_w, new_h), interpolation=interp)
```

```python
# snipshot_engine/rendering/text_render.py
line_spacing_ratio = line_spacing if line_spacing else 0.15
spacing_y = max(int(font_size * line_spacing_ratio), 3)
```

Failure mode:
- On constrained targets, inner-box reduction and shrink activate frequently, reducing effective text size.

Resulting user-visible issue:
- Text can look smaller than expected even if initial font size appears reasonable.

---

## 5) Shared-Bubble Split Strategy Is Geometry-Simple

File:
- `snipshot_engine/rendering/bubble.py`

Technical behavior:
- If two regions overlap the same bubble rectangle (IoU >= 0.5), split is always vertical by center Y.

### Code Snippet: Vertical Split Assumption

```python
# snipshot_engine/rendering/bubble.py
if _rect_iou(r1, r2) < 0.5:
    continue

# Shared bubble -> split vertically between the two regions.
_split_vertically(text_regions, bubble_rects, i, j)
```

Failure mode:
- Diagonal side-by-side text blocks or non-vertical composition can be split incorrectly.

Resulting user-visible issue:
- Correct text appears in wrong sub-area of the bubble.

---

## 6) Perspective Warp Depends Entirely on Destination Quad Quality

File:
- `snipshot_engine/rendering/__init__.py`

Technical behavior:
- Final placement uses homography from text patch rectangle to destination quadrilateral.
- If destination quadrilateral is off, text appears shifted/skewed.

### Code Snippet: Homography Placement

```python
# snipshot_engine/rendering/__init__.py
M, _ = cv2.findHomography(src_pts, dst_points, cv2.RANSAC, 5.0)
rgba = cv2.warpPerspective(box, M, (img.shape[1], img.shape[0]), ...)
```

Failure mode:
- Any earlier geometry error propagates directly into final placement.

Resulting user-visible issue:
- Text looks "not in the right place" despite clean inpaint.

---

## Priority of Fixes (Data-Driven, Low Risk)

This section does not apply broad redesign. It orders small changes by likely impact and risk.

1. Bubble-detection reliability first
- Rationale: biggest measured gap between easy and complex pages.
- Validation metric: bubble detection rate by complexity group.

2. Vertical-aware font fit check in `_find_optimal_font_size`
- Rationale: direct sizing correctness for non-horizontal layout.
- Validation metric: reduction in outlier font sizes on vertical regions.

3. Controlled fallback cap relaxation (only in overflow-confirmed cases)
- Rationale: 1.15x cap likely too strict for missed-bubble scenarios.
- Validation metric: lower small-font incidence without new clipping.

4. Margin/spacing tuning only after 1-3
- Rationale: otherwise masks root geometry issues.
- Validation metric: readability gains with no clipping regressions.

---

## Recommended Validation Gates

For each single change, rerun sample diagnostics and compare against baseline.

Required non-regression gates:
- Easy pages keep near-current quality.
- No increase in processing failures.

Improvement gates:
- Bubble detection rate (complex, medium) increases.
- Small-font ratio (`<16px`) decreases on complex pages.
- Manual visual spot-check on at least 3 complex pages confirms improved placement.

---

## Summary

When inpainting looks correct but text placement is wrong, the dominant problem is usually target geometry selection, not glyph drawing.

In this codebase, the highest-leverage fault line is:
- Bubble detection miss -> fallback geometry -> constrained/shifted placement.

Secondary contributors are:
- Horizontal-oriented fit logic in bubble font sizing,
- Strict fallback scale cap,
- conservative margin/spacing plus shrink.

These are all in `snipshot_engine/rendering`, which matches your suspicion.
