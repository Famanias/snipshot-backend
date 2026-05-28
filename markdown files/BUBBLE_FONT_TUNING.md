# Bubble Font Size Tuning Guide

This guide explains how bubble text size is chosen and how to tune it safely.

## Where Bubble Font Size Is Calculated

Bubble font search happens in:

- `snipshot_engine/rendering/__init__.py`
- Function: `_find_optimal_font_size(...)`

The dispatch flow is:

1. `detect_bubbles(...)` finds bubble rectangles.
2. `_find_optimal_font_size(...)` binary-searches the largest size that still fits.
3. The result is assigned to `region.font_size` and rendered.

## Main Tuning Knobs

All knobs below are inside `_find_optimal_font_size(...)`.

### 1) `margin_ratio` and `margin`

What it does:

- Reserves inner padding inside the bubble before text fit checks.
- Bigger margin means smaller final font.

How to tune:

- To reduce oversized text: increase `margin_ratio` or increase complexity multiplier.
- To allow larger text: decrease them slightly.

Current logic also scales margin by text density (`complexity`), which helps complex pages.

### 2) `max_fs` formula

What it does:

- Sets the upper bound for binary search.
- Lower cap means safer/smaller output.

How to tune:

- If text is still too large: reduce multipliers/caps in the `min_dim` branches.
- If text looks too small: raise them gradually.

### 3) `growth_mult`

What it does:

- Limits how much bigger final font can be compared to the region's initial font.

How to tune:

- Smaller `growth_mult` => more conservative growth.
- Larger `growth_mult` => more aggressive enlargement.

### 4) `h_slack` / `w_slack`

What it does:

- Slight tolerance in fit checks (`<= box * slack`).
- Higher slack allows larger fonts to pass.

How to tune:

- Reduce slack to tighten fit and avoid oversized text.
- Increase slack only if text appears too constrained.

## Recommended Safe Ranges

Use small incremental changes and test a few representative pages.

- `growth_mult`: adjust by `0.1` to `0.3` per pass.
- slack values (`h_slack`, `w_slack`): adjust by `0.01` per pass.
- complexity multipliers: adjust by about `0.05` per pass.
- `max_fs` branch multipliers: adjust by about `0.02` per pass.

## Quick Recipes

### Make Bubble Text Smaller (complex pages)

1. Lower `growth_mult` by `0.2`.
2. Reduce each slack value by `0.01`.
3. Increase complexity impact (for example: `0.12 -> 0.16` in max size damping).
4. Re-run sample pages and verify readability.

### Make Bubble Text Slightly Larger

1. Raise `growth_mult` by `0.1`.
2. Increase each slack value by `0.01`.
3. Slightly relax `max_fs` multipliers.
4. Validate that text still stays inside bubble boundaries.

## Suggested Validation Loop

1. Run your sample diagnostics.
2. Check pages with dense bubbles and low-confidence bubble detection.
3. Confirm no clipping and no obvious oversized text.
4. Keep a small changelog of parameter edits so you can roll back quickly.

## Notes

- Keep changes minimal and isolated to `_find_optimal_font_size(...)` unless you are intentionally changing fallback behavior.
- Prefer multiple small tuning passes over one large jump.
