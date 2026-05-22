# QUICK REFERENCE: Code Changes Summary

## File: snipshot_engine/rendering/text_render.py

### Change 1: Improve Line Spacing (Line 608, 573)
**BEFORE:**
```python
spacing_y = int(font_size * (line_spacing or 0.01))
spacing_x = int(font_size * (line_spacing or 0.2))
```

**AFTER:**
```python
spacing_y = int(font_size * (line_spacing if line_spacing else 0.15))
spacing_x = int(font_size * (line_spacing if line_spacing else 0.10))
spacing_y = max(spacing_y, 3)  # Minimum 3px even for tiny fonts
spacing_x = max(spacing_x, 2)  # Minimum 2px
```

---

### Change 2: Smart Word Syllabification (Around Line 396)
**ADD NEW HELPER FUNCTION** (before `calc_horizontal`):
```python
def _syllabify_word(word: str, hyphenator=None) -> List[str]:
    """Smart syllabification that avoids character-level breaking."""
    if len(word) <= 3:
        return [word]

    # Try hyphenator for long words
    if hyphenator and len(word) >= 10:
        try:
            return hyphenator.syllables(word)
        except:
            pass

    # For 4-9 chars: group by consonant clusters (avoid single chars)
    if 4 <= len(word) <= 9:
        result = []
        current = ""
        vowels = "aeiouAEIOU"
        for i, c in enumerate(word):
            current += c
            if c in vowels and (i + 1 < len(word) and word[i+1] not in vowels):
                result.append(current)
                current = ""
        if current:
            result.append(current)
        if len(result) <= 3:  # Only use if breaks cleanly
            return result

    # Fallback: character groups of 2-3
    if len(word) > 15:
        chunk_size = 3
        return [word[i:i+chunk_size] for i in range(0, len(word), chunk_size)]

    return [word]
```

**REPLACE LINE 396:**
```python
# OLD:
# syls = [word] if len(word) <= 3 else list(word)

# NEW:
syls = _syllabify_word(word, hyphenator=hyphenator if hyphenate else None)
```

---

## File: snipshot_engine/rendering/__init__.py

### Change 3: Redesigned Font Sizing (Replace Lines 253-280)
**REPLACE ENTIRE FUNCTION:**
```python
def _find_optimal_font_size(text, box_w, box_h, initial_fs, lang, min_fs=10):
    """Binary-search for optimal font size with adaptive constraints."""

    # Adaptive margin (5-15% of smallest dimension)
    margin = max(8, int(min(box_w, box_h) * 0.08))
    safe_w = max(margin * 2, box_w - margin * 2)
    safe_h = max(margin * 2, box_h - margin * 2)

    # Dynamic max_fs based on bubble size class
    bubble_area = box_w * box_h
    if bubble_area < 5000:      # Tiny
        max_fs = min(int(safe_h / 1.8), 28)
    elif bubble_area < 15000:   # Small
        max_fs = min(int(safe_h / 2.5), 36)
    elif bubble_area < 50000:   # Medium
        max_fs = min(int(safe_h / 3.5), 52)
    else:                        # Large
        max_fs = min(int(safe_h / 4), 72)

    max_fs = min(max_fs, initial_fs * 2.5)  # Don't exceed 2.5x original
    max_fs = max(max_fs, min_fs + 4)        # Ensure room to grow

    lo, hi = min_fs, max_fs
    best = min_fs

    for _ in range(14):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        if mid < 1:
            break

        lines, widths = text_render.calc_horizontal(mid, text, safe_w, safe_h, lang)

        # Calculate total height with NEW line spacing
        bg_size = int(max(mid * 0.07, 1))
        line_spacing_px = int(mid * 0.15)  # 15% of font size
        total_h = mid * len(lines) + line_spacing_px * max(0, len(lines) - 1) + margin * 2

        if total_h <= box_h and max(widths) + margin * 2 <= box_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best
```

---

### Change 4: Better Text Centering (Lines 206-237)
**ADD NEW HELPER FUNCTION** (before `_render_region`):
```python
def _center_text_in_box(temp_box: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Center text in box with proper padding on all sides."""
    h, w, c = temp_box.shape

    top_margin = max(0, (target_h - h) // 2)
    left_margin = max(0, (target_w - w) // 2)

    output = np.zeros((target_h, target_w, c), dtype=np.uint8)
    y_end = min(top_margin + h, target_h)
    x_end = min(left_margin + w, target_w)

    if y_end > top_margin and x_end > left_margin:
        output[top_margin:y_end, left_margin:x_end] = \
            temp_box[:y_end - top_margin, :x_end - left_margin]

    return output
```

**REPLACE LINES 206-237 IN `_render_region` WITH:**
```python
# OLD: Complex if/else logic for aspect ratio padding

# NEW: Simple centered padding
margin = max(8, int(min(norm_h[0], norm_v[0]) * 0.08))
box_w = int(max(norm_h[0] - margin * 2, 20))
box_h = int(max(norm_v[0] - margin * 2, 20))

box = _center_text_in_box(temp_box, box_w, box_h)
```

---

## File: snipshot_engine/rendering/bubble.py

### Change 5: Adaptive Padding (Line 22-24)
**ADD HELPER FUNCTION:**
```python
def _calculate_adaptive_padding(bubble_area: int) -> int:
    """Adaptive padding based on bubble size."""
    if bubble_area < 5000:
        base_padding = 6
    elif bubble_area < 15000:
        base_padding = 10
    elif bubble_area < 50000:
        base_padding = 12
    else:
        base_padding = 16
    return base_padding
```

**MODIFY `detect_bubbles()` SIGNATURE & BODY:**
```python
# OLD:
def detect_bubbles(
    inpainted_img: np.ndarray,
    text_regions: List[TextBlock],
    min_bubble_area: int = 800,
    max_bubble_ratio: float = 0.3,
    padding: int = 12,  # ← REMOVE or make None
) -> List[Optional[np.ndarray]]:

# NEW:
def detect_bubbles(
    inpainted_img: np.ndarray,
    text_regions: List[TextBlock],
    min_bubble_area: int = 800,
    max_bubble_ratio: float = 0.3,
    padding: Optional[int] = None,  # ← Make optional
) -> List[Optional[np.ndarray]]:

    # ... existing code ...

    for idx, region in enumerate(text_regions):
        # Determine padding for this region
        if padding is None:
            # Will calculate after flood fill
            region_padding = None
        else:
            region_padding = padding

        rect = _detect_single_bubble(
            gray, cx, cy, h, w, img_area, region,
            min_area, max_ratio, region_padding or 12,
        )

        # Recalculate with adaptive padding if not provided
        if rect is None and padding is None:
            bubble_area = ...  # Get from flood fill
            adaptive_padding = _calculate_adaptive_padding(bubble_area)
            rect = _detect_single_bubble(
                gray, cx, cy, h, w, img_area, region,
                min_area, max_ratio, adaptive_padding,
            )
```

---

# Difficulty Levels

| Change | File | Effort | Impact | Risk |
|--------|------|--------|--------|------|
| 1. Line Spacing | text_render.py | 🟢 5min | 🟠 Medium | 🟢 None |
| 2. Smart Syllabify | text_render.py | 🟢 30min | 🔴 High | 🟢 Low |
| 3. Font Sizing | __init__.py | 🟠 90min | 🔴 High | 🟡 Medium |
| 4. Text Centering | __init__.py | 🟠 60min | 🟠 Medium | 🟡 Medium |
| 5. Adaptive Padding | bubble.py | 🟡 45min | 🟠 Medium | 🟢 Low |

---

# Testing Checklist

After implementing each change:
- [ ] Small bubble (50x60px): Text readable at 24-28px
- [ ] Large bubble (300x300px): Text at 48-64px, filling space
- [ ] Multi-line (3+ lines): Readable spacing, not cramped
- [ ] Long word (20+ chars): Hyphenation natural
- [ ] Different aspect ratios: Centered vertically/horizontally
- [ ] Verify no regression on existing working bubbles

