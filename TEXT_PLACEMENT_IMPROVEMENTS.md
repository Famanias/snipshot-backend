# TEXT PLACEMENT QUALITY ANALYSIS

## Executive Summary
The current snipshot_engine has **5 critical weaknesses** in text placement:
1. Font sizing algorithm too conservative for small bubbles, under-utilizes large bubbles
2. Line wrapping breaks at single characters, creating unnatural hyphens
3. Line spacing too tight (1% instead of 15-20% of font size)
4. Vertical centering fails due to complex homography + rigid padding
5. No adaptive padding based on bubble size/shape

---

## Issue #1: Suboptimal Font Sizing

### Current Logic (rendering/__init__.py:253-280)
```python
def _find_optimal_font_size(text, box_w, box_h, initial_fs, lang, min_fs=10):
    # PROBLEM: These bounds are too restrictive
    max_fs = min(int(box_w / 2), int(box_h * 0.4), initial_fs * 3, 80)
    # For 60x60 bubble:  min(30, 24, ?, 80) = 24px MAX
    # For 400x300 bubble: min(200, 120, ?, 80) = 80px MAX
```

### Why This Fails
- **For small bubbles (60x60)**: `box_h * 0.4 = 24px` is already optimal, but then horiz text needs even more lines
- **For large bubbles (400x300)**: Caps out at 80px even though we could use 120-150px
- **Ignores aspect ratio**: Tall thin bubble vs wide short bubble get same constraints
- **No margin accommodation**: Doesn't reserve space for padding (typically 15-20% of bubble interior)

### Observed in Images
- Small circular "Huh?" bubble: Text at ~14px (too small, low readability)
- Large oval bubbles: Text at ~20-24px (should be 40-50px)
- Cramped lines in mid-size bubbles

---

## Issue #2: Poor Line Breaking (calc_horizontal Line 396)

### Current Logic
```python
syllables = []
for word in words:
    syls = []
    if hyphenator and len(word) <= 100:
        syls = hyphenator.syllables(word)  # Works for long words
    if not syls:
        syls = [word] if len(word) <= 3 else list(word)  # ← PROBLEM
        #      ^-- "it" stays as "it"
        #                              ^-- "fine" becomes ["f","i","n","e"]
```

### Why This Fails
- Words 1-3 chars: Kept whole (`"it"` → `["it"]`) ✓
- Words 4+ chars: Split to single chars (`"fine"` → `["f","i","n","e"]`) ✗
- English text rendering: Char-by-char looks unnatural, hyphenation looks broken

### Observed in Images
- "W-w-ell I'm f-fin-e..." in bottom-left bubble
- "He's using honor-if-ics..." - unnecessary character-level breaks
- Should be: "Well I'm fine..." (minimal breaks) or "He's using honor-ifics..." (word-level groups)

---

## Issue #3: Inadequate Line Spacing

### Current Logic (text_render.py:608)
```python
spacing_y = int(font_size * (line_spacing or 0.01))
# Default 0.01 = 1% of font size
# For 20px font: spacing_y = 0.2px (rounded to 0) ← INVISIBLE
# For 50px font: spacing_y = 0.5px (rounded to 0) ← INVISIBLE
```

### Why This Fails
- Professional typesetting: 120-150% line-height (20-50% actual spacing)
- Current: 1% is vanishingly small
- Multi-line text in same bubble: Lines visually merge, harder to read

### Observed in Images
- "He im-mediately noticed when I changed my hair-style..." text is cramped
- Vertical compression makes speech flow harder to parse

---

## Issue #4: Text Centering Failures

### Problem Cascade
1. `calc_horizontal()` returns text lines, but no centering info
2. `put_text_horizontal()` creates canvas larger than needed
3. `_render_region()` tries to fix via aspect-ratio padding (lines 206-237)
4. Homography warping distorts the result

### Current Logic in _render_region (206-237)
```python
if render_h:
    if r_temp > r_orig:  # Text too wide
        h_ext = int((w / r_orig - h) // 2)
        if h_ext >= 0:
            box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)
            box[h_ext:h_ext + h, 0:w] = temp_box  # ← Centers vertically
    else:  # Text too tall
        w_ext = int((h * r_orig - w) // 2)
        box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)
        box[0:h, 0:w] = temp_box  # ← Text stays left-aligned!
```

### Why This Fails
- When text is too tall (needs tall narrow bubble), text is placed at **top-left**, not centered
- Padding calculation is mathematically sound but centering placement is inconsistent
- Homography warping then distorts this, losing any centering benefit

### Observed in Images
- Text in tall narrow bubbles: Clustered at top
- Large oval bubbles: Text not vertically centered

---

## Issue #5: No Adaptive Padding

### Current Approach
1. **bubble.py**: Fixed padding of 12px for all bubbles (line 23: `padding: int = 12`)
2. **text_render.py**: Fixed border size ~7% of font size (line 572, 608)
3. **Result**: 12px padding on 60x60 bubble = 20% of height (TOO MUCH)
          12px padding on 300x300 bubble = 4% of height (TOO LITTLE)

### Observed Issues
- Small bubbles: Text squeezed into tiny area despite detector working
- Large bubbles: Lots of wasted space, text too small
- Safe area calculation doesn't account for line spacing either

---

---

# RECOMMENDED IMPROVEMENTS (Ranked by Impact)

## TOP PRIORITY

### 1. Redesign Font Sizing Algorithm (HIGH IMPACT, MEDIUM EFFORT)

**Replace static bounds with dynamic calculation:**

```python
def _find_optimal_font_size_v2(text, box_w, box_h, initial_fs, lang, min_fs=10):
    """
    NEW: Adaptive font sizing that considers:
    - Bubble aspect ratio (tall vs wide)
    - Available margin for padding
    - Text length (longer text needs smaller font)
    - Minimum readability threshold
    """

    # Calculate adaptive safe margin (5-15% of smallest dimension)
    margin = max(8, int(min(box_w, box_h) * 0.08))
    safe_w = max(margin * 2, box_w - margin * 2)
    safe_h = max(margin * 2, box_h - margin * 2)

    # For horizontal text, be more conservative with height
    # For vertical text, be more conservative with width
    # Estimate: roughly 1-3 lines for typical bubble
    text_lines = max(1, len(text) // (safe_w // 15))  # Rough estimate: 15px per char

    # Adaptive max_fs based on bubble size class
    bubble_area = box_w * box_h
    if bubble_area < 5000:      # Tiny (e.g., 50x100)
        max_fs = min(int(safe_h / 1.8), 28)  # 1-2 lines max
    elif bubble_area < 15000:   # Small (e.g., 100x150)
        max_fs = min(int(safe_h / 2.5), 32)  # 2-3 lines
    elif bubble_area < 50000:   # Medium (150x300)
        max_fs = min(int(safe_h / 3.5), 48)  # 3-4 lines
    else:                        # Large (300x300+)
        max_fs = min(int(safe_h / 4), 64)   # 4+ lines

    max_fs = min(max_fs, initial_fs * 2.5)  # Don't exceed 2.5x original
    max_fs = max(max_fs, min_fs + 4)        # Ensure room for growth

    # Binary search: find largest font that fits with lines + margin
    lo, hi = min_fs, max_fs
    best = min_fs

    for _ in range(14):  # More iterations for precision
        if lo > hi:
            break
        mid = (lo + hi) // 2
        if mid < 1:
            break

        lines, widths = text_render.calc_horizontal(mid, text, safe_w, safe_h, lang)

        # Account for padding & line spacing in height
        bg_size = int(max(mid * 0.07, 1))
        line_spacing_px = int(mid * 0.15)  # 15% of font size (NEW)
        total_h = mid * len(lines) + line_spacing_px * (len(lines) - 1) + margin * 2

        # Check fit with explicit margin buffer
        if total_h <= box_h and max(widths) <= safe_w:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best
```

**Benefits:**
- ✅ Small bubbles: 28-32px (readable), not 14px
- ✅ Large bubbles: 48-64px (fills space), not capped at 80px
- ✅ Considers available safe area, not just raw dimensions
- ✅ Adapts to text length (long text → smaller font)

---

### 2. Smart Line Breaking (HIGH IMPACT, LOW EFFORT)

**Fix calc_horizontal() syllable fallback:**

```python
def _syllabify_word(word, hyphenator=None, max_width_px=None, font_size=None):
    """
    NEW: Smarter word breaking for English
    - Uses hyphenation for long words (10+ chars)
    - Keeps short words intact (1-3 chars)
    - For mid-length words (4-9), prefer smart grouping over char splitting
    """
    if len(word) <= 3:
        return [word]  # Keep intact

    # Try hyphenator for long words
    if hyphenator and len(word) >= 10 and len(word) <= 100:
        try:
            return hyphenator.syllables(word)
        except:
            pass

    # For 4-9 char words: group by consonant clusters
    if len(word) <= 9:
        # Simple heuristic: break after vowels when possible
        # "medium" → ["me", "di", "um"] instead of ["m","e","d","i","u","m"]
        result = []
        current = ""
        vowels = "aeiouAEIOU"
        for i, c in enumerate(word):
            current += c
            # Break after vowel if next is consonant (or end of word)
            if c in vowels and (i + 1 < len(word) and word[i+1] not in vowels):
                result.append(current)
                current = ""
        if current:
            result.append(current)
        if result and len(result) <= 3:  # Only use if breaks cleanly
            return result

    # Fallback for very long words: character level, but in groups of 2-3
    if len(word) > 15:
        chunk_size = 3
        return [word[i:i+chunk_size] for i in range(0, len(word), chunk_size)]

    # Everything else: keep whole word
    return [word]
```

**In calc_horizontal() around line 396:**
```python
# OLD:
#   syls = [word] if len(word) <= 3 else list(word)

# NEW:
    syls = _syllabify_word(word, hyphenator=hyphenator)
```

**Benefits:**
- ✅ No more "f-i-n-e" → "fi-ne" or just "[fine]"
- ✅ Proper "medium" → "me-di-um" instead of "m-e-d-i-u-m"
- ✅ Fallback to character level only for truly long words
- ✅ Looks more professional

---

### 3. Fix Line Spacing (HIGH IMPACT, LOW EFFORT)

**Change default line spacing:**

**text_render.py line 608, 573:**
```python
# OLD:
# spacing_y = int(font_size * (line_spacing or 0.01))
# spacing_x = int(font_size * (line_spacing or 0.2))

# NEW: Smarter defaults
line_spacing_ratio_h = line_spacing if line_spacing else 0.15  # 15% for horizontal
line_spacing_ratio_v = line_spacing if line_spacing else 0.10  # 10% for vertical

spacing_y = int(font_size * line_spacing_ratio_h)
spacing_x = int(font_size * line_spacing_ratio_v)

# Minimum absolute values
spacing_y = max(spacing_y, 3)  # At least 3px even for tiny fonts
spacing_x = max(spacing_x, 2)
```

**Benefits:**
- ✅ 15% line-height matches professional typesetting (150-160% total line-height)
- ✅ 3px minimum ensures visible separation even for small fonts
- ✅ Multi-line text is now readable, not cramped
- ✅ Backward compatible: caller can still override with explicit `line_spacing` param

---

## MEDIUM PRIORITY

### 4. Improve Vertical Text Centering (MEDIUM IMPACT, MEDIUM EFFORT)

**Replace complex padding logic in _render_region():**

```python
def _center_text_in_box(temp_box, target_w, target_h):
    """
    NEW: Center text in box cleanly (replacing lines 206-237)
    - temp_box: rendered text (h, w, 4)
    - target_w, target_h: desired output size
    Returns: padded box with text centered
    """
    h, w, c = temp_box.shape

    # Calculate centered margins
    top_margin = (target_h - h) // 2
    left_margin = (target_w - w) // 2

    # Ensure non-negative
    top_margin = max(0, top_margin)
    left_margin = max(0, left_margin)

    # Create output box
    output = np.zeros((target_h, target_w, c), dtype=np.uint8)

    # Place centered
    y_end = min(top_margin + h, target_h)
    x_end = min(left_margin + w, target_w)
    output[top_margin:y_end, left_margin:x_end] = \
        temp_box[:y_end - top_margin, :x_end - left_margin]

    return output
```

**In _render_region(), replace old logic (line 208-237):**
```python
# OLD: Complex if/else for aspect ratio matching
# NEW:
box = _center_text_in_box(
    temp_box,
    int(norm_h[0] - margin),  # target_w with margin
    int(norm_v[0] - margin)   # target_h with margin
)
```

**Benefits:**
- ✅ Text always centered, both horizontal & vertical
- ✅ Much clearer logic, easier to maintain
- ✅ Works for all bubble orientations
- ✅ No distortion from homography on unbalanced padding

---

### 5. Adaptive Padding (MEDIUM IMPACT, MEDIUM EFFORT)

**In bubble.py, make padding adaptive:**

```python
def _calculate_adaptive_padding(bubble_area, text_length):
    """
    NEW: Padding scales with bubble size
    - Tiny bubbles: less padding (already constrained)
    - Large bubbles: more padding for breathing room
    - Accounts for text length: longer text needs tighter padding
    """
    if bubble_area < 5000:      # Tiny
        base_padding = 6
    elif bubble_area < 15000:   # Small
        base_padding = 10
    elif bubble_area < 50000:   # Medium
        base_padding = 12
    else:                        # Large
        base_padding = 16

    # Adjust for text length
    if text_length < 20:        # Few words
        padding = base_padding * 1.2
    elif text_length > 100:     # Many words
        padding = base_padding * 0.8
    else:
        padding = base_padding

    return int(padding)
```

**In detect_bubbles() call:**
```python
# OLD:
padding = 12

# NEW:
padding = _calculate_adaptive_padding(
    int(np.sum(bubble_mask > 0)),  # bubble_area
    sum(len(r.text) for r in text_regions) // len(text_regions)  # avg text length
)
```

**Benefits:**
- ✅ Small bubbles get less padding squeezing, more usable space
- ✅ Large bubbles get ample margins
- ✅ Text-aware: dense paragraphs get tighter, sparse text gets breathing room

---

## BONUS: Advanced Ideas

### 6. Aspect-Ratio Smart Resizing (LOW IMPACT, HIGH EFFORT)
**Idea**: If text overflows, slightly expand bubble in the longer axis instead of shrinking font

```python
# In _resize_regions_to_font_size(), after calculating needed_rows/cols:
if needed_rows > used_rows:
    # Instead of just scaling x: also try expanding bubble slightly
    scale_x = min(((needed_rows - used_rows) / used_rows) + 1, 1.15)  # Max 15% expansion
    # Expand bubble if it's already wide enough
```

### 7. Shape-Aware Text Placement (VERY HIGH EFFORT, NICHE)
**Idea**: For non-rectangular bubbles (diamonds, complex shapes), calculate actual safe text area via contour analysis instead of bounding box

```python
# Would require:
# 1. Non-rectangular bubble detection (in bubble.py)
# 2. Contour-based masking for text placement
# 3. Dynamic break points for multi-line text inside irregular shapes
# Impact: Only helpful for ~5-10% of bubbles with complex shapes
```

### 8. Kerning & Ligature Support (NICHE, MEDIUM EFFORT)
**Idea**: Use FreeType's kerning tables for more professional text spacing
- Currently: Each char placed independently, no kerning
- Improvement: `kerning_x = face.get_kerning(prev_char, cur_char)`

---

---

# Implementation Priority Roadmap

**Phase 1 (Immediate - HIGH REWARD):**
1. **Smart line breaking** (30 min) - Fixes obvious hyphenation issues
2. **Improve font sizing** (1-2 hrs) - Biggest visual impact
3. **Fix line spacing** (15 min) - Easy, high readability gain

**Phase 2 (Medium - POLISH):**
4. **Vertical centering** (1 hr) - Improves uniformity
5. **Adaptive padding** (1 hr) - More sophisticated bubble usage

**Phase 3 (Polish - TIME PERMITTING):**
6. **Aspect-ratio resizing** - Subtle improvements for edge cases

---

# Testing Strategy

After each change, test against:
- **Small bubbles** (< 100px²): Font size readable, not cramped
- **Large bubbles** (> 50,000px²): Font fills space, not tiny
- **Multi-line text** (3-5 lines): Spacing readable, not cramped
- **Single-line text**: Centered, good margins
- **Long words** (20+ chars): Hyphenation natural, not character-level
- **Aspect extremes**: Very tall narrow vs very wide short bubbles

**Recommendation**: A/B compare before/after on `test-image.png` and complex test images.

