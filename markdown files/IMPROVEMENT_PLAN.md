# SnipShot Image Output Improvement Plan

## Executive Summary

This document outlines a strategic roadmap to improve image output quality with careful, validated approaches. Rather than making sweeping changes, we will focus on **incremental, measured improvements** with extensive testing at each stage.

**Key Principle:** Each improvement will be tested locally before deployment and compared against baseline output to ensure no regressions.

---

## Phase 0: Diagnostics & Metrics (Foundation)

Before making any code changes, establish objective baselines.

### 0.1 Create Visual Test Suite
- [ ] Collect 5-10 sample manga/manhwa pages representing different challenges:
  - Small text in tight bubbles
  - Large text in wide bubbles
  - Vertical text (manga style)
  - Horizontal text with CJK characters
  - Mixed orientation (horizontal + vertical)
  - High-density text pages
  - Wide spacing pages

### 0.2 Establish Comparison Framework
- [ ] Create a test harness that:
  - Saves before/after pairs for side-by-side visual review
  - Tracks rendering metrics: font size, line spacing, text overflow, centering offset
  - Logs any errors or fallbacks during rendering
  - Allows easy regression detection

### 0.3 Document Current Behavior
- [ ] Run baseline on all test images
- [ ] For each image, document:
  - Font size ranges observed
  - Line spacing measurements
  - Overflow handling approach (scaling, truncation)
  - Centering accuracy
  - Any visual artifacts (overlaps, distortion, gaps)

---

## Phase 1: Analysis & Root Cause Investigation

### 1.1 Identify Current Limitations
Review the existing pipeline and document actual pain points:

- **Font Sizing**
  - Current approach: Uses `_find_optimal_font_size()` with binary search (14 iterations)
  - Observation: Font size formula uses `min(box_w, box_h)` for area calculation
  - Unknown: How often does it hit edge cases? Are there bubbles where sizing fails?

- **Line Spacing**
  - Current approach: `put_text_horizontal()` uses hardcoded 1% for line spacing
  - Observation: May be tight for multi-line Japanese/Korean text
  - Unknown: Is this actually causing readability issues or is it acceptable?

- **Text Centering**
  - Current approach: `_center_text_in_box()` centers rendered text within the target box
  - Observation: Uses simple centering with proportional shrinking on overflow
  - Unknown: Does this cause text to appear off-center due to bubble shape distortion?

- **Bubble Detection**
  - Current approach: Flood-fill with fixed 12px erosion padding
  - Observation: Padding is independent of bubble size
  - Unknown: Noise in detection? False negatives for irregular bubbles?

- **Translation-Induced Overflow**
  - Current approach: Resize regions based on character count comparison
  - Observation: Uses `_count_text_length()` with 0.5x for small Japanese characters
  - Unknown: Accurate for all languages? Issues with expansion ratios?

### 1.2 Hypothesis Testing Plan
For each identified limitation, propose a testable hypothesis:

| Component | Current Behavior | Hypothesis | Test |
|-----------|-----------------|-----------|------|
| Font sizing | Min dimension formula | Small bubbles get undersized | Compare font sizes for bubbles <50px vs >100px |
| Line spacing | 1% + hardcoded | Multi-line text overlaps | Measure actual spacing on CJK pages |
| Text centering | Simple center + shrink | Homography distortion causes off-center appearance | Check centering on rotated bubbles |
| Bubble detection | Fixed 12px erosion | Irregular shapes cause detection failures | Count false negatives per page |
| Language handling | Character count heuristic | Misses expansion for languages like Arabic or Thai | Test non-CJK languages |

### 1.3 Create Instrumentation
- [ ] Add logging to measure:
  - Font size before/after scaling
  - Text overflow percentage
  - Bubble detection success rate
  - Line break distribution
  - Centering offsets

---

## Phase 2: Incremental, Low-Risk Improvements

### 2.1 Fix Obvious Bugs (if any)
- [ ] Review error logs and fix clear bugs that don't require design changes

### 2.2 Improve Line Spacing (Safe)
**Hypothesis:** Line spacing is too tight on multi-line text.  
**Approach:** Increase line spacing *slightly* and validate with visual tests.

```python
# Current: 1% of max height, minimum 1 px
# Proposed: 2-3% of max height, minimum 1 px (conservative increase)

line_spacing_ratio = 0.02  # 2% instead of 1%
```

**Testing:**
- Run on 5 test pages
- Measure line overlap visually
- If acceptable: Keep change, else revert

**Risk:** Very low — only affects spacing, not positioning

### 2.3 Improve Bubble Detection Padding (Measured)
**Hypothesis:** Fixed 12px padding doesn't scale with bubble size.  
**Approach:** Make padding adaptive but with safe bounds.

```python
# Current: fixed padding = 12
# Proposed: padding = min(12, bubble_width * 0.08, bubble_height * 0.08)

# This keeps padding smaller for tiny bubbles, same for large bubbles
```

**Testing:**
- Compare bubble detection success rate before/after
- Check for text displacement
- Verify no text cutoff

**Risk:** Low — only affects bubble boundary, not text rendering

### 2.4 Font Size Smoothing (Conservative)
**Hypothesis:** Current font sizing could be optimized as languages change.  
**Approach:** Only if analysis shows clear undersizing pattern.

- Do NOT change the binary search mechanism (it works)
- Only if needed: Fine-tune the safe margin calculation

**Testing:**
- Compare rendered font sizes for known-good cases
- Ensure no regressions on existing test images

**Risk:** Medium — font sizing is critical

---

## Phase 3: High-Confidence Improvements (After Phase 0-2 Analysis)

### 3.1 Language-Aware Features
Only implement after confirming specific language issues:

- [ ] Detect language via existing `region.target_lang`
- [ ] Adjust parameters per language:
  - **CJK (Chinese, Japanese, Korean):** Wider spacing, fixed-width assumption
  - **Arabic/Farsi:** RTL awareness, compression factor
  - **Thai:** Word-breaking rules, ligature handling

### 3.2 Better Overflow Handling
Only after confirming current overflow is inadequate:

- [ ] Use actual rendered text dimensions instead of character count
- [ ] Proportional scaling instead of binary expansion

### 3.3 Improved Text Centering
Only if homography distortion is confirmed problematic:

- [ ] Measure actual centering offset on rotated bubbles
- [ ] Apply correction if offset > acceptable threshold (e.g., 3px)

---

## Phase 4: Advanced Features (Future)

These should only be considered after all previous issues are resolved:

- [ ] Shape-aware placement (use contours instead of bounding box)
- [ ] Kerning support (adjust character spacing)
- [ ] Optical alignment (visual weight centering, not geometric)
- [ ] Font weight/emphasis preservation from original OCR

---

## Testing & Validation Strategy

### Before Each Change
1. Create a baseline output for all test images
2. Measure metrics (spacing, font size, overflow)
3. Document expected visual change

### After Each Change
1. Run on same test images
2. Compare metrics before/after
3. Visual review (side-by-side)
4. Check for regressions

### Rollback Criteria
- [ ] Text overlaps introduced
- [ ] Text cut off from bubble
- [ ] Font size significantly smaller than baseline
- [ ] Line spacing too tight (visual confirmation)
- [ ] Bubble detection failures increase
- [ ] Any visual artifact not present before

### Success Criteria
- [ ] Improvements visible on 80%+ of test cases
- [ ] No regressions on existing working cases
- [ ] Metrics show measurable improvement

---

## Proposed Investigation Questions

Before implementing changes, answer these questions:

1. **Font Sizing**
   - How many bubbles have font sizes < 16px?
   - How many have font sizes > 80px?
   - Do small bubbles look hard to read?

2. **Line Spacing**
   - On multi-line translations, is spacing visually acceptable?
   - Are any characters overlapping?
   - Is spacing adequate for readability?

3. **Bubble Detection**
   - What percentage of bubbles are correctly detected?
   - How many fail due to irregular shape vs. erosion padding?
   - Are detection misses causing text rendering problems?

4. **Text Overflow**
   - How often does text overflow occur?
   - When it does, is the scaling approach acceptable?
   - Are any languages consistently causing overflow?

5. **Centering**
   - On rotated bubbles (angle > 10°), is text centered visually?
   - Are there any left/right alignment issues?
   - Do asymmetric bubble shapes cause centering problems?

---

## Implementation Roadmap

### Week 1: Phase 0 (Diagnostics)
- [ ] Collect test images and establish baseline
- [ ] Create comparison framework
- [ ] Document metrics for all components

### Week 2: Phase 1 (Analysis)
- [ ] Answer investigation questions via testing
- [ ] Identify which components actually have problems
- [ ] Document root causes (not assumptions)

### Week 3: Phase 2 (Safe Improvements)
- [ ] Implement low-risk changes (line spacing, bubble padding)
- [ ] Validate each change independently
- [ ] Keep detailed before/after logs

### Week 4: Phase 3+ (High-Confidence)
- [ ] Based on Phase 1 findings, implement targeted improvements
- [ ] Skip improvements that testing shows aren't needed
- [ ] Focus on actual pain points, not theoretical ones

---

## Key Principles

1. **Measure Before Changing:** Understand current behavior objectively
2. **Test Incrementally:** Change one thing at a time
3. **Revert Quickly:** If a change makes things worse, revert immediately
4. **Document Everything:** Keep records of what worked and what didn't
5. **Skip Untested Assumptions:** Don't implement "improvements" for problems that don't exist
6. **Prioritize Stability:** A working 80% solution is better than a broken 95% attempt

---

## What NOT to Do

- ❌ Make sweeping changes to font sizing without measurement
- ❌ Change line spacing globally without testing on CJK text
- ❌ Modify bubble detection without validation on irregular shapes
- ❌ Assume a fix works without side-by-side visual comparison
- ❌ Import new dependencies without explicit need
- ❌ Implement Phase 3+ features before Phase 0-1 analysis

---

## Next Steps

1. **Immediate:** Run Phase 0 diagnostics on representative test images
2. **Short-term:** Investigate which of the 5 components actually has issues
3. **Medium-term:** Implement only improvements supported by Phase 1 findings
4. **Long-term:** Build on validated knowledge for Phase 3+ features

---

## Questions to Discuss

1. What specific visual problems are you seeing in the current output?
2. Do you have example images that show the issues?
3. Which aspects are most important to fix? (font size, spacing, centering, overflow, detection)
4. Are there particular languages or text styles that cause problems?
5. What would success look like visually?

