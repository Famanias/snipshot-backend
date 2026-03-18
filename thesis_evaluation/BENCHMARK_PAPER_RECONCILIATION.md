# Benchmark–Paper Reconciliation & Validation Report

**Date:** March 19, 2026  
**Paper:** "SnipShot: A Deep Learning-Based Screen-Snipping Translation Tool for Digital Comics" (ICITE 2026 Submission)  
**Benchmark:** `BENCHMARK_RESULTS_9IMAGES.md` (9 images × 3 runs, February 8, 2026)  
**Resolution Benchmark:** `resolution_benchmark_20260208_071153.json`

---

## Overview

This document cross-references the numerical claims in the published research paper against the raw benchmark data produced by the automated evaluation pipeline. The paper is treated as the authoritative **source of truth** for final reported results. Where discrepancies exist, hypotheses are provided and a methodology for resolution is proposed.

---

## Source of Truth

The **ICITE 2026 submission PDF** is the authoritative reference for all reported metrics. The benchmark JSON and markdown outputs are supporting artifacts that should corroborate—not override—the paper's claims. When values diverge, the paper's values take precedence and the benchmark data must be re-examined for methodological differences.

---

## Identified Discrepancies

### D1 — Overall Mean BLEU Score

| Source | Value |
|--------|:-----:|
| Paper (Table 3) | **0.6290** |
| Benchmark (9-image) | **0.6139** |
| **Δ** | **−0.0151** |

The paper's Table 3 reports BLEU computed over 3 images (15.jpg, 155.jpg, 1555555.jpg) with 12 total samples. The benchmark uses 9 different images with 79 total translation samples across 27 runs. These are **different test sets**, so direct comparison is expected to diverge.

### D2 — Test Image Set & Sample Counts

| Attribute | Paper | Benchmark |
|-----------|:-----:|:---------:|
| Images tested | 3 | 9 |
| Image names | 15.jpg, 155.jpg, 1555555.jpg | 13.jpg, 14.jpg, 15.jpg, 3 medium, 3 complex |
| Total translation samples | 12 | 79 |
| Runs per image | 3 (implied) | 3 |

Only **15.jpg** appears in both sets, and even then the paper reports 6 translated samples for it (BLEU 0.8107) vs. the benchmark's 1 translation per run (mean BLEU 0.3363). This indicates either different ground truth reference counts or a different version of the image/ground truth file.

### D3 — OCR Accuracy

| Source | OCR Accuracy | CER |
|--------|:------------:|:---:|
| Paper (Abstract) | **93%** | — |
| Paper (Table 2) | **88.98%** | **1.02%** |
| Paper (IPO diagram) | **93%** (with per-language: JPN 94.2%, CHS 92.8%, KOR 91.5%) |   |
| Benchmark (9-image) | **100.00%** | **0.00%** |

Three different OCR accuracy values appear across the paper itself:
- **93%** in the abstract and IPO model refers to a broader evaluation across 100 manga panels (40 JPN, 30 CHS, 30 KOR) with independently verified ground truth.
- **88.98%** in Table 2 is the mean OCR confidence score, *mislabeled* as "Mean OCR Accuracy" — this value exactly matches the benchmark's mean OCR confidence (88.85% ≈ 88.98%, difference attributable to the different image set).
- **100%** in the benchmark is an artifact of self-referential ground truth (detected texts were used as their own ground truth), not a true accuracy measurement.

### D4 — Mean OCR Confidence

| Source | Value |
|--------|:-----:|
| Paper (Table 2) | **85.92%** |
| Benchmark (9-image) | **88.85%** |
| **Δ** | **+2.93%** |

Different test image sets with different text complexity distributions produce different confidence means.

### D5 — Processing Time

| Metric | Paper (Table 1) | Benchmark (9-image overall) |
|--------|:---------:|:----------:|
| Mean Total | **4,980 ms** | **60,592 ms** |
| Detection | 691 ms (13.9%) | 592 ms (0.98%) |
| OCR | 580 ms (11.6%) | 1,049 ms (1.73%) |
| Translation | 1,200 ms (24.1%) | 42,510 ms (70.16%) |
| Inpainting | 703 ms (14.1%) | 955 ms (1.58%) |
| Rendering | 1,805 ms (36.3%) | 13,908 ms (22.95%) |

This is the largest numerical discrepancy. The paper reports ~5s total; the benchmark reports ~60.6s. The paper's timing data appears to come from **easy-category images only** (3 simple Korean panels with 1–2 translations each, mean ~4.7s in the benchmark). The 9-image benchmark includes complex images with up to 27 translations that take 100–223s — heavily shifting the overall mean upward.

The paper's assertion of "3.8-second average processing time for 1080p images" (in the abstract) aligns most closely with the easy-category mean (4,748 ms ≈ 4.7s) or the resolution benchmark's 1920×1080 detection-only time (285 ms total, but no text was found at that resolution).

### D6 — Processing Time Distribution

| Stage | Paper % | Benchmark % |
|-------|:-------:|:-----------:|
| Rendering | **36.3%** | 22.95% |
| Translation | **24.1%** | 70.16% |
| Inpainting | **14.1%** | 1.58% |
| Detection | **13.9%** | 0.98% |
| OCR | **11.6%** | 1.73% |

The paper shows a balanced time distribution; the benchmark is dominated by Translation (70%). This is consistent with D5 — the paper's timing data originates from simple images where translation is fast (~500ms for 1–2 texts), making other stages proportionally larger. For complex images with 10–27 translations, the Groq API call time dominates.

### D7 — Multi-Resolution Benchmark

| Resolution | Paper Total (ms) | Benchmark Total (ms) |
|:----------:|:----------------:|:--------------------:|
| 640×480 | 2,504.61 | 6,467.57 |
| 800×600 | 560.12 | 19,485.08 |
| 1024×768 | 650.41 | 12,468.99 |
| 1280×720 | 474.45 | 623.00 |
| 1920×1080 | 400.61 | 285.54 |
| 2560×1440 | 506.16 | 358.67 |

| Resolution | Paper Detection (ms) | Benchmark Detection (ms) |
|:----------:|:--------------------:|:------------------------:|
| 640×480 | 1,477.42 | 3,697.36 |
| 800×600 | 290.72 | 354.51 |
| 1024×768 | 353.11 | 354.87 |
| 1280×720 | 287.56 | 622.76 |
| 1920×1080 | 302.68 | 285.31 |
| 2560×1440 | 417.88 | 358.39 |

The paper and benchmark used **different source images**. The paper used the original 3-image test set; the benchmark used `test-image-medium.png` (1022×1850 portrait). The portrait-to-landscape distortion in the benchmark caused detection to fail at 1280×720, 1920×1080, and 2560×1440 (0 texts found), while 800×600 and 1024×768 found partial text, triggering full pipeline execution (translation + rendering) and inflating total times. An earlier resolution benchmark (`resolution_benchmark_20260205_195446.json`) with different source images shows values closer to the paper.

### D8 — Abstract vs. Body Inconsistencies (Internal to Paper)

| Metric | Abstract | Body |
|--------|:--------:|:----:|
| OCR Accuracy | 93% | 88.98% (Table 2) |
| Processing Time | 3.8s | 4,980 ms / ~5s (Table 1) |
| PEU | 4.6/5 | 3.50/5 (Table 6) |
| PU | 4.1/5 | 3.92/5 (Table 5) |

The abstract reports different values from the body tables. The 93% figure likely comes from a separate multi-language evaluation (100 panels). The 3.8s vs. 5s discrepancy may reflect a single-resolution (1080p) measurement vs. the mean across all resolution tests. The TAM scores in the abstract (PEU 4.6, PU 4.1) do not match the body tables (PEU 3.50, PU 3.92), suggesting the abstract may reference an earlier evaluation round or different scoring criteria.

---

## Analysis of Differences

### Different Test Sets (D1, D2, D5, D6)
The paper's technical evaluation used 3 simple Korean manga images (15.jpg, 155.jpg, 1555555.jpg) with 12 total translation samples. The benchmark used 9 images spanning easy/medium/complex categories with 79 samples. Different datasets inherently produce different aggregate metrics. The paper's simpler test set yields faster times and different BLEU distributions.

### Self-Referential Ground Truth (D3)
The benchmark's 100% OCR accuracy is methodologically invalid as a true accuracy measure — the ground truth was auto-generated from the system's own detection, producing a tautological result. The paper's 93% figure from 100 human-verified panels is the more meaningful metric. The 88.98% in Table 2 appears to be the OCR **confidence** score (model's self-assessed certainty), not character-level accuracy.

### API Latency Variability (D5, D7)
Translation times via the Groq API are highly variable depending on: (a) number of text regions requiring translation, (b) API load/queue at request time, (c) rate limiting and retry delays. The paper's 1,200ms translation time is consistent with 1–2 short text translations; the benchmark's 42,510ms mean reflects images with up to 27 translations including rate-limit retries.

### Resolution Benchmark Source Image (D7)
Using a portrait manga for landscape resolution targets introduces severe aspect-ratio distortion. The paper likely used a landscape or square source image, producing valid text detection at all resolutions.

---

## Corroboration Methodology

### Step 1: Reproduce the Paper's Original Test Set

```bash
# Run with the paper's original 3 images and ground truth
python -m thesis_evaluation.run_thesis_evaluation \
  --images 15.jpg 155.jpg 1555555.jpg \
  --ground-truth thesis_evaluation/ground_truth.json \
  --runs 3 --output-dir benchmark_results
```

Verify that: the output produces 12 total translation samples, per-image BLEU matches Table 3 (0.8107, 0.5045, 0.3902), and overall mean = 0.6290.

### Step 2: Validate OCR Accuracy with Independent Ground Truth

1. Manually transcribe text from 10+ images (covering JPN, CHS, KOR) to create **human-verified** ground truth independent of the system.
2. Run the OCR pipeline and compute character-level accuracy against this reference.
3. Compare against the paper's claimed 93% (abstract) and 88.98% (Table 2).
4. Clarify whether Table 2's "Mean OCR Accuracy" is character accuracy or confidence score.

### Step 3: Isolate Processing Time for 1080p

```bash
# Single-resolution test matching the abstract's "3.8s for 1080p" claim
python -m thesis_evaluation.run_thesis_evaluation \
  --images <simple_test_image.jpg> \
  --resolution-test --resolutions 1920x1080 \
  --runs 3 --output-dir benchmark_results
```

Use a source image with aspect ratio close to 16:9 or with few text regions to match the paper's conditions.

### Step 4: Re-run Resolution Benchmark with Aspect-Preserving Resize

Modify `run_resolution_benchmark` to preserve the source image aspect ratio (e.g., resize by scaling factor, not target dimensions) or use a source image whose native ratio matches the target resolutions.

### Step 5: Cross-Check Paper's Internal Consistency

| Check | Expected |
|---|---|
| Abstract OCR = Body OCR? | Should reconcile 93% vs. 88.98% — document which evaluation each refers to |
| Abstract time = Body time? | 3.8s vs. 5.0s — specify which resolution or image subset |
| Abstract TAM = Body TAM? | PEU 4.6→3.50, PU 4.1→3.92 — verify survey data and scoring method |

---

## Recommendations

1. **Align abstract with body.** Update the abstract to match the tables in the body, or explicitly note when the abstract references a different evaluation scope (e.g., "93% across 100 multi-language panels" vs. "88.98% on the 9-image benchmark set").

2. **Separate OCR confidence from OCR accuracy.** Table 2 should clearly distinguish between character-level accuracy (Levenshtein-based) and model confidence (softmax output). Currently "Mean OCR Accuracy" at 88.98% appears to be the confidence score, not accuracy.

3. **Report per-category BLEU.** The 9-image benchmark shows category BLEU of Easy=0.41, Medium=0.71, Complex=0.72. Reporting only the 3-image BLEU (0.629) may underrepresent translation quality on realistic complex content.

4. **Specify test conditions for timing claims.** State explicitly which images and resolution were used for the "3.8s" claim. The 9-image mean (60.6s) is dramatically different and would mislead if readers assume it applies to complex images.

5. **Use independent ground truth for OCR accuracy.** Replace the self-referential ground truth (which produces 100% accuracy trivially) with human-verified transcriptions to produce a meaningful accuracy metric.

6. **Fix the resolution benchmark methodology.** Either (a) use source images with matching aspect ratios for each target resolution, or (b) resize while preserving aspect ratio (with padding/cropping) to avoid distortion artifacts.

7. **Document the BLEU computation method.** Specify tokenization strategy (word-level for English, character-level for CJK), smoothing method, and n-gram order (BLEU-4) so results are reproducible. Note: the IPO diagram reports "BLEU-4: 28.7" which appears to be a scaled BLEU (0–100); the body reports 0.6290 on a 0–1 scale. Standardize to one scale.



#	Discrepancy	Paper	Benchmark	Root Cause
D1	Overall BLEU	0.6290	0.6139	Different test sets (3 vs 9 images)
D3	OCR Accuracy	93% / 88.98%	100%	Self-referential ground truth in benchmark; Table 2 likely reports confidence not accuracy
D5	Mean Total Time	4,980 ms	60,592 ms	Paper used simple images; benchmark includes complex (27-translation) images
D8	Abstract vs Body	PEU 4.6, PU 4.1	PEU 3.50, PU 3.92	Internal paper inconsistency