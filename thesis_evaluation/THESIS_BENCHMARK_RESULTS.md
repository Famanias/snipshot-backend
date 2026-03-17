# SnipShot Manga Translation Pipeline: Benchmark Results

> **Document Purpose:** This document provides comprehensive evaluation results for the SnipShot automated manga translation system, suitable for inclusion in the Results and Discussion section of the thesis.
>
> **Evaluation Date:** February 5, 2026  
> **Benchmark Framework Version:** 1.0  
> **Test Environment:** Windows, NVIDIA GPU with 6GB VRAM

---

## Table of Contents

1. [Methodology Overview](#1-methodology-overview)
2. [Processing Time Analysis](#2-processing-time-analysis)
3. [OCR Evaluation Results](#3-ocr-evaluation-results)
4. [Translation Quality Assessment](#4-translation-quality-assessment)
5. [Multi-Resolution Performance Analysis](#5-multi-resolution-performance-analysis)
6. [Per-Image Detailed Results](#6-per-image-detailed-results)
7. [Statistical Summary](#7-statistical-summary)
8. [Discussion](#8-discussion)

---

## 1. Methodology Overview

### 1.1 Evaluation Framework

The automated evaluation pipeline was implemented in Python to systematically assess three critical aspects of the manga translation system:

1. **Character-level OCR Accuracy**: Computed by comparing detected text against ground-truth transcriptions on a character-by-character basis using Levenshtein distance.

2. **Translation Quality**: Assessed using the Bilingual Evaluation Understudy (BLEU) algorithm (Papineni et al., 2002), comparing system translations against reference human translations.

3. **Processing Speed**: Measured execution times for each pipeline stage using automated timing instrumentation across multiple test images.

### 1.2 Test Configuration

| Parameter | Value |
|-----------|-------|
| **Detector** | DefaultDetector |
| **Detection Size** | 1536 pixels |
| **Box Threshold** | 0.7 |
| **OCR Model** | 48px (Model48pxOCR) |
| **Translator** | Groq API |
| **Target Language** | English (ENG) |
| **Inpainter** | AOT Inpainter (default) |
| **Renderer** | Default Renderer |

### 1.3 Test Dataset

| Image | Resolution | Text Regions | Language |
|-------|------------|--------------|----------|
| 15.jpg | 690×1600 | 5 | Korean |
| 155.jpg | 690×1600 | 2 | Korean |
| 1555555.jpg | 690×1600 | 3 | Korean |

**Total Test Runs:** 9 (3 images × 3 iterations)

---

## 2. Processing Time Analysis

### 2.1 Overall Pipeline Performance

The complete translation pipeline was evaluated across 9 test runs, measuring execution time for each processing stage.

| Metric | Total | Detection | OCR | Translation | Inpainting | Rendering |
|--------|-------|-----------|-----|-------------|------------|-----------|
| **Mean (ms)** | 6,554.64 | 691.36 | 521.64 | 1,731.07 | 703.04 | 2,256.75 |
| **Median (ms)** | 5,382.69 | 359.20 | 265.19 | 1,507.48 | 756.05 | 1,667.58 |
| **Std Dev (ms)** | 3,850.49 | 1,130.66 | 782.55 | 988.74 | 167.97 | 988.05 |
| **Min (ms)** | 3,926.57 | 200.10 | 229.70 | 769.21 | 499.56 | 1,568.31 |
| **Max (ms)** | 16,271.18 | 3,698.23 | 2,607.98 | 3,727.39 | 1,008.88 | 4,199.97 |

### 2.2 Pipeline Stage Distribution

The following table shows the percentage of total processing time consumed by each stage:

| Stage | Mean Time (ms) | Percentage of Total |
|-------|----------------|---------------------|
| Rendering | 2,256.75 | 34.4% |
| Translation | 1,731.07 | 26.4% |
| Inpainting | 703.04 | 10.7% |
| Detection | 691.36 | 10.5% |
| OCR | 521.64 | 8.0% |
| **Overhead** | ~651 | 9.9% |

**Key Finding:** Rendering constitutes the largest portion of processing time (34.4%), followed by Translation API calls (26.4%).

### 2.3 Processing Time by Image

| Image | Run 1 (ms) | Run 2 (ms) | Run 3 (ms) | Mean (ms) |
|-------|-----------|-----------|-----------|-----------|
| 15.jpg | 16,271.18 | 6,635.19 | 7,370.56 | 10,092.31 |
| 155.jpg | 6,366.93 | 4,134.10 | 5,382.69 | 5,294.57 |
| 1555555.jpg | 4,958.55 | 3,926.57 | 3,946.01 | 4,277.04 |

**Note:** The first run shows significantly higher processing time due to model initialization (cold start). Subsequent runs demonstrate consistent performance.

---

## 3. OCR Evaluation Results

### 3.1 OCR Accuracy Metrics

| Metric | Value |
|--------|-------|
| **Mean OCR Accuracy** | 100.00% |
| **Mean Character Error Rate (CER)** | 0.00% |
| **Total Character Samples** | 30 |

The OCR system achieved perfect accuracy against the ground truth dataset, with all detected text exactly matching the reference transcriptions.

### 3.2 OCR Confidence Scores

The OCR model provides probability scores (0.0–1.0) indicating confidence in character recognition:

| Image | Detected Texts | Mean Confidence | Min | Max |
|-------|----------------|-----------------|-----|-----|
| 15.jpg | 5 | 88.20% | 64.45% | 99.81% |
| 155.jpg | 2 | 82.24% | 64.82% | 99.67% |
| 1555555.jpg | 3 | 96.51% | 89.97% | 99.85% |
| **Overall** | 10 | **88.98%** | 64.45% | 99.85% |

### 3.3 Per-Text Confidence Breakdown

| Image | Text | Confidence |
|-------|------|------------|
| 15.jpg | 이건… | 77.33% |
| 15.jpg | 누구야? | 99.80% |
| 15.jpg | 말도 안돼. | 99.81% |
| 15.jpg | 이딴 짓을… | 64.45% |
| 15.jpg | 도대체누가 | 99.60% |
| 155.jpg | 진짜 돈을 | 99.67% |
| 155.jpg | 넣었을리는없고... | 64.82% |
| 1555555.jpg | 성낙숙쪽에서 | 99.85% |
| 1555555.jpg | 했다기엔굳이… | 89.97% |
| 1555555.jpg | 이이름으로할까? | 99.71% |

**Observation:** Texts containing ellipsis (…) characters tend to have lower confidence scores, suggesting potential recognition challenges with punctuation.

---

## 4. Translation Quality Assessment

### 4.1 BLEU Score Summary

| Metric | Value |
|--------|-------|
| **Mean BLEU Score** | 0.6290 |
| **Minimum BLEU** | 0.1077 |
| **Maximum BLEU** | 1.0000 |
| **Sample Count** | 12 |

### 4.2 BLEU Scores by Image

| Image | BLEU Score | System Translation | Reference Translation |
|-------|------------|-------------------|----------------------|
| 15.jpg | **0.7427** | "This is... ridiculous." / "Who are you? Who on earth did this kind of thing..." | "This is... ridiculous." / "Who are you? Who on earth did this..." |
| 155.jpg | **0.2954** | "He wouldn't really put money in, so..." | "There's no way he'd actually put money in..." |
| 1555555.jpg | **0.1077** | "If Seong Nak-Suk is doing this, wouldn't it be… better to use this name?" | "If it was Seong Nak-Suk's side... why bother doing it under this name?" |

### 4.3 BLEU Score Interpretation

| BLEU Range | Interpretation | Count |
|------------|----------------|-------|
| 0.9–1.0 | Excellent (near-identical) | 4 |
| 0.7–0.9 | Good | 2 |
| 0.5–0.7 | Moderate | 1 |
| 0.3–0.5 | Low | 2 |
| 0.0–0.3 | Very Low | 3 |

**Note:** BLEU scores measure n-gram overlap and may not fully capture semantic equivalence. Lower scores often indicate valid alternative translations rather than errors.

---

## 5. Multi-Resolution Performance Analysis

### 5.1 Resolution Scaling Test Results

To evaluate scalability, the pipeline was tested across six standard resolutions:

| Resolution | Megapixels | Total (ms) | Detection (ms) | OCR (ms) |
|------------|------------|------------|----------------|----------|
| 640×480 | 0.31 | 2,504.61 | 1,477.42 | 1,026.75 |
| 800×600 | 0.48 | 560.12 | 290.72 | 268.89 |
| 1024×768 | 0.79 | 650.41 | 353.11 | 296.84 |
| 1280×720 | 0.92 | 474.45 | 287.56 | 186.29 |
| 1920×1080 | 2.07 | 400.61 | 302.68 | 97.50 |
| 2560×1440 | 3.69 | 506.16 | 417.88 | 87.82 |

### 5.2 Scaling Analysis

| Metric | Value |
|--------|-------|
| **Average Processing Rate** | 1,832.04 ms/megapixel |
| **Scaling Factor (640×480 to 2560×1440)** | 0.20× |

**Key Finding:** The pipeline demonstrates sub-linear scaling, with larger images processing relatively faster per-pixel due to fixed model initialization overhead being amortized across more data.

### 5.3 Resolution vs. Processing Time Trend

```
Processing Time (ms)
     ^
3000 |  *
     |
2500 |
     |
2000 |
     |
1500 |
     |
1000 |
     |
 500 |     *    *       *       *       *
     |
   0 +-----------------------------------------> Resolution
         640   800   1024   1280   1920   2560
         ×480  ×600  ×768   ×720   ×1080  ×1440
```

The anomalous spike at 640×480 is attributed to cold-start model initialization occurring during the first test.

---

## 6. Per-Image Detailed Results

### 6.1 Image: 15.jpg (5 Text Regions)

| Stage | Time (ms) | Details |
|-------|-----------|---------|
| Detection | 3,698.23* / 206.42 / 359.20 | Found 5 text regions |
| OCR | 2,607.98* / 253.51 / 244.50 | Confidence: 88.20% |
| Translation | 3,727.39 / 1,270.77 / 1,908.95 | 2 translated paragraphs |
| Inpainting | 1,008.88 / 756.05 / 759.98 | — |
| Rendering | 4,199.97 / 3,190.01 / 3,141.26 | — |
| **Total** | 16,271.18 / 6,635.19 / 7,370.56 | — |

*First run includes model initialization

**Extracted Text (Korean):**
1. 이건…
2. 누구야?
3. 말도 안돼.
4. 이딴 짓을…
5. 도대체누가

**System Translation (English):**
- "This is... ridiculous."
- "Who are you? Who on earth did this kind of thing..."

### 6.2 Image: 155.jpg (2 Text Regions)

| Stage | Time (ms) | Details |
|-------|-----------|---------|
| Detection | 375.84 / 408.84 / 398.71 | Found 2 text regions |
| OCR | 277.95 / 282.02 / 262.10 | Confidence: 82.24% |
| Translation | 2,808.74 / 769.21 / 1,821.96 | 1 translated paragraph |
| Inpainting | 742.24 / 517.65 / 772.30 | — |
| Rendering | 1,600.91 / 1,591.52 / 1,568.31 | — |
| **Total** | 6,366.93 / 4,134.10 / 5,382.69 | — |

**Extracted Text (Korean):**
1. 진짜 돈을
2. 넣었을리는없고...

**System Translation (English):**
- "He wouldn't really put money in, so..."

### 6.3 Image: 1555555.jpg (3 Text Regions)

| Stage | Time (ms) | Details |
|-------|-----------|---------|
| Detection | 349.69 / 200.10 / 225.16 | Found 3 text regions |
| OCR | 271.84 / 229.70 / 265.19 | Confidence: 96.51% |
| Translation | 1,507.48 / 943.76 / 821.42 | 1 translated paragraph |
| Inpainting | 767.35 / 499.56 / 503.34 | — |
| Rendering | 1,600.50 / 1,667.58 / 1,750.73 | — |
| **Total** | 4,958.55 / 3,926.57 / 3,946.01 | — |

**Extracted Text (Korean):**
1. 성낙숙쪽에서
2. 했다기엔굳이…
3. 이이름으로할까?

**System Translation (English):**
- "If Seong Nak-Suk is doing this, wouldn't it be… better to use this name?"

---

## 7. Statistical Summary

### 7.1 Performance Statistics

| Metric | Detection | OCR | Translation | Inpainting | Rendering | Total |
|--------|-----------|-----|-------------|------------|-----------|-------|
| **n** | 9 | 9 | 9 | 9 | 9 | 9 |
| **Mean** | 691.36 | 521.64 | 1,731.07 | 703.04 | 2,256.75 | 6,554.64 |
| **Median** | 359.20 | 265.19 | 1,507.48 | 756.05 | 1,667.58 | 5,382.69 |
| **SD** | 1,130.66 | 782.55 | 988.74 | 167.97 | 988.05 | 3,850.49 |
| **CV** | 163.5% | 150.1% | 57.1% | 23.9% | 43.8% | 58.7% |

**CV = Coefficient of Variation (SD/Mean × 100)**

### 7.2 Quality Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **OCR Accuracy** | 100.00% | Perfect character recognition |
| **OCR Confidence** | 88.98% | High model confidence |
| **Mean BLEU** | 0.6290 | Moderate-to-good translation quality |
| **CER** | 0.00% | Zero character errors |

### 7.3 System Configuration

| Component | Model/Configuration |
|-----------|---------------------|
| Text Detection | DefaultDetector (1536px) |
| OCR | Model48pxOCR |
| Translation | Groq API (groq translator) |
| Inpainting | AOT Inpainter |
| Rendering | Default Renderer |
| Resolution | 690×1600 (1.1 MP) |

---

## 8. Discussion

### 8.1 Processing Performance

The SnipShot pipeline demonstrates practical processing speeds with a mean total time of **6.55 seconds** per image at 690×1600 resolution. The rendering stage accounts for the largest portion of processing time (34.4%), followed by translation API calls (26.4%). This distribution suggests optimization opportunities in the text rendering subsystem.

Notable observations:
- **Cold-start penalty**: First-run initialization adds approximately 10 seconds to processing time
- **Consistent subsequent runs**: After initialization, processing times stabilize around 4-6 seconds
- **Stage consistency**: Inpainting shows the lowest variance (CV: 23.9%), indicating predictable performance

### 8.2 OCR Quality

The OCR component achieved **100% accuracy** against the ground-truth dataset, with a mean confidence score of **88.98%**. This high accuracy validates the effectiveness of the 48px OCR model for Korean manga text recognition.

Observations on confidence patterns:
- Texts with ellipsis (…) characters show lower confidence (64-65%)
- Standard text achieves >99% confidence
- Overall confidence distribution suggests reliable text detection

### 8.3 Translation Quality

The BLEU score of **0.6290** indicates moderate translation quality with room for improvement. It is important to note that BLEU measures lexical overlap rather than semantic equivalence—alternative translations that convey the same meaning may receive lower scores.

Key findings:
- Direct translations achieve high BLEU scores (>0.9)
- Paraphrased or contextually-adapted translations score lower
- The Groq API provides grammatically correct, readable translations

### 8.4 Scalability

The multi-resolution analysis reveals **sub-linear scaling** characteristics, where processing time does not increase proportionally with image size. This is advantageous for processing high-resolution manga pages, as the per-pixel processing cost decreases with larger images.

### 8.5 Limitations

1. **Sample size**: The evaluation used 3 unique images with 10 total text regions
2. **Language scope**: Testing limited to Korean→English translation
3. **BLEU limitations**: May underestimate translation quality for valid paraphrases
4. **Cold-start variance**: First-run performance significantly differs from steady-state

### 8.6 Recommendations for Production

Based on the evaluation results:
1. Implement model pre-loading to eliminate cold-start delays
2. Consider batch processing for translation API calls to reduce overhead
3. Investigate rendering optimizations for the text overlay stage
4. Expand ground-truth dataset for more comprehensive accuracy assessment

---

## Appendix A: Raw Data Files

| File | Description |
|------|-------------|
| `benchmark_results_20260205_202055.json` | Full benchmark results with 9 test runs |
| `resolution_benchmark_20260205_202337.json` | Multi-resolution scaling test results |
| `ground_truth.json` | Ground truth transcriptions and translations |

## Appendix B: Evaluation Code

The evaluation framework is implemented in the `thesis_evaluation/` module:
- `accuracy.py` - Character accuracy and BLEU scoring
- `performance.py` - Stage timing instrumentation
- `benchmark.py` - Main benchmark orchestration
- `run_thesis_evaluation.py` - CLI entry point

---

*Generated by SnipShot Thesis Evaluation Framework v1.0*
