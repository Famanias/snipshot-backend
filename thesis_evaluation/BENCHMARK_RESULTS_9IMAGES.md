# SnipShot Benchmark Results — 9-Image Evaluation

**Date:** February 8, 2026  
**Configuration:** DBNet-ResNet34 (1536px) → Roformer+XPOS (48px) → Groq Llama-3-70b → LaMa Inpainting → Default Renderer  
**Runs per image:** 3  
**Total evaluations:** 27 (9 images × 3 runs)

---

## 1. Test Images Overview

| # | Image | Category | Resolution | Text Regions | Translations | Source Language |
|---|-------|----------|------------|:------------:|:------------:|----------------|
| 1 | 13.jpg | Easy | 690×1600 | 5 | 2 | Korean |
| 2 | 14.jpg | Easy | 690×1600 | 3 | 1 | Korean |
| 3 | 15.jpg | Easy | 690×1600 | 2 | 1 | Korean |
| 4 | test-image-medium.png | Medium | 1022×1850 | 18 | 5 | Japanese |
| 5 | test-image-medium1.png | Medium | 1235×909 | 19 | 9 | Japanese |
| 6 | test-image-medium2.png | Medium | 1068×1240 | 19 | 10 | Japanese |
| 7 | test-image-complex1.png | Complex | 1251×899 | 52 | 14 | Japanese |
| 8 | test-image-complex2.jpg | Complex | 3000×2250 | 24 | 10 | Japanese |
| 9 | test-image-complex3.png | Complex | 1600×1200 | 64 | 27 | Japanese |

---

## 2. BLEU Translation Quality Scores

### Table 1: Per-Image BLEU Scores (3 Runs)

| Image | Category | Translations | Run 1 | Run 2 | Run 3 | Mean | Std Dev |
|-------|----------|:------------:|:-----:|:-----:|:-----:|:----:|:-------:|
| 13.jpg | Easy | 2 | 0.6610 | 0.8195 | 0.8195 | **0.7667** | 0.0915 |
| 14.jpg | Easy | 1 | 0.1164 | 0.1703 | 0.0999 | **0.1289** | 0.0365 |
| 15.jpg | Easy | 1 | 0.4624 | 0.2954 | 0.2510 | **0.3363** | 0.1117 |
| test-image-medium.png | Medium | 5 | 0.7074 | 0.8952 | 0.6121 | **0.7382** | 0.1430 |
| test-image-medium1.png | Medium | 9 | 0.6510 | 0.6076 | 0.6860 | **0.6482** | 0.0393 |
| test-image-medium2.png | Medium | 10 | 0.7946 | 0.7176 | 0.7077 | **0.7400** | 0.0478 |
| test-image-complex1.png | Complex | 14 | 0.6332 | 0.6331 | 0.6001 | **0.6221** | 0.0191 |
| test-image-complex2.jpg | Complex | 10 | 0.7859 | 0.7709 | 0.7801 | **0.7790** | 0.0076 |
| test-image-complex3.png | Complex | 27 | 0.7451 | 0.7456 | 0.8063 | **0.7657** | 0.0353 |

### Table 2: BLEU Scores by Category

| Category | Images | Total Translations | Mean BLEU | Min BLEU | Max BLEU |
|----------|:------:|:------------------:|:---------:|:--------:|:--------:|
| Easy | 3 | 4 | **0.4106** | 0.0999 | 0.8195 |
| Medium | 3 | 24 | **0.7088** | 0.6076 | 0.8952 |
| Complex | 3 | 51 | **0.7223** | 0.6001 | 0.8063 |
| **Overall** | **9** | **79** | **0.6139** | 0.0999 | 0.8952 |

---

## 3. OCR Performance

### Table 3: OCR Confidence per Image

| Image | Category | Texts Detected | Mean OCR Confidence | OCR Accuracy | CER |
|-------|----------|:--------------:|:-------------------:|:------------:|:---:|
| 13.jpg | Easy | 5 | 88.20% | 100.00% | 0.00% |
| 14.jpg | Easy | 3 | 96.51% | 100.00% | 0.00% |
| 15.jpg | Easy | 2 | 82.24% | 100.00% | 0.00% |
| test-image-medium.png | Medium | 18 | 88.79% | 100.00% | 0.00% |
| test-image-medium1.png | Medium | 19 | 91.46% | 100.00% | 0.00% |
| test-image-medium2.png | Medium | 19 | 87.97% | 100.00% | 0.00% |
| test-image-complex1.png | Complex | 52 | 96.47% | 100.00% | 0.00% |
| test-image-complex2.jpg | Complex | 24 | 81.87% | 100.00% | 0.00% |
| test-image-complex3.png | Complex | 64 | 86.15% | 100.00% | 0.00% |

### Table 4: OCR Confidence by Category

| Category | Total Texts | Mean Confidence |
|----------|:-----------:|:---------------:|
| Easy | 10 | 88.98% |
| Medium | 56 | 89.41% |
| Complex | 140 | 88.16% |
| **Overall** | **206** | **88.85%** |

> **Note:** OCR accuracy is 100% because the ground truth was generated from the system's own initial detection, validating **deterministic consistency** of the OCR pipeline across runs.

---

## 4. Processing Time Analysis

### Table 5: Mean Processing Time per Image (ms, averaged over 3 runs)

| Image | Category | Detection | OCR | Translation | Inpainting | Rendering | **Total** |
|-------|----------|:---------:|:---:|:-----------:|:----------:|:---------:|:---------:|
| 13.jpg | Easy | 681.1 | 462.6 | 645.6 | 639.8 | 3,163.1 | **6,597.3** |
| 14.jpg | Easy | 414.2 | 252.8 | 395.3 | 565.4 | 1,589.3 | **3,602.8** |
| 15.jpg | Easy | 569.3 | 270.0 | 499.2 | 565.2 | 1,541.4 | **4,044.0** |
| test-image-medium.png | Medium | 520.5 | 929.5 | 1,891.6 | 1,229.9 | 8,674.1 | **15,437.2** |
| test-image-medium1.png | Medium | 511.6 | 644.0 | 32,999.0 | 732.0 | 13,845.0 | **49,567.5** |
| test-image-medium2.png | Medium | 563.8 | 766.4 | 52,010.1 | 895.3 | 15,633.5 | **70,495.2** |
| test-image-complex1.png | Complex | 591.5 | 1,275.1 | 79,708.1 | 769.8 | 21,603.6 | **105,261.9** |
| test-image-complex2.jpg | Complex | 850.6 | 2,001.4 | 41,405.1 | 1,950.8 | 15,927.8 | **66,888.9** |
| test-image-complex3.png | Complex | 628.5 | 2,839.6 | 173,036.9 | 1,243.6 | 43,189.6 | **223,432.0** |

### Table 6: Mean Processing Time by Category (ms)

| Category | Detection | OCR | Translation | Inpainting | Rendering | **Total** |
|----------|:---------:|:---:|:-----------:|:----------:|:---------:|:---------:|
| Easy | 554.9 | 328.5 | 513.4 | 590.1 | 2,097.9 | **4,748.0** |
| Medium | 532.0 | 780.0 | 28,966.9 | 952.4 | 12,717.5 | **45,166.6** |
| Complex | 690.2 | 2,038.7 | 98,050.0 | 1,321.4 | 26,907.0 | **131,860.9** |
| **Overall** | **592.3** | **1,049.1** | **42,510.1** | **954.6** | **13,907.5** | **60,591.9** |

### Table 7: Processing Time Distribution (% of total)

| Category | Detection | OCR | Translation | Inpainting | Rendering |
|----------|:---------:|:---:|:-----------:|:----------:|:---------:|
| Easy | 11.69% | 6.92% | 10.81% | 12.43% | 44.18% |
| Medium | 1.18% | 1.73% | 64.13% | 2.11% | 28.15% |
| Complex | 0.52% | 1.55% | 74.37% | 1.00% | 20.41% |
| **Overall** | **0.98%** | **1.73%** | **70.16%** | **1.58%** | **22.95%** |

---

## 5. Detailed Per-Run Results

### Table 8: All 27 Individual Run Results

| Run | Image | Category | Total (ms) | BLEU | OCR Conf. | Texts | Translations |
|:---:|-------|----------|:----------:|:----:|:---------:|:-----:|:------------:|
| 1 | 13.jpg | Easy | 8,141.1 | 0.6610 | 88.20% | 5 | 2 |
| 1 | 14.jpg | Easy | 3,514.0 | 0.1164 | 96.51% | 3 | 1 |
| 1 | 15.jpg | Easy | 3,949.7 | 0.4624 | 82.24% | 2 | 1 |
| 1 | test-image-medium.png | Medium | 15,232.9 | 0.7074 | 88.79% | 18 | 5 |
| 1 | test-image-medium1.png | Medium | 42,154.0 | 0.6510 | 91.46% | 19 | 9 |
| 1 | test-image-medium2.png | Medium | 69,481.2 | 0.7946 | 87.97% | 19 | 10 |
| 1 | test-image-complex1.png | Complex | 107,725.3 | 0.6332 | 96.47% | 52 | 14 |
| 1 | test-image-complex2.jpg | Complex | 65,914.5 | 0.7859 | 81.87% | 24 | 10 |
| 1 | test-image-complex3.png | Complex | 226,825.4 | 0.7451 | 86.15% | 64 | 27 |
| 2 | 13.jpg | Easy | 6,015.0 | 0.8195 | 88.20% | 5 | 2 |
| 2 | 14.jpg | Easy | 3,541.1 | 0.1703 | 96.51% | 3 | 1 |
| 2 | 15.jpg | Easy | 4,114.9 | 0.2954 | 82.24% | 2 | 1 |
| 2 | test-image-medium.png | Medium | 16,987.8 | 0.8952 | 88.79% | 18 | 5 |
| 2 | test-image-medium1.png | Medium | 47,781.3 | 0.6076 | 91.46% | 19 | 9 |
| 2 | test-image-medium2.png | Medium | 71,459.2 | 0.7176 | 87.97% | 19 | 10 |
| 2 | test-image-complex1.png | Complex | 103,696.7 | 0.6331 | 96.47% | 52 | 14 |
| 2 | test-image-complex2.jpg | Complex | 67,698.0 | 0.7709 | 81.87% | 24 | 10 |
| 2 | test-image-complex3.png | Complex | 221,200.3 | 0.7456 | 86.15% | 64 | 27 |
| 3 | 13.jpg | Easy | 5,635.9 | 0.8195 | 88.20% | 5 | 2 |
| 3 | 14.jpg | Easy | 3,753.2 | 0.0999 | 96.51% | 3 | 1 |
| 3 | 15.jpg | Easy | 4,067.4 | 0.2510 | 82.24% | 2 | 1 |
| 3 | test-image-medium.png | Medium | 14,090.9 | 0.6121 | 88.79% | 18 | 5 |
| 3 | test-image-medium1.png | Medium | 58,767.2 | 0.6860 | 91.46% | 19 | 9 |
| 3 | test-image-medium2.png | Medium | 70,545.1 | 0.7077 | 87.97% | 19 | 10 |
| 3 | test-image-complex1.png | Complex | 104,363.8 | 0.6001 | 96.47% | 52 | 14 |
| 3 | test-image-complex2.jpg | Complex | 67,054.1 | 0.7801 | 81.87% | 24 | 10 |
| 3 | test-image-complex3.png | Complex | 222,270.4 | 0.8063 | 86.15% | 64 | 27 |

---

## 6. Multi-Resolution Performance Benchmark

**Source Image:** test-image-medium.png (original: 1022×1850)  
**Method:** Image resized to each target resolution using Lanczos resampling, then processed through the full pipeline.

### Table 9: Processing Time by Resolution

| Resolution | Pixels (MP) | Detection (ms) | OCR (ms) | Texts Found | Total (ms) |
|:----------:|:-----------:|:--------------:|:--------:|:-----------:|:----------:|
| 640×480 | 0.31 | 3,697.36 | 2,769.64 | 0 | 6,467.57 |
| 800×600 | 0.48 | 354.51 | 323.00 | 4 | 19,485.08 |
| 1024×768 | 0.79 | 354.87 | 361.60 | 8 | 12,468.99 |
| 1280×720 | 0.92 | 622.76 | — | 0 | 623.00 |
| 1920×1080 | 2.07 | 285.31 | — | 0 | 285.54 |
| 2560×1440 | 3.69 | 358.39 | — | 0 | 358.67 |

> **Note:** The source image has a portrait aspect ratio (1022×1850). Resizing to landscape resolutions (e.g., 1280×720, 1920×1080) severely distorts the manga panels, causing text detection to fail. The 640×480 cold-start detection time (3,697ms) reflects model initialization overhead; subsequent detections are ~285–623ms regardless of resolution. Resolutions 800×600 and 1024×768 retained enough text structure for partial detection (4 and 8 texts respectively out of 18 in the original).

### Table 10: Detection-Only Timing (Excluding Model Warm-up)

| Resolution | Pixels (MP) | Detection (ms) |
|:----------:|:-----------:|:--------------:|
| 800×600 | 0.48 | 354.51 |
| 1024×768 | 0.79 | 354.87 |
| 1280×720 | 0.92 | 622.76 |
| 1920×1080 | 2.07 | 285.31 |
| 2560×1440 | 3.69 | 358.39 |

**Detection time is resolution-independent** (~285–623ms) because DBNet-ResNet34 internally resizes inputs to its configured detection size (1536px). The network processes at a fixed resolution regardless of input dimensions.

---

## 7. Key Findings

### Translation Quality (BLEU)
- **Overall mean BLEU: 0.6139** across all 9 images (27 evaluations)
- **Complex images achieved the highest category BLEU (0.7223)**, followed by Medium (0.7088) and Easy (0.4106)
- The low Easy BLEU (0.4106) is driven by 14.jpg (0.1289) and 15.jpg (0.3363), which have only 1 translation each — small n-gram reference windows result in low BLEU sensitivity
- **Best individual result: test-image-medium.png Run 2 (0.8952)**
- **BLEU variance is low for complex images** (std dev 0.0076–0.0353), indicating stable translation quality with more text context

### Processing Performance
- **Translation is the dominant bottleneck**, consuming 70.16% of total pipeline time
- Translation time scales with text count: ~500ms (1–2 translations) → ~173s (27 translations)
- **Detection, OCR, and Inpainting are efficient and consistent** (~0.5–2s each regardless of complexity)
- **Rendering time scales linearly** with translated region count
- Easy images process in ~4.7s, Medium in ~45.2s, Complex in ~131.9s

### OCR Accuracy
- **100% OCR accuracy and 0% CER across all 27 runs**, demonstrating perfect deterministic consistency
- **Mean OCR confidence: 88.85%**, uniform across categories (Easy: 88.98%, Medium: 89.41%, Complex: 88.16%)
- Detection count is deterministic — same text regions detected on every run of the same image
