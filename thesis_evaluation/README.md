# SnipShot Thesis Evaluation Pipeline

Automated testing tools for thesis documentation measuring:
1. **Accuracy Testing** - OCR character-level accuracy and Translation BLEU scores
2. **Processing Speed Evaluation** - Per-stage timing across multiple image resolutions

## Quick Start

```bash
# Test that modules are working
python -m thesis_evaluation.test_modules

# Run quick accuracy demonstration
python -m thesis_evaluation.run_thesis_evaluation --quick-accuracy-test

# Run full benchmark on an image
python -m thesis_evaluation.run_thesis_evaluation --images 15.jpg

# Run with ground truth for accuracy evaluation
python -m thesis_evaluation.run_thesis_evaluation --images 15.jpg --ground-truth ground_truth.json

# Run multi-resolution performance test
python -m thesis_evaluation.run_thesis_evaluation --images 15.jpg --resolution-test
```

## Accuracy Testing

### Character-Level OCR Accuracy

Measures how accurately the OCR module detects text compared to ground truth transcriptions.

**Metrics:**
- **Character Accuracy**: Percentage of correctly detected characters
- **Character Error Rate (CER)**: Levenshtein distance normalized by reference length

```python
from thesis_evaluation.accuracy import CharacterAccuracy, calculate_character_accuracy

# Quick calculation
accuracy = calculate_character_accuracy("原文テキスト", "原文テクスト")
print(f"Accuracy: {accuracy:.2%}")

# Detailed results
result = CharacterAccuracy.compute_accuracy("原文テキスト", "原文テクスト")
print(f"Accuracy: {result.accuracy:.2%}")
print(f"CER: {result.character_error_rate:.2%}")
print(f"Correct chars: {result.correct_chars}/{result.total_chars}")
```

### Translation BLEU Score

Measures translation quality using the Bilingual Evaluation Understudy (BLEU) algorithm.

**Reference:** Papineni, K., et al. (2002). *BLEU: a Method for Automatic Evaluation of Machine Translation*

```python
from thesis_evaluation.accuracy import BLEUScore, calculate_bleu

# Quick calculation
bleu = calculate_bleu("This is correct.", "This is correct.")
print(f"BLEU: {bleu:.4f}")

# Detailed results
result = BLEUScore.compute_bleu(
    reference="The quick brown fox",
    hypothesis="A fast brown fox"
)
print(f"BLEU: {result.bleu_score:.4f}")
print(f"Precisions (1-4 gram): {result.precisions}")
print(f"Brevity Penalty: {result.brevity_penalty:.4f}")
```

## Processing Speed Evaluation

### Per-Stage Timing

Measures execution time for each pipeline stage:
- Detection (text region detection)
- OCR (optical character recognition)
- Translation (text translation)
- Inpainting (background reconstruction)
- Rendering (text rendering)

```python
from thesis_evaluation.performance import PerformanceProfiler, StageTimer

# Using the profiler
profiler = PerformanceProfiler()

with profiler.time_stage('detection', {'resolution': (1920, 1080)}):
    # detection code here
    pass

# Get statistics
stats = profiler.get_all_statistics()
print(f"Detection mean: {stats['detection'].mean_ms:.2f}ms")
```

### Multi-Resolution Testing

Tests performance scaling across different image resolutions:

```bash
python -m thesis_evaluation.run_thesis_evaluation --images 15.jpg --resolution-test
```

Custom resolutions:
```bash
python -m thesis_evaluation.run_thesis_evaluation --images 15.jpg --resolution-test --resolutions 640x480 1280x720 1920x1080
```

## Ground Truth Format

Create a JSON file with the following structure:

```json
{
  "images": [
    {
      "path": "image1.jpg",
      "text_regions": [
        {"text": "原文テキスト", "bbox": [100, 100, 200, 150]}
      ],
      "translations": [
        {"source": "原文テキスト", "target": "English translation"}
      ]
    }
  ]
}
```

Generate a template:
```bash
python -m thesis_evaluation.run_thesis_evaluation --create-ground-truth --images image1.jpg image2.jpg
```

## Full Benchmark Example

```bash
# Run 5 iterations for statistical reliability
python -m thesis_evaluation.run_thesis_evaluation \
    --images 15.jpg 155.jpg \
    --ground-truth ground_truth.json \
    --runs 5 \
    --output-dir benchmark_results
```

## Output Files

Results are exported to JSON files in the output directory:
- `benchmark_results_{timestamp}.json` - Full results including timing and accuracy
- `resolution_benchmark_{timestamp}.json` - Resolution-specific results

### Sample Output Structure

```json
{
  "summary": {
    "total_images": 2,
    "successful": 2,
    "timing": {
      "mean_total_ms": 1500.0,
      "mean_detection_ms": 150.0,
      "mean_ocr_ms": 200.0,
      "mean_translation_ms": 500.0,
      "mean_inpainting_ms": 400.0,
      "mean_rendering_ms": 250.0
    },
    "accuracy": {
      "mean_accuracy": 0.95,
      "mean_cer": 0.05
    },
    "bleu": {
      "mean_bleu": 0.75
    }
  }
}
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--images` | Image path(s) to benchmark |
| `--ground-truth, -g` | Path to ground truth JSON |
| `--output-dir, -o` | Output directory (default: benchmark_results) |
| `--runs, -n` | Number of benchmark runs per image |
| `--resolution-test` | Run multi-resolution benchmark |
| `--resolutions` | Custom resolutions (format: WxH) |
| `--create-ground-truth` | Create ground truth template |
| `--quick-accuracy-test` | Run quick module tests |
| `--detector` | Detector to use (default, ctd, craft, paddle) |
| `--translator` | Translator to use (groq, chatgpt, google, deepl) |
| `--target-lang` | Target language code (default: ENG) |
| `--quiet, -q` | Suppress verbose output |

## For Thesis Documentation

### Methodology Section

> Data were collected using an automated performance benchmarking pipeline measuring processing time and translation accuracy. For accuracy testing, an automated evaluation pipeline implemented in Python was developed to:
> - (a) compute character-level accuracy by comparing detected text against ground-truth transcriptions on a character-by-character basis using Levenshtein distance;
> - (b) calculate translation accuracy using the BLEU (Bilingual Evaluation Understudy) algorithm (Papineni et al., 2002).
> 
> For processing speed evaluation, execution times for each pipeline stage (detection, OCR, translation, inpainting, rendering) were measured across multiple image resolutions using automated timing instrumentation.

### Results Tables

The exported JSON can be converted to tables for your thesis. Key metrics to report:

**Table: OCR Accuracy Results**
| Metric | Value |
|--------|-------|
| Mean Character Accuracy | XX.XX% |
| Character Error Rate (CER) | X.XX% |
| Sample Count | N |

**Table: Translation Quality (BLEU)**
| Metric | Value |
|--------|-------|
| Mean BLEU Score | 0.XXXX |
| Min BLEU | 0.XXXX |
| Max BLEU | 0.XXXX |

**Table: Processing Time by Stage (ms)**
| Stage | Mean | Median | Std Dev |
|-------|------|--------|---------|
| Detection | XXX | XXX | XX |
| OCR | XXX | XXX | XX |
| Translation | XXX | XXX | XX |
| Inpainting | XXX | XXX | XX |
| Rendering | XXX | XXX | XX |
| **Total** | XXX | XXX | XX |

**Table: Processing Time by Resolution**
| Resolution | Total Time (ms) | ms/Megapixel |
|------------|-----------------|--------------|
| 640×480 | XXX | XXX |
| 1280×720 | XXX | XXX |
| 1920×1080 | XXX | XXX |
