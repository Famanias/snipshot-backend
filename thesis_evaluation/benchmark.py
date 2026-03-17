"""
Benchmark Runner for Thesis Evaluation

This module provides:
1. Integrated benchmarking combining accuracy and performance testing
2. Multi-image batch processing
3. Multi-resolution testing
4. Comprehensive results export for thesis documentation
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import io

from PIL import Image
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from .accuracy import AccuracyEvaluator, CharacterAccuracyResult, BLEUResult
from .performance import PerformanceProfiler, AsyncStageTimer


@dataclass
class GroundTruthEntry:
    """Ground truth data for a single image"""
    image_path: str
    text_regions: List[Dict]  # List of {text: str, bbox: [x1,y1,x2,y2]}
    translations: List[Dict]  # List of {source: str, target: str}


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single image"""
    image_path: str
    resolution: Tuple[int, int]
    
    # Timing results
    detection_time_ms: float = 0.0
    ocr_time_ms: float = 0.0
    translation_time_ms: float = 0.0
    inpainting_time_ms: float = 0.0
    rendering_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Accuracy results
    ocr_accuracy: float = 0.0
    ocr_cer: float = 0.0
    bleu_score: float = 0.0
    
    # Detailed results
    ocr_results: List[CharacterAccuracyResult] = field(default_factory=list)
    translation_results: List[BLEUResult] = field(default_factory=list)
    
    # Detected data
    detected_texts: List[str] = field(default_factory=list)
    translated_texts: List[str] = field(default_factory=list)
    ocr_confidences: List[float] = field(default_factory=list)  # OCR confidence scores
    
    # Pipeline info
    detection_resolution: str = ""
    inpainting_resolution: str = ""
    regions_detected: int = 0
    tokens_used: int = 0
    
    # Metadata
    timestamp: str = ""
    config_used: Dict = field(default_factory=dict)
    error: str = ""


class BenchmarkRunner:
    """
    Main benchmark runner integrating the manga translation pipeline
    with accuracy and performance evaluation.
    """
    
    def __init__(self, 
                 ground_truth_path: str = None,
                 output_dir: str = None,
                 verbose: bool = True):
        """
        Initialize benchmark runner.
        
        Args:
            ground_truth_path: Path to ground truth JSON file
            output_dir: Directory for output results
            verbose: Print progress information
        """
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(__file__), '..', 'benchmark_results'
        )
        self.verbose = verbose
        
        self.accuracy_evaluator = AccuracyEvaluator()
        self.profiler = PerformanceProfiler()
        
        self.ground_truth: Dict[str, GroundTruthEntry] = {}
        self.results: List[BenchmarkResult] = []
        
        # Load ground truth if provided
        if ground_truth_path and os.path.exists(ground_truth_path):
            self.load_ground_truth(ground_truth_path)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_ground_truth(self, filepath: str):
        """
        Load ground truth data from JSON file.
        
        Expected format:
        {
            "images": [
                {
                    "path": "image1.jpg",
                    "text_regions": [
                        {"text": "原文テキスト", "bbox": [x1, y1, x2, y2]},
                        ...
                    ],
                    "translations": [
                        {"source": "原文", "target": "English translation"},
                        ...
                    ]
                }
            ]
        }
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for entry in data.get('images', []):
            image_path = entry.get('path', '')
            self.ground_truth[image_path] = GroundTruthEntry(
                image_path=image_path,
                text_regions=entry.get('text_regions', []),
                translations=entry.get('translations', [])
            )
        
        if self.verbose:
            print(f"Loaded ground truth for {len(self.ground_truth)} images")
    
    async def benchmark_single_image(self,
                                     image_path: str,
                                     config: Dict = None,
                                     translator_instance = None) -> BenchmarkResult:
        """
        Run benchmark on a single image.
        
        Args:
            image_path: Path to test image
            config: Translation configuration
            translator_instance: Pre-initialized MangaTranslator instance
            
        Returns:
            BenchmarkResult
        """
        from manga_translator import Config
        from manga_translator.manga_translator import MangaTranslator
        from manga_translator.utils import Context
        
        result = BenchmarkResult(
            image_path=image_path,
            resolution=(0, 0),
            timestamp=datetime.now().isoformat()
        )
        
        try:
            # Load image
            img = Image.open(image_path).convert('RGB')
            result.resolution = img.size
            
            if self.verbose:
                print(f"\nProcessing: {image_path}")
                print(f"  Resolution: {result.resolution[0]}x{result.resolution[1]}")
            
            # Default config
            if config is None:
                config = {
                    "detector": {"detector": "default", "detection_size": 1536},
                    "ocr": {"ocr": "48px"},
                    "translator": {"translator": "groq", "target_lang": "ENG"},
                    "inpainter": {"inpainter": "lama_large"},
                    "render": {"renderer": "default"}
                }
            
            result.config_used = config
            config_obj = Config(**config)
            
            # Create translator if not provided
            if translator_instance is None:
                translator_instance = MangaTranslator({
                    'use_gpu': True,
                    'kernel_size': 3,
                    'verbose': False
                })
            
            ctx = Context()
            ctx.input = img
            ctx.result = None
            ctx.img_rgb = np.array(img)
            
            # Time each stage
            metadata = {'resolution': result.resolution, 'image': image_path}
            total_start = time.perf_counter()
            
            # Detection
            async with AsyncStageTimer('detection', metadata) as timer:
                detection_result = await translator_instance._run_detection(config_obj, ctx)
                ctx.textlines = detection_result[0] if isinstance(detection_result, tuple) else detection_result
                ctx.mask_raw = detection_result[1] if isinstance(detection_result, tuple) and len(detection_result) > 1 else None
            result.detection_time_ms = timer.duration_ms
            self.profiler.record_timing('detection', timer.duration_ms, metadata)
            
            if self.verbose:
                print(f"  Detection: {timer.duration_ms:.2f}ms - Found {len(ctx.textlines) if ctx.textlines else 0} regions")
            
            # OCR
            if ctx.textlines:
                async with AsyncStageTimer('ocr', metadata) as timer:
                    ctx.textlines = await translator_instance._run_ocr(config_obj, ctx)
                result.ocr_time_ms = timer.duration_ms
                self.profiler.record_timing('ocr', timer.duration_ms, metadata)
                
                # Extract detected texts and confidence scores
                for tl in ctx.textlines:
                    if hasattr(tl, 'text') and tl.text:
                        result.detected_texts.append(tl.text)
                        # Extract OCR confidence (prob attribute)
                        if hasattr(tl, 'prob'):
                            result.ocr_confidences.append(float(tl.prob))
                        else:
                            result.ocr_confidences.append(1.0)  # Default if not available
                
                if self.verbose:
                    avg_conf = sum(result.ocr_confidences) / len(result.ocr_confidences) if result.ocr_confidences else 0
                    print(f"  OCR: {timer.duration_ms:.2f}ms - Recognized {len(result.detected_texts)} texts (avg confidence: {avg_conf:.2%})")
            
            # Textline merge
            if ctx.textlines:
                ctx.text_regions = await translator_instance._run_textline_merge(config_obj, ctx)
                
                # Mask refinement
                ctx.mask = await translator_instance._run_mask_refinement(config_obj, ctx)
            
            # Translation
            if hasattr(ctx, 'text_regions') and ctx.text_regions:
                async with AsyncStageTimer('translation', metadata) as timer:
                    ctx.text_regions = await translator_instance._run_text_translation(config_obj, ctx)
                result.translation_time_ms = timer.duration_ms
                self.profiler.record_timing('translation', timer.duration_ms, metadata)
                
                # Extract translations
                for region in ctx.text_regions:
                    if hasattr(region, 'translation') and region.translation:
                        result.translated_texts.append(region.translation)
                
                if self.verbose:
                    print(f"  Translation: {timer.duration_ms:.2f}ms - Translated {len(result.translated_texts)} regions")
            
            # Inpainting
            if hasattr(ctx, 'mask') and ctx.mask is not None:
                try:
                    async with AsyncStageTimer('inpainting', metadata) as timer:
                        ctx.img_inpainted = await translator_instance._run_inpainting(config_obj, ctx)
                    result.inpainting_time_ms = timer.duration_ms
                    self.profiler.record_timing('inpainting', timer.duration_ms, metadata)
                    
                    if self.verbose:
                        print(f"  Inpainting: {timer.duration_ms:.2f}ms")
                except Exception as inpaint_error:
                    if self.verbose:
                        print(f"  Inpainting: SKIPPED (error: {str(inpaint_error)[:50]}...)")
                    ctx.img_inpainted = ctx.img_rgb  # Use original image if inpainting fails
            else:
                ctx.img_inpainted = ctx.img_rgb
            
            # Rendering
            if hasattr(ctx, 'text_regions') and ctx.text_regions:
                try:
                    async with AsyncStageTimer('rendering', metadata) as timer:
                        ctx.result = await translator_instance._run_text_rendering(config_obj, ctx)
                    result.rendering_time_ms = timer.duration_ms
                    self.profiler.record_timing('rendering', timer.duration_ms, metadata)
                    
                    if self.verbose:
                        print(f"  Rendering: {timer.duration_ms:.2f}ms")
                except Exception as render_error:
                    if self.verbose:
                        print(f"  Rendering: SKIPPED (error: {str(render_error)[:50]}...)")
            
            # Total time
            result.total_time_ms = (time.perf_counter() - total_start) * 1000
            self.profiler.record_timing('total', result.total_time_ms, metadata)
            
            if self.verbose:
                print(f"  Total: {result.total_time_ms:.2f}ms")
            
            # Calculate accuracy if ground truth available
            await self._evaluate_accuracy(result)
            
        except Exception as e:
            result.error = str(e)
            if self.verbose:
                print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        self.results.append(result)
        return result
    
    async def _evaluate_accuracy(self, result: BenchmarkResult):
        """Evaluate OCR and translation accuracy against ground truth"""
        image_name = os.path.basename(result.image_path)
        
        gt_entry = self.ground_truth.get(image_name) or self.ground_truth.get(result.image_path)
        
        if not gt_entry:
            if self.verbose:
                print(f"  No ground truth available for accuracy evaluation")
            return
        
        # OCR Accuracy
        if gt_entry.text_regions and result.detected_texts:
            gt_texts = [r.get('text', '') for r in gt_entry.text_regions]
            
            # Match detected texts to ground truth (simple sequential matching)
            ocr_accuracies = []
            for i, gt_text in enumerate(gt_texts):
                if i < len(result.detected_texts):
                    det_text = result.detected_texts[i]
                    acc_result = self.accuracy_evaluator.evaluate_ocr(
                        gt_text, det_text, 
                        image_id=f"{image_name}_{i}"
                    )
                    result.ocr_results.append(acc_result)
                    ocr_accuracies.append(acc_result.accuracy)
            
            if ocr_accuracies:
                result.ocr_accuracy = sum(ocr_accuracies) / len(ocr_accuracies)
                result.ocr_cer = sum(r.character_error_rate for r in result.ocr_results) / len(result.ocr_results)
                
                if self.verbose:
                    print(f"  OCR Accuracy: {result.ocr_accuracy:.2%} (CER: {result.ocr_cer:.2%})")
        
        # Translation BLEU
        if gt_entry.translations and result.translated_texts:
            gt_translations = [t.get('target', '') for t in gt_entry.translations]
            
            bleu_scores = []
            for i, ref in enumerate(gt_translations):
                if i < len(result.translated_texts):
                    hyp = result.translated_texts[i]
                    bleu_result = self.accuracy_evaluator.evaluate_translation(
                        ref, hyp,
                        image_id=f"{image_name}_{i}",
                        target_lang="ENG"
                    )
                    result.translation_results.append(bleu_result)
                    bleu_scores.append(bleu_result.bleu_score)
            
            if bleu_scores:
                result.bleu_score = sum(bleu_scores) / len(bleu_scores)
                
                if self.verbose:
                    print(f"  BLEU Score: {result.bleu_score:.4f}")
    
    async def run_benchmark(self,
                           image_paths: List[str],
                           config: Dict = None,
                           num_runs: int = 1) -> Dict:
        """
        Run benchmark on multiple images.
        
        Args:
            image_paths: List of image paths to test
            config: Translation configuration
            num_runs: Number of runs per image for averaging
            
        Returns:
            Benchmark summary dictionary
        """
        from manga_translator.manga_translator import MangaTranslator
        
        if self.verbose:
            print("=" * 60)
            print("THESIS BENCHMARK - MANGA TRANSLATION PIPELINE")
            print("=" * 60)
            print(f"Images: {len(image_paths)}")
            print(f"Runs per image: {num_runs}")
            print("=" * 60)
        
        # Initialize translator once
        translator = MangaTranslator({
            'use_gpu': True,
            'kernel_size': 3,
            'verbose': False
        })
        
        for run in range(num_runs):
            if self.verbose and num_runs > 1:
                print(f"\n--- Run {run + 1}/{num_runs} ---")
            
            for idx, image_path in enumerate(image_paths):
                if os.path.exists(image_path):
                    # Add delay between images to avoid API rate limiting
                    if idx > 0:
                        import asyncio
                        await asyncio.sleep(5)
                    self.profiler.new_run()
                    await self.benchmark_single_image(
                        image_path, config, translator
                    )
                else:
                    print(f"WARNING: Image not found: {image_path}")
        
        return self.generate_summary()
    
    async def run_resolution_benchmark(self,
                                       image_path: str,
                                       resolutions: List[Tuple[int, int]] = None,
                                       config: Dict = None) -> Dict:
        """
        Test performance across different image resolutions.
        
        Args:
            image_path: Path to test image
            resolutions: List of (width, height) tuples to test
            config: Translation configuration
            
        Returns:
            Resolution benchmark results
        """
        if resolutions is None:
            resolutions = [
                (640, 480),
                (800, 600),
                (1024, 768),
                (1280, 720),
                (1920, 1080),
                (2560, 1440),
            ]
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("MULTI-RESOLUTION BENCHMARK")
            print("=" * 60)
        
        original_img = Image.open(image_path).convert('RGB')
        
        resolution_results = []
        
        for width, height in resolutions:
            # Resize image
            resized = original_img.resize((width, height), Image.Resampling.LANCZOS)
            
            # Save to temp file
            temp_path = os.path.join(self.output_dir, f"temp_{width}x{height}.png")
            resized.save(temp_path)
            
            if self.verbose:
                print(f"\nTesting resolution: {width}x{height}")
            
            # Run benchmark
            result = await self.benchmark_single_image(temp_path, config)
            resolution_results.append({
                'resolution': f"{width}x{height}",
                'width': width,
                'height': height,
                'pixels': width * height,
                'total_time_ms': result.total_time_ms,
                'detection_time_ms': result.detection_time_ms,
                'ocr_time_ms': result.ocr_time_ms,
                'translation_time_ms': result.translation_time_ms,
                'inpainting_time_ms': result.inpainting_time_ms,
                'rendering_time_ms': result.rendering_time_ms
            })
            
            # Cleanup temp file
            os.remove(temp_path)
        
        return {
            'source_image': image_path,
            'results': resolution_results,
            'analysis': self._analyze_resolution_scaling(resolution_results)
        }
    
    def _analyze_resolution_scaling(self, results: List[Dict]) -> Dict:
        """Analyze how processing time scales with resolution"""
        if len(results) < 2:
            return {}
        
        # Calculate scaling factors
        pixels = [r['pixels'] for r in results]
        times = [r['total_time_ms'] for r in results]
        
        # Simple linear regression for ms/megapixel
        avg_ms_per_megapixel = sum(
            t / (p / 1_000_000) for t, p in zip(times, pixels)
        ) / len(results)
        
        return {
            'avg_ms_per_megapixel': avg_ms_per_megapixel,
            'min_resolution': results[0]['resolution'],
            'max_resolution': results[-1]['resolution'],
            'scaling_factor': times[-1] / times[0] if times[0] > 0 else 0
        }
    
    def generate_summary(self) -> Dict:
        """Generate comprehensive benchmark summary"""
        if not self.results:
            return {'error': 'No results available'}
        
        # Filter successful results
        successful = [r for r in self.results if not r.error]
        
        if not successful:
            return {'error': 'All benchmarks failed'}
        
        summary = {
            'total_images': len(self.results),
            'successful': len(successful),
            'failed': len(self.results) - len(successful),
            
            'timing': {
                'mean_total_ms': sum(r.total_time_ms for r in successful) / len(successful),
                'mean_detection_ms': sum(r.detection_time_ms for r in successful) / len(successful),
                'mean_ocr_ms': sum(r.ocr_time_ms for r in successful) / len(successful),
                'mean_translation_ms': sum(r.translation_time_ms for r in successful) / len(successful),
                'mean_inpainting_ms': sum(r.inpainting_time_ms for r in successful) / len(successful),
                'mean_rendering_ms': sum(r.rendering_time_ms for r in successful) / len(successful),
            },
            
            'ocr_confidence': {
                'mean': sum(sum(r.ocr_confidences) / len(r.ocr_confidences) if r.ocr_confidences else 0 for r in successful) / len(successful),
                'all_scores': [conf for r in successful for conf in r.ocr_confidences]
            },
            
            'accuracy': self.accuracy_evaluator.get_aggregate_ocr_accuracy(),
            'bleu': self.accuracy_evaluator.get_aggregate_bleu(),
            
            'profiler_stats': self.profiler.export_results()
        }
        
        return summary
    
    def export_results(self, filename: str = None) -> str:
        """
        Export all results to JSON file.
        
        Args:
            filename: Output filename (default: benchmark_results_{timestamp}.json)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        export_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_images': len(self.results),
                'ground_truth_file': self.ground_truth_path
            },
            'summary': self.generate_summary(),
            'detailed_results': [
                {
                    'image_path': r.image_path,
                    'resolution': f"{r.resolution[0]}x{r.resolution[1]}",
                    'timing': {
                        'total_ms': r.total_time_ms,
                        'detection_ms': r.detection_time_ms,
                        'ocr_ms': r.ocr_time_ms,
                        'translation_ms': r.translation_time_ms,
                        'inpainting_ms': r.inpainting_time_ms,
                        'rendering_ms': r.rendering_time_ms
                    },
                    'accuracy': {
                        'ocr_accuracy': r.ocr_accuracy,
                        'ocr_cer': r.ocr_cer,
                        'bleu_score': r.bleu_score
                    },
                    'ocr_details': {
                        'texts': r.detected_texts,
                        'confidences': r.ocr_confidences,
                        'mean_confidence': sum(r.ocr_confidences) / len(r.ocr_confidences) if r.ocr_confidences else 0
                    },
                    'detected_texts': r.detected_texts,
                    'translated_texts': r.translated_texts,
                    'config': r.config_used,
                    'error': r.error
                }
                for r in self.results
            ],
            'accuracy_evaluation': self.accuracy_evaluator.export_results(),
            'performance_profile': self.profiler.export_results()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        if self.verbose:
            print(f"\nResults exported to: {filepath}")
        
        return filepath
    
    def print_summary(self):
        """Print formatted benchmark summary"""
        summary = self.generate_summary()
        
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        
        print(f"\nImages Tested: {summary.get('total_images', 0)}")
        print(f"Successful: {summary.get('successful', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        
        timing = summary.get('timing', {})
        print("\n--- PROCESSING TIME (ms) ---")
        print(f"  Total:        {timing.get('mean_total_ms', 0):.2f}")
        print(f"  Detection:    {timing.get('mean_detection_ms', 0):.2f}")
        print(f"  OCR:          {timing.get('mean_ocr_ms', 0):.2f}")
        print(f"  Translation:  {timing.get('mean_translation_ms', 0):.2f}")
        print(f"  Inpainting:   {timing.get('mean_inpainting_ms', 0):.2f}")
        print(f"  Rendering:    {timing.get('mean_rendering_ms', 0):.2f}")
        
        accuracy = summary.get('accuracy', {})
        print("\n--- OCR ACCURACY ---")
        print(f"  Mean Accuracy: {accuracy.get('mean_accuracy', 0):.2%}")
        print(f"  Mean CER:      {accuracy.get('mean_cer', 0):.2%}")
        print(f"  Samples:       {accuracy.get('sample_count', 0)}")
        
        bleu = summary.get('bleu', {})
        print("\n--- TRANSLATION BLEU ---")
        print(f"  Mean BLEU:     {bleu.get('mean_bleu', 0):.4f}")
        print(f"  Samples:       {bleu.get('sample_count', 0)}")
        
        print("=" * 70)


def create_ground_truth_template(output_path: str, image_paths: List[str] = None):
    """
    Create a template ground truth JSON file.
    
    Args:
        output_path: Path for output JSON file
        image_paths: Optional list of image paths to include
    """
    template = {
        "description": "Ground truth data for thesis evaluation",
        "instructions": [
            "Fill in 'text' with the exact Japanese/source text in each speech bubble",
            "Fill in 'source' with the original text and 'target' with the correct English translation",
            "bbox format: [x1, y1, x2, y2] - coordinates of the text region"
        ],
        "images": []
    }
    
    if image_paths:
        for path in image_paths:
            template["images"].append({
                "path": os.path.basename(path),
                "text_regions": [
                    {"text": "テキスト例1", "bbox": [100, 100, 200, 150]},
                    {"text": "テキスト例2", "bbox": [100, 200, 200, 250]}
                ],
                "translations": [
                    {"source": "テキスト例1", "target": "Example text 1"},
                    {"source": "テキスト例2", "target": "Example text 2"}
                ]
            })
    else:
        template["images"].append({
            "path": "example_image.jpg",
            "text_regions": [
                {"text": "原文テキスト", "bbox": [100, 100, 200, 150]}
            ],
            "translations": [
                {"source": "原文テキスト", "target": "Original text"}
            ]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    
    print(f"Ground truth template created: {output_path}")
