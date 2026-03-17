#!/usr/bin/env python3
"""
SnipShot Thesis Evaluation - Main Entry Point

This script runs the automated evaluation pipeline for thesis documentation:
1. Accuracy Testing (OCR character-level accuracy + Translation BLEU scores)
2. Processing Speed Evaluation (Per-stage timing across multiple resolutions)

Usage:
    python run_thesis_evaluation.py --images 15.jpg --ground-truth ground_truth.json
    python run_thesis_evaluation.py --images 15.jpg --resolution-test
    python run_thesis_evaluation.py --images image1.jpg image2.jpg --runs 3
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from thesis_evaluation.benchmark import BenchmarkRunner, create_ground_truth_template
from thesis_evaluation.accuracy import AccuracyEvaluator, calculate_bleu, calculate_character_accuracy
from thesis_evaluation.performance import PerformanceProfiler


def parse_args():
    parser = argparse.ArgumentParser(
        description="SnipShot Thesis Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark on single image
  python run_thesis_evaluation.py --images 15.jpg

  # Run benchmark with ground truth for accuracy evaluation
  python run_thesis_evaluation.py --images 15.jpg --ground-truth ground_truth.json

  # Run multi-resolution performance test
  python run_thesis_evaluation.py --images 15.jpg --resolution-test

  # Run multiple iterations for statistical reliability
  python run_thesis_evaluation.py --images 15.jpg --runs 5

  # Create ground truth template
  python run_thesis_evaluation.py --create-ground-truth --images 15.jpg 155.jpg

  # Quick accuracy test (BLEU and character accuracy)
  python run_thesis_evaluation.py --quick-accuracy-test
        """
    )
    
    parser.add_argument(
        '--images', nargs='+', 
        help='Image paths to benchmark'
    )
    
    parser.add_argument(
        '--ground-truth', '-g',
        help='Path to ground truth JSON file for accuracy evaluation'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='benchmark_results',
        help='Directory for output results (default: benchmark_results)'
    )
    
    parser.add_argument(
        '--runs', '-n', type=int, default=1,
        help='Number of benchmark runs per image for averaging (default: 1)'
    )
    
    parser.add_argument(
        '--resolution-test', action='store_true',
        help='Run multi-resolution performance benchmark'
    )
    
    parser.add_argument(
        '--resolutions', nargs='+',
        help='Custom resolutions for resolution test (format: WxH, e.g., 1920x1080)'
    )
    
    parser.add_argument(
        '--create-ground-truth', action='store_true',
        help='Create a ground truth template file'
    )
    
    parser.add_argument(
        '--quick-accuracy-test', action='store_true',
        help='Run quick accuracy module tests'
    )
    
    parser.add_argument(
        '--config', '-c',
        help='Path to translation config JSON file'
    )
    
    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Suppress verbose output'
    )
    
    parser.add_argument(
        '--detector', default='default',
        choices=['default', 'ctd', 'craft', 'paddle'],
        help='Text detector to use (default: default)'
    )
    
    parser.add_argument(
        '--translator', default='groq',
        choices=['groq', 'chatgpt', 'google', 'deepl', 'none'],
        help='Translator to use (default: groq)'
    )
    
    parser.add_argument(
        '--target-lang', default='ENG',
        help='Target language code (default: ENG)'
    )
    
    return parser.parse_args()


async def run_benchmark(args):
    """Run the main benchmark"""
    print("\n" + "=" * 70)
    print("SNIPSHOT THESIS EVALUATION PIPELINE")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Build config from args
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "detector": {
                "detector": args.detector,
                "detection_size": 1536,
                "box_threshold": 0.7
            },
            "ocr": {
                "ocr": "48px"
            },
            "translator": {
                "translator": args.translator,
                "target_lang": args.target_lang
            },
            "inpainter": {
                "inpainter": "default"
            },
            "render": {
                "renderer": "default"
            }
        }
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    # Validate images exist
    valid_images = []
    for img_path in args.images:
        if os.path.exists(img_path):
            valid_images.append(img_path)
        else:
            print(f"WARNING: Image not found: {img_path}")
    
    if not valid_images:
        print("ERROR: No valid images found")
        return
    
    # Run benchmark
    summary = await runner.run_benchmark(
        image_paths=valid_images,
        config=config,
        num_runs=args.runs
    )
    
    # Print summary
    runner.print_summary()
    
    # Export results
    results_file = runner.export_results()
    print(f"\nDetailed results saved to: {results_file}")
    
    return summary


async def run_resolution_benchmark(args):
    """Run multi-resolution performance benchmark"""
    print("\n" + "=" * 70)
    print("MULTI-RESOLUTION PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    if not args.images or len(args.images) == 0:
        print("ERROR: Please specify at least one image with --images")
        return
    
    image_path = args.images[0]
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    # Parse custom resolutions if provided
    resolutions = None
    if args.resolutions:
        resolutions = []
        for res in args.resolutions:
            try:
                w, h = res.lower().split('x')
                resolutions.append((int(w), int(h)))
            except:
                print(f"WARNING: Invalid resolution format: {res}")
    
    runner = BenchmarkRunner(
        output_dir=args.output_dir,
        verbose=not args.quiet
    )
    
    results = await runner.run_resolution_benchmark(
        image_path=image_path,
        resolutions=resolutions
    )
    
    # Print resolution results
    print("\n" + "-" * 50)
    print("RESOLUTION BENCHMARK RESULTS")
    print("-" * 50)
    print(f"{'Resolution':<15} {'Total (ms)':<12} {'Detection':<12} {'OCR':<12} {'Inpaint':<12}")
    print("-" * 50)
    
    for r in results['results']:
        print(f"{r['resolution']:<15} {r['total_time_ms']:<12.2f} {r['detection_time_ms']:<12.2f} "
              f"{r['ocr_time_ms']:<12.2f} {r['inpainting_time_ms']:<12.2f}")
    
    if 'analysis' in results:
        analysis = results['analysis']
        print("\n" + "-" * 50)
        print("SCALING ANALYSIS")
        print("-" * 50)
        print(f"Average ms per megapixel: {analysis.get('avg_ms_per_megapixel', 0):.2f}")
        print(f"Scaling factor (min to max): {analysis.get('scaling_factor', 0):.2f}x")
    
    # Export results
    runner.export_results(f"resolution_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


def run_quick_accuracy_test():
    """Run quick tests of the accuracy evaluation module"""
    print("\n" + "=" * 70)
    print("ACCURACY MODULE QUICK TEST")
    print("=" * 70)
    
    # Test character accuracy
    print("\n--- Character-Level Accuracy Tests ---")
    
    test_cases = [
        ("これはテストです", "これはテストです", "Perfect match"),
        ("これはテストです", "これはテストだ", "Minor difference"),
        ("Hello World", "Hello World", "English - perfect"),
        ("Hello World", "Helo Wrld", "English - typos"),
        ("日本語テキスト", "日本語テクスト", "Japanese - one char diff"),
    ]
    
    for gt, detected, desc in test_cases:
        accuracy = calculate_character_accuracy(gt, detected)
        print(f"  {desc}:")
        print(f"    GT: '{gt}' | Det: '{detected}'")
        print(f"    Accuracy: {accuracy:.2%}")
    
    # Test BLEU score
    print("\n--- BLEU Score Tests ---")
    
    translation_tests = [
        ("This is a test sentence.", "This is a test sentence.", "Perfect match"),
        ("This is a test sentence.", "This is a sample sentence.", "One word diff"),
        ("The quick brown fox", "A fast brown fox", "Paraphrase"),
        ("私は学生です", "私は学生です", "Japanese - exact"),
    ]
    
    for ref, hyp, desc in translation_tests:
        bleu = calculate_bleu(ref, hyp)
        print(f"  {desc}:")
        print(f"    Ref: '{ref}'")
        print(f"    Hyp: '{hyp}'")
        print(f"    BLEU: {bleu:.4f}")
    
    print("\n" + "=" * 70)
    print("Accuracy module tests completed!")


def create_ground_truth(args):
    """Create ground truth template file"""
    output_path = os.path.join(args.output_dir, 'ground_truth_template.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    create_ground_truth_template(output_path, args.images)
    print(f"\nGround truth template created: {output_path}")
    print("Edit this file to add your ground truth data for accuracy evaluation.")


def main():
    args = parse_args()
    
    # Handle different modes
    if args.quick_accuracy_test:
        run_quick_accuracy_test()
        return
    
    if args.create_ground_truth:
        create_ground_truth(args)
        return
    
    if not args.images:
        print("ERROR: Please specify images with --images")
        print("Use --help for usage information")
        return
    
    # Run appropriate benchmark
    if args.resolution_test:
        asyncio.run(run_resolution_benchmark(args))
    else:
        asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
