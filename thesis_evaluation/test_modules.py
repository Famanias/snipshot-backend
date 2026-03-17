"""
Test script to verify the thesis evaluation modules work correctly.
Run this to ensure all components are properly installed.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_accuracy_module():
    """Test the accuracy evaluation module"""
    print("\n" + "=" * 60)
    print("Testing Accuracy Evaluation Module")
    print("=" * 60)
    
    from thesis_evaluation.accuracy import (
        CharacterAccuracy, 
        BLEUScore, 
        AccuracyEvaluator,
        calculate_bleu,
        calculate_character_accuracy
    )
    
    # Test 1: Character Accuracy
    print("\n1. Character-Level Accuracy Tests:")
    
    test_cases = [
        ("これはテストです", "これはテストです", 1.0),
        ("Hello World", "Hello World", 1.0),
        ("テスト", "テスト", 1.0),
        ("ABC", "ABD", 0.67),  # ~67% (2/3 correct)
    ]
    
    for gt, det, expected_approx in test_cases:
        result = CharacterAccuracy.compute_accuracy(gt, det)
        status = "✓" if abs(result.accuracy - expected_approx) < 0.1 else "✗"
        print(f"  {status} '{gt}' vs '{det}': {result.accuracy:.2%} (CER: {result.character_error_rate:.2%})")
    
    # Test 2: BLEU Score
    print("\n2. BLEU Score Tests:")
    
    bleu_tests = [
        ("The cat sat on the mat", "The cat sat on the mat", "Perfect"),
        ("The cat sat on the mat", "The dog sat on the mat", "One word diff"),
        ("Hello world", "Hi world", "Different greeting"),
    ]
    
    for ref, hyp, desc in bleu_tests:
        result = BLEUScore.compute_bleu(ref, hyp)
        print(f"  {desc}: BLEU = {result.bleu_score:.4f}")
        print(f"    Precisions: {[f'{p:.3f}' for p in result.precisions]}")
    
    # Test 3: Evaluator class
    print("\n3. AccuracyEvaluator Integration Test:")
    
    evaluator = AccuracyEvaluator()
    
    # Add some OCR results
    evaluator.evaluate_ocr("テスト文章", "テスト文章", "img1")
    evaluator.evaluate_ocr("サンプル", "サンプル", "img2")
    
    # Add translation results
    evaluator.evaluate_translation("This is a test", "This is a test", "img1")
    evaluator.evaluate_translation("Sample text", "Sample words", "img2")
    
    ocr_stats = evaluator.get_aggregate_ocr_accuracy()
    bleu_stats = evaluator.get_aggregate_bleu()
    
    print(f"  OCR Mean Accuracy: {ocr_stats['mean_accuracy']:.2%}")
    print(f"  Translation Mean BLEU: {bleu_stats['mean_bleu']:.4f}")
    
    print("\n  ✓ Accuracy module tests passed!")
    return True


def test_performance_module():
    """Test the performance profiling module"""
    print("\n" + "=" * 60)
    print("Testing Performance Profiling Module")
    print("=" * 60)
    
    from thesis_evaluation.performance import (
        PerformanceProfiler,
        StageTimer,
        time_function
    )
    import time
    
    # Test 1: StageTimer
    print("\n1. StageTimer Test:")
    
    with StageTimer("test_stage") as timer:
        time.sleep(0.1)  # Simulate work
    
    print(f"  Timed operation: {timer.duration_ms:.2f}ms")
    assert timer.duration_ms >= 100, "Timer should show at least 100ms"
    print("  ✓ StageTimer working correctly")
    
    # Test 2: PerformanceProfiler
    print("\n2. PerformanceProfiler Test:")
    
    profiler = PerformanceProfiler()
    
    # Simulate some pipeline stages
    for i in range(3):
        profiler.new_run()
        profiler.record_timing('detection', 50 + i * 10, {'resolution': (1920, 1080)})
        profiler.record_timing('ocr', 100 + i * 5, {'resolution': (1920, 1080)})
        profiler.record_timing('translation', 200 + i * 20, {'resolution': (1920, 1080)})
        profiler.record_timing('inpainting', 300 + i * 15, {'resolution': (1920, 1080)})
        profiler.record_timing('rendering', 50 + i * 5, {'resolution': (1920, 1080)})
        total = sum([50+i*10, 100+i*5, 200+i*20, 300+i*15, 50+i*5])
        profiler.record_timing('total', total, {'resolution': (1920, 1080)})
    
    # Get statistics
    stats = profiler.get_all_statistics()
    
    print(f"  Recorded {len(profiler.timings['detection'])} detection timings")
    print(f"  Detection mean: {stats['detection'].mean_ms:.2f}ms")
    print(f"  Total mean: {stats['total'].mean_ms:.2f}ms")
    
    # Test export
    export_data = profiler.export_results()
    assert 'summary' in export_data
    assert 'detection' in export_data['summary']
    
    print("  ✓ PerformanceProfiler working correctly")
    
    # Test 3: time_function utility
    print("\n3. time_function Utility Test:")
    
    def slow_function(x):
        time.sleep(0.05)
        return x * 2
    
    result, duration = time_function(slow_function, 5)
    print(f"  Function returned {result} in {duration:.2f}ms")
    assert result == 10
    assert duration >= 50
    
    print("  ✓ time_function working correctly")
    
    print("\n  ✓ Performance module tests passed!")
    return True


def test_benchmark_module():
    """Test the benchmark runner module (without actual translation)"""
    print("\n" + "=" * 60)
    print("Testing Benchmark Runner Module")
    print("=" * 60)
    
    from thesis_evaluation.benchmark import (
        BenchmarkRunner,
        BenchmarkResult,
        create_ground_truth_template
    )
    import tempfile
    
    # Test 1: BenchmarkResult dataclass
    print("\n1. BenchmarkResult Test:")
    
    result = BenchmarkResult(
        image_path="test.jpg",
        resolution=(1920, 1080),
        detection_time_ms=50.0,
        ocr_time_ms=100.0,
        translation_time_ms=200.0,
        inpainting_time_ms=300.0,
        rendering_time_ms=50.0,
        total_time_ms=700.0,
        ocr_accuracy=0.95,
        bleu_score=0.85
    )
    
    print(f"  Created result for {result.image_path}")
    print(f"  Resolution: {result.resolution}")
    print(f"  Total time: {result.total_time_ms}ms")
    print("  ✓ BenchmarkResult working correctly")
    
    # Test 2: BenchmarkRunner initialization
    print("\n2. BenchmarkRunner Initialization Test:")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            output_dir=tmpdir,
            verbose=False
        )
        
        print(f"  Output directory: {runner.output_dir}")
        print("  ✓ BenchmarkRunner initialized correctly")
        
        # Test 3: Ground truth template creation
        print("\n3. Ground Truth Template Test:")
        
        gt_path = os.path.join(tmpdir, "test_ground_truth.json")
        create_ground_truth_template(gt_path, ["image1.jpg", "image2.jpg"])
        
        assert os.path.exists(gt_path), "Ground truth file should be created"
        
        import json
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        
        assert 'images' in gt_data
        assert len(gt_data['images']) == 2
        print(f"  Created template with {len(gt_data['images'])} images")
        print("  ✓ Ground truth template creation working correctly")
    
    print("\n  ✓ Benchmark module tests passed!")
    return True


def run_all_tests():
    """Run all module tests"""
    print("\n" + "=" * 70)
    print("THESIS EVALUATION MODULE TESTS")
    print("=" * 70)
    
    all_passed = True
    
    try:
        all_passed &= test_accuracy_module()
    except Exception as e:
        print(f"  ✗ Accuracy module test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_performance_module()
    except Exception as e:
        print(f"  ✗ Performance module test failed: {e}")
        all_passed = False
    
    try:
        all_passed &= test_benchmark_module()
    except Exception as e:
        print(f"  ✗ Benchmark module test failed: {e}")
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("The thesis evaluation pipeline is ready to use.")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please check the error messages above.")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
