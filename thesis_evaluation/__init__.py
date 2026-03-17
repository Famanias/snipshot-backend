"""
Thesis Evaluation Pipeline for SnipShot Manga Translation System

This module provides automated testing tools for:
1. Accuracy Testing:
   - Character-level OCR accuracy
   - Translation accuracy using BLEU score

2. Processing Speed Evaluation:
   - Per-stage timing instrumentation
   - Multi-resolution performance testing
"""

from .accuracy import CharacterAccuracy, BLEUScore, AccuracyEvaluator
from .performance import PerformanceProfiler, StageTimer
from .benchmark import BenchmarkRunner
