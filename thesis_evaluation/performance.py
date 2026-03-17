"""
Performance Profiling Module for Thesis Testing

This module provides:
1. Per-stage timing instrumentation for the translation pipeline
2. Multi-resolution performance benchmarking
3. Statistical analysis of processing times
"""

import time
import statistics
import json
from typing import List, Dict, Optional, Callable, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from datetime import datetime
import asyncio


@dataclass
class TimingRecord:
    """Single timing measurement"""
    stage_name: str
    duration_ms: float
    timestamp: str
    metadata: Dict = field(default_factory=dict)


@dataclass
class StageStatistics:
    """Aggregated statistics for a pipeline stage"""
    stage_name: str
    mean_ms: float
    median_ms: float
    std_dev_ms: float
    min_ms: float
    max_ms: float
    sample_count: int
    total_time_ms: float


class StageTimer:
    """
    Context manager for timing individual pipeline stages.
    
    Usage:
        with StageTimer("detection") as timer:
            # detection code here
            pass
        print(f"Detection took {timer.duration_ms:.2f} ms")
    """
    
    def __init__(self, stage_name: str, metadata: Dict = None):
        self.stage_name = stage_name
        self.metadata = metadata or {}
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0
        self._record: Optional[TimingRecord] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self._record = TimingRecord(
            stage_name=self.stage_name,
            duration_ms=self.duration_ms,
            timestamp=datetime.now().isoformat(),
            metadata=self.metadata
        )
        return False  # Don't suppress exceptions
    
    @property
    def record(self) -> Optional[TimingRecord]:
        return self._record


class AsyncStageTimer:
    """
    Async context manager for timing async pipeline stages.
    
    Usage:
        async with AsyncStageTimer("translation") as timer:
            await translation_function()
        print(f"Translation took {timer.duration_ms:.2f} ms")
    """
    
    def __init__(self, stage_name: str, metadata: Dict = None):
        self.stage_name = stage_name
        self.metadata = metadata or {}
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0
        self._record: Optional[TimingRecord] = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self._record = TimingRecord(
            stage_name=self.stage_name,
            duration_ms=self.duration_ms,
            timestamp=datetime.now().isoformat(),
            metadata=self.metadata
        )
        return False
    
    @property
    def record(self) -> Optional[TimingRecord]:
        return self._record


class PerformanceProfiler:
    """
    Comprehensive performance profiler for the manga translation pipeline.
    
    Tracks timing for each stage:
    - Detection: Text region detection
    - OCR: Optical character recognition
    - Translation: Text translation
    - Inpainting: Background reconstruction
    - Rendering: Text rendering
    
    Provides statistical analysis across multiple runs.
    """
    
    PIPELINE_STAGES = [
        'detection',
        'ocr',
        'translation',
        'inpainting',
        'rendering',
        'total'
    ]
    
    def __init__(self):
        self.timings: Dict[str, List[TimingRecord]] = {
            stage: [] for stage in self.PIPELINE_STAGES
        }
        self.current_run: Dict[str, TimingRecord] = {}
        self._run_count = 0
    
    def record_timing(self, stage: str, duration_ms: float, metadata: Dict = None):
        """
        Record a timing measurement for a stage.
        
        Args:
            stage: Pipeline stage name
            duration_ms: Duration in milliseconds
            metadata: Additional metadata (e.g., image resolution)
        """
        if stage not in self.timings:
            self.timings[stage] = []
        
        record = TimingRecord(
            stage_name=stage,
            duration_ms=duration_ms,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.timings[stage].append(record)
        self.current_run[stage] = record
    
    @contextmanager
    def time_stage(self, stage: str, metadata: Dict = None):
        """
        Context manager to time a synchronous stage.
        
        Args:
            stage: Pipeline stage name
            metadata: Additional metadata
            
        Yields:
            StageTimer instance
        """
        timer = StageTimer(stage, metadata)
        try:
            with timer:
                yield timer
        finally:
            if timer.record:
                self.record_timing(stage, timer.duration_ms, metadata)
    
    def get_stage_statistics(self, stage: str) -> Optional[StageStatistics]:
        """
        Get statistical summary for a pipeline stage.
        
        Args:
            stage: Pipeline stage name
            
        Returns:
            StageStatistics or None if no data
        """
        if stage not in self.timings or not self.timings[stage]:
            return None
        
        durations = [r.duration_ms for r in self.timings[stage]]
        
        return StageStatistics(
            stage_name=stage,
            mean_ms=statistics.mean(durations),
            median_ms=statistics.median(durations),
            std_dev_ms=statistics.stdev(durations) if len(durations) > 1 else 0.0,
            min_ms=min(durations),
            max_ms=max(durations),
            sample_count=len(durations),
            total_time_ms=sum(durations)
        )
    
    def get_all_statistics(self) -> Dict[str, StageStatistics]:
        """Get statistics for all stages"""
        return {
            stage: self.get_stage_statistics(stage)
            for stage in self.PIPELINE_STAGES
            if self.get_stage_statistics(stage) is not None
        }
    
    def get_current_run_summary(self) -> Dict[str, float]:
        """Get timing summary for the most recent run"""
        return {
            stage: record.duration_ms
            for stage, record in self.current_run.items()
        }
    
    def new_run(self):
        """Start a new profiling run"""
        self.current_run = {}
        self._run_count += 1
    
    def get_resolution_analysis(self) -> Dict[str, Dict]:
        """
        Analyze performance by image resolution.
        
        Returns:
            Dictionary mapping resolution to timing statistics
        """
        resolution_data: Dict[str, List[float]] = {}
        
        for stage, records in self.timings.items():
            if stage == 'total':
                for record in records:
                    resolution = record.metadata.get('resolution', 'unknown')
                    if isinstance(resolution, tuple):
                        resolution = f"{resolution[0]}x{resolution[1]}"
                    
                    if resolution not in resolution_data:
                        resolution_data[resolution] = []
                    resolution_data[resolution].append(record.duration_ms)
        
        return {
            resolution: {
                'mean_ms': statistics.mean(times),
                'median_ms': statistics.median(times),
                'min_ms': min(times),
                'max_ms': max(times),
                'sample_count': len(times)
            }
            for resolution, times in resolution_data.items()
        }
    
    def export_results(self, filepath: str = None) -> Dict:
        """
        Export all profiling results.
        
        Args:
            filepath: Optional path to save JSON results
            
        Returns:
            Dictionary with all results
        """
        stats = self.get_all_statistics()
        
        export_data = {
            'summary': {
                stage: {
                    'mean_ms': s.mean_ms,
                    'median_ms': s.median_ms,
                    'std_dev_ms': s.std_dev_ms,
                    'min_ms': s.min_ms,
                    'max_ms': s.max_ms,
                    'sample_count': s.sample_count
                }
                for stage, s in stats.items()
            },
            'resolution_analysis': self.get_resolution_analysis(),
            'raw_timings': {
                stage: [
                    {
                        'duration_ms': r.duration_ms,
                        'timestamp': r.timestamp,
                        'metadata': r.metadata
                    }
                    for r in records
                ]
                for stage, records in self.timings.items()
            },
            'total_runs': self._run_count
        }
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        return export_data
    
    def print_summary(self):
        """Print a formatted summary of all statistics"""
        print("\n" + "=" * 70)
        print("PERFORMANCE PROFILING SUMMARY")
        print("=" * 70)
        
        stats = self.get_all_statistics()
        
        # Header
        print(f"{'Stage':<15} {'Mean (ms)':<12} {'Median (ms)':<12} {'Std Dev':<12} {'Min':<10} {'Max':<10} {'N':<5}")
        print("-" * 70)
        
        for stage in self.PIPELINE_STAGES:
            if stage in stats:
                s = stats[stage]
                print(f"{s.stage_name:<15} {s.mean_ms:<12.2f} {s.median_ms:<12.2f} "
                      f"{s.std_dev_ms:<12.2f} {s.min_ms:<10.2f} {s.max_ms:<10.2f} {s.sample_count:<5}")
        
        print("=" * 70)
        
        # Resolution analysis
        res_analysis = self.get_resolution_analysis()
        if res_analysis:
            print("\nPERFORMANCE BY RESOLUTION:")
            print("-" * 50)
            for resolution, data in sorted(res_analysis.items()):
                print(f"  {resolution}: Mean={data['mean_ms']:.2f}ms, "
                      f"Median={data['median_ms']:.2f}ms (n={data['sample_count']})")
        
        print()
    
    def reset(self):
        """Clear all timing data"""
        self.timings = {stage: [] for stage in self.PIPELINE_STAGES}
        self.current_run = {}
        self._run_count = 0


class PipelineInstrumentor:
    """
    Instruments the MangaTranslator pipeline for timing measurements.
    
    This class provides hooks that can be integrated with the existing
    manga_translator pipeline to capture timing data.
    """
    
    def __init__(self, profiler: PerformanceProfiler = None):
        self.profiler = profiler or PerformanceProfiler()
        self._stage_start_times: Dict[str, float] = {}
    
    def stage_start(self, stage: str):
        """Mark the start of a pipeline stage"""
        self._stage_start_times[stage] = time.perf_counter()
    
    def stage_end(self, stage: str, metadata: Dict = None):
        """Mark the end of a pipeline stage and record timing"""
        if stage in self._stage_start_times:
            duration_ms = (time.perf_counter() - self._stage_start_times[stage]) * 1000
            self.profiler.record_timing(stage, duration_ms, metadata)
            del self._stage_start_times[stage]
    
    def create_progress_hook(self, metadata: Dict = None) -> Callable:
        """
        Create a progress hook compatible with MangaTranslator.
        
        Args:
            metadata: Metadata to attach to timings (e.g., image resolution)
            
        Returns:
            Progress hook function
        """
        stage_mapping = {
            'detection': 'detection',
            'ocr': 'ocr',
            'translating': 'translation',
            'inpainting': 'inpainting',
            'render': 'rendering'
        }
        
        def progress_hook(stage: str, finished: bool):
            mapped_stage = None
            for key, value in stage_mapping.items():
                if key in stage.lower():
                    mapped_stage = value
                    break
            
            if mapped_stage:
                if not finished:
                    self.stage_start(mapped_stage)
                else:
                    self.stage_end(mapped_stage, metadata)
        
        return progress_hook


# Utility function for quick timing
def time_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Time a synchronous function execution.
    
    Returns:
        Tuple of (result, duration_ms)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms


async def time_async_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Time an asynchronous function execution.
    
    Returns:
        Tuple of (result, duration_ms)
    """
    start = time.perf_counter()
    result = await func(*args, **kwargs)
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms
