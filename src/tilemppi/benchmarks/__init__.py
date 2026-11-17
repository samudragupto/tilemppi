"""Benchmark suite and publication-grade SVG chart generators for TileMPPI."""

from __future__ import annotations

from tilemppi.benchmarks.plots import generate_all_benchmark_svgs
from tilemppi.benchmarks.runner import (
    BenchmarkSuiteReport,
    format_benchmark_markdown,
    run_benchmark_suite,
)

__all__ = [
    "BenchmarkSuiteReport",
    "format_benchmark_markdown",
    "generate_all_benchmark_svgs",
    "run_benchmark_suite",
]
