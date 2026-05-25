"""
flatmem.bench -- unified benchmark harness.

Provides:
  - BenchmarkAdapter: organism-agnostic train/eval interface
  - adapters.FlatmemAdapter, adapters.SklearnMlpAdapter
  - benchmarks (continual learning): permuted_mnist

Usage:
    from flatmem.bench import run_permuted_mnist
    results = run_permuted_mnist(n_tasks=5, seed=42)
"""
from .interface import BenchmarkAdapter
from .adapters import FlatmemAdapter, SklearnMlpAdapter
from .permuted_mnist import run_permuted_mnist

__all__ = [
    "BenchmarkAdapter",
    "FlatmemAdapter",
    "SklearnMlpAdapter",
    "run_permuted_mnist",
]
