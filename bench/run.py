"""CLI entry: python -m flatmem.bench.run [n_tasks] [seed]"""
import sys
from .permuted_mnist import run_permuted_mnist, print_summary

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    res = run_permuted_mnist(n_tasks=n, seed=s, verbose=True)
    print_summary(res)
