"""
Split-MNIST style continual-learning benchmark.

10 classes split into 5 binary tasks:
    T0: {0,1}   T1: {2,3}   T2: {4,5}   T3: {6,7}   T4: {8,9}

Uses sklearn 8x8 digits.  Each task: binary classification within its class
pair, but evaluation is open-set against all 10 candidates.  Sequential
training across tasks; eval all prior tasks after each new task.

Standard continual-learning benchmark from EWC / GEM literature.
"""
import time
import json
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .adapters import FlatmemAdapter, SklearnMlpAdapter


def make_split_tasks(X, y, test_size=0.3, seed=42):
    """Return list of (X_tr, y_tr, X_te, y_te, class_pair) for 5 binary tasks."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tasks = []
    for c0, c1 in pairs:
        m_tr = (y_tr == c0) | (y_tr == c1)
        m_te = (y_te == c0) | (y_te == c1)
        tasks.append((X_tr[m_tr], y_tr[m_tr], X_te[m_te], y_te[m_te], (c0, c1)))
    return tasks


def evaluate(adapter, X, y):
    if len(X) == 0: return 0.0
    return float(np.mean(adapter.predict(X) == y))


def run_split_mnist(seed=42, verbose=True):
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target
    tasks = make_split_tasks(X, y, seed=seed)
    n_tasks = len(tasks)

    adapters = [
        FlatmemAdapter(d=512, M=4096, M_rel=8192, k=32, seed=seed,
                       n_classes=10, n_tasks=n_tasks),
        SklearnMlpAdapter(hidden=(128,), n_classes=10, seed=seed,
                          lr=0.05, epochs_per_task=50),
    ]

    results = {}
    for ad in adapters:
        if verbose: print(f'\n=== {ad.name} ===')
        ad.reset()
        acc_matrix = np.zeros((n_tasks, n_tasks))
        t0 = time.perf_counter()
        for i, (X_tr, y_tr, _, _, pair_i) in enumerate(tasks):
            if hasattr(ad, 'task'): ad.task = i
            ad.train(X_tr, y_tr)
            row = []
            for j, (_, _, X_te, y_te, pair_j) in enumerate(tasks):
                if hasattr(ad, 'task'): ad.task = j
                acc = evaluate(ad, X_te, y_te)
                acc_matrix[i, j] = acc
                row.append(acc)
            if verbose:
                row_str = '  '.join(f'T{j}{tasks[j][4]}: {a:.0%}'
                                    for j, a in enumerate(row))
                print(f'  after task {i} (digits {pair_i}): {row_str}')
        elapsed = time.perf_counter() - t0
        final_avg = acc_matrix[-1].mean()
        diag = np.diag(acc_matrix); last = acc_matrix[-1]
        bwt = float((last[:-1] - diag[:-1]).mean()) if n_tasks > 1 else 0.0
        results[ad.name] = {
            'acc_matrix': acc_matrix.tolist(),
            'final_avg': float(final_avg),
            'bwt': bwt,
            'elapsed_s': elapsed,
        }
        if verbose:
            print(f'  final avg: {final_avg:.0%}   BWT: {bwt:+.0%}   '
                  f'({elapsed:.1f}s)')
    return results


def print_summary(results):
    print('\n' + '=' * 64)
    print('Split-Digits continual-learning summary')
    print('=' * 64)
    print(f'  {"adapter":<14s}  {"final avg":>10s}  {"BWT":>7s}  {"sec":>6s}')
    for name, r in results.items():
        print(f'  {name:<14s}  {r["final_avg"]:>10.1%}  '
              f'{r["bwt"]:>+7.1%}  {r["elapsed_s"]:>6.1f}')


if __name__ == '__main__':
    res = run_split_mnist(seed=42, verbose=True)
    print_summary(res)
    with open('split_digits_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('\nResults dumped to split_digits_results.json')
