"""
Permuted-MNIST style continual-learning benchmark.

Uses sklearn 8x8 digits (faster than full 28x28 MNIST).  N tasks; each task
is the same dataset with a fixed random permutation of pixel positions.
Sequential training reveals catastrophic forgetting in neural networks
while a flat-memory substrate retains by construction.

Metrics:
  - Per-task accuracy AFTER each subsequent task (forgetting matrix)
  - Final average accuracy across all tasks (mean of last row)
  - Backward Transfer (BWT): average drop on prior tasks after later training
"""
import time
import json
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .adapters import FlatmemAdapter, SklearnMlpAdapter


def make_permutations(n_tasks, n_features, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_features) for _ in range(n_tasks)]


def permute(X, perm):
    return X[:, perm]


def evaluate(adapter, X, y):
    if len(X) == 0: return 0.0
    pred = adapter.predict(X)
    return float(np.mean(pred == y))


def run_permuted_mnist(n_tasks=5, seed=42, verbose=True):
    """
    Train each adapter sequentially on n_tasks permutations.
    After each task, eval on ALL prior tasks (including current).
    Return forgetting matrix per adapter.
    """
    data = load_digits()
    X = data.data.astype(np.float32) / 16.0
    y = data.target

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)

    perms = make_permutations(n_tasks, X.shape[1], seed=seed)
    tasks_tr = [(permute(X_tr, p), y_tr) for p in perms]
    tasks_te = [(permute(X_te, p), y_te) for p in perms]

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
        for i in range(n_tasks):
            if hasattr(ad, 'task'): ad.task = i
            ad.train(tasks_tr[i][0], tasks_tr[i][1])
            row = []
            for j in range(n_tasks):
                if hasattr(ad, 'task'): ad.task = j
                acc = evaluate(ad, tasks_te[j][0], tasks_te[j][1])
                acc_matrix[i, j] = acc
                row.append(acc)
            if verbose:
                row_str = '  '.join(f'T{j}: {a:.0%}' for j, a in enumerate(row))
                print(f'  after task {i}: {row_str}')
        elapsed = time.perf_counter() - t0
        # final-avg = mean of last row over tasks that have been trained
        final_avg = acc_matrix[-1].mean()
        # BWT = mean drop on prior tasks comparing diagonal (when first learned)
        #       vs last row (final eval)
        diag = np.diag(acc_matrix)
        last = acc_matrix[-1]
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
    print('Permuted-Digits continual-learning summary')
    print('=' * 64)
    print(f'  {"adapter":<14s}  {"final avg":>10s}  {"BWT":>7s}  {"sec":>6s}')
    for name, r in results.items():
        bwt = r['bwt']
        print(f'  {name:<14s}  {r["final_avg"]:>10.1%}  '
              f'{bwt:>+7.1%}  {r["elapsed_s"]:>6.1f}')
    print('\nBWT (Backward Transfer): negative = forgetting prior tasks.')
    print('Flat memory: BWT near zero = no forgetting.  '
          'MLP: BWT strongly negative = catastrophic forgetting.')


if __name__ == '__main__':
    res = run_permuted_mnist(n_tasks=5, seed=42, verbose=True)
    print_summary(res)
    with open('permuted_digits_results.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('\nResults dumped to permuted_digits_results.json')
