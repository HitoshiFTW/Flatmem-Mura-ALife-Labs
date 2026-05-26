"""
Full 28x28 MNIST continual-learning runner.

Runs Permuted-MNIST and Split-MNIST with three adapters:
  - flatmem        (MultiRoleMemory, per-task role-binding)
  - sklearn-mlp    (sklearn MLPClassifier, partial_fit)
  - torch-cnn      (small CNN trained sequentially with SGD)

Subsamples MNIST for tractable CPU runtime; the catastrophic-forgetting
phenomenon shows clearly even at 5K-train-per-class scale.
"""
import time
import json
import numpy as np

from .adapters import FlatmemAdapter, SklearnMlpAdapter
from .torch_adapter import TorchCnnAdapter
from .datasets import load_mnist_28x28


def evaluate(adapter, X, y):
    if len(X) == 0: return 0.0
    return float(np.mean(adapter.predict(X) == y))


def make_permutations(n_tasks, n_features, seed=42):
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_features) for _ in range(n_tasks)]


def run_permuted_full_mnist(n_tasks=5, n_train=5000, n_test=1000, seed=42, verbose=True):
    X_tr, X_te, y_tr, y_te = load_mnist_28x28(n_train=n_train, n_test=n_test, seed=seed)
    perms = make_permutations(n_tasks, X_tr.shape[1], seed=seed)
    tasks_tr = [(X_tr[:, p], y_tr) for p in perms]
    tasks_te = [(X_te[:, p], y_te) for p in perms]

    adapters = [
        FlatmemAdapter(d=512, M=8192, M_rel=16384, k=32, seed=seed,
                       n_classes=10, n_tasks=n_tasks),
        SklearnMlpAdapter(hidden=(128,), n_classes=10, seed=seed,
                          lr=0.05, epochs_per_task=20),
        TorchCnnAdapter(n_classes=10, seed=seed, lr=0.01,
                        epochs_per_task=10, input_shape=(28, 28)),
    ]
    return _run_continual('Permuted-MNIST', adapters, tasks_tr, tasks_te, verbose)


def make_split_tasks_mnist(X_tr, y_tr, X_te, y_te):
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    tasks = []
    for c0, c1 in pairs:
        m_tr = (y_tr == c0) | (y_tr == c1)
        m_te = (y_te == c0) | (y_te == c1)
        tasks.append(((X_tr[m_tr], y_tr[m_tr]), (X_te[m_te], y_te[m_te]), (c0, c1)))
    return tasks


def run_split_full_mnist(n_train=5000, n_test=1000, seed=42, verbose=True):
    X_tr, X_te, y_tr, y_te = load_mnist_28x28(n_train=n_train, n_test=n_test, seed=seed)
    tasks = make_split_tasks_mnist(X_tr, y_tr, X_te, y_te)
    n_tasks = len(tasks)
    tasks_tr = [t[0] for t in tasks]
    tasks_te = [t[1] for t in tasks]

    adapters = [
        FlatmemAdapter(d=512, M=8192, M_rel=16384, k=32, seed=seed,
                       n_classes=10, n_tasks=n_tasks),
        SklearnMlpAdapter(hidden=(128,), n_classes=10, seed=seed,
                          lr=0.05, epochs_per_task=20),
        TorchCnnAdapter(n_classes=10, seed=seed, lr=0.01,
                        epochs_per_task=10, input_shape=(28, 28)),
    ]
    return _run_continual('Split-MNIST', adapters, tasks_tr, tasks_te, verbose)


def _run_continual(label, adapters, tasks_tr, tasks_te, verbose):
    n_tasks = len(tasks_tr)
    results = {}
    print(f'\n========== {label} ({n_tasks} tasks) ==========')
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
                print(f'  after T{i}: {row_str}')
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
            print(f'  final avg: {final_avg:.0%}   BWT: {bwt:+.0%}   ({elapsed:.0f}s)')
    return results


def print_summary(label, results):
    print(f'\n{"="*60}')
    print(f'{label} summary')
    print(f'{"="*60}')
    print(f'  {"adapter":<14s}  {"final avg":>10s}  {"BWT":>7s}  {"sec":>6s}')
    for name, r in results.items():
        print(f'  {name:<14s}  {r["final_avg"]:>10.1%}  '
              f'{r["bwt"]:>+7.1%}  {r["elapsed_s"]:>6.0f}')


if __name__ == '__main__':
    perm_res = run_permuted_full_mnist(n_tasks=5, n_train=5000, n_test=1000, seed=42)
    print_summary('Permuted-MNIST (28x28)', perm_res)
    with open('permuted_mnist_full_results.json', 'w') as f:
        json.dump(perm_res, f, indent=2)

    split_res = run_split_full_mnist(n_train=5000, n_test=1000, seed=42)
    print_summary('Split-MNIST (28x28)', split_res)
    with open('split_mnist_full_results.json', 'w') as f:
        json.dump(split_res, f, indent=2)
