"""
Dataset loaders for benchmarks.

  load_digits_8x8     -- sklearn 8x8 toy digits (fast)
  load_mnist_28x28    -- real MNIST 28x28 via openml (cached after first call)
"""
import numpy as np


def load_digits_8x8(test_size=0.3, seed=42):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    d = load_digits()
    X = d.data.astype(np.float32) / 16.0
    y = d.target.astype(np.int64)
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def load_mnist_28x28(n_train=5000, n_test=1000, seed=42):
    """
    Real MNIST 28x28 via openml. Cached locally by sklearn after first call.
    Subsampled for tractable continual-learning runtime.
    Returns (X_tr, X_te, y_tr, y_te) -- X shape (N, 784), float32 [0,1].
    """
    from sklearn.datasets import fetch_openml
    data = fetch_openml('mnist_784', version=1, as_frame=False,
                         parser='liac-arff', cache=True)
    X = data.data.astype(np.float32) / 255.0
    y = data.target.astype(np.int64)
    rng = np.random.default_rng(seed)
    # stratified subsample across all 10 classes
    def subsample(X, y, n):
        idx = []
        per_class = n // 10
        for c in range(10):
            cls_idx = np.where(y == c)[0]
            pick = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
            idx.extend(pick.tolist())
        return X[idx], y[idx]
    # split into train + test halves (using openml's natural order: first 60K train)
    X_tr_full, X_te_full = X[:60000], X[60000:]
    y_tr_full, y_te_full = y[:60000], y[60000:]
    X_tr, y_tr = subsample(X_tr_full, y_tr_full, n_train)
    X_te, y_te = subsample(X_te_full, y_te_full, n_test)
    return X_tr, X_te, y_tr, y_te
