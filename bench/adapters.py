"""
Adapters: wrap concrete organisms behind the BenchmarkAdapter contract.

  FlatmemAdapter      -- flat-memory MultiRoleMemory; superposition write,
                         cleanup recall against label candidates.
  SklearnMlpAdapter   -- sklearn MLPClassifier with partial_fit (online SGD).
                         Demonstrates catastrophic forgetting on sequential
                         tasks; canonical neural-network baseline.
"""
import numpy as np

from .interface import BenchmarkAdapter
from flatmem import MultiRoleMemory
from flatmem.encoders import random_projection_encoder


# ── flatmem ────────────────────────────────────────────────────────────────

class FlatmemAdapter(BenchmarkAdapter):
    """
    Encode example via random projection -> phasor HV.  Write under a
    per-TASK role ('class_T{task_id}') with label key as data.  Each task
    gets its own role -> orthogonal addresses -> different hard locations.
    Architectural use of role-binding for continual learning.

    Caller sets `adapter.task = N` before train/predict for task N.
    """
    name = "flatmem"

    def __init__(self, d=512, M=4096, M_rel=8192, k=32, seed=42,
                 enc_seed=7, n_classes=10, n_tasks=5, pca_components=None):
        self.d = d
        self.M = M; self.M_rel = M_rel; self.k = k; self.seed = seed
        self.enc_seed = enc_seed
        self.n_classes = n_classes
        self.n_tasks = n_tasks
        self.task = 0
        self.pca_components = pca_components   # None = raw random projection
        self._proj = None
        self._pca = None
        self._build()

    def fit_encoder(self, X):
        """Pre-fit PCA on training data. One-time before any continual task.
        Pack 134 fix: raw-pixel random projection collapses class structure
        at high input dim; PCA components restore separation cheaply."""
        if self.pca_components is None: return
        from sklearn.decomposition import PCA
        self._pca = PCA(n_components=int(self.pca_components),
                        random_state=self.seed)
        self._pca.fit(np.asarray(X))

    def _build(self):
        roles = tuple(f'class_T{i}' for i in range(self.n_tasks))
        self.mem = MultiRoleMemory(
            d=self.d, M=self.M, M_rel=self.M_rel, k=self.k, seed=self.seed,
            roles=roles,
        )
        self._class_keys = {c: self.mem.ck.key(str(c)) for c in range(self.n_classes)}

    def _encode(self, X, bandwidth=2.0):
        """
        Encode -> phasor HV. Two-stage normalization (Pack 134 v2):
          1. L2-normalize input rows -> ||x||=1 (data-range invariant)
          2. Project via N(0, bandwidth) so phase std == bandwidth
        Bandwidth ~2 rad is the sweet spot: enough structure preservation
        (low wrap), enough discriminative spread (avoids over-clustering).
        Works on ANY input dim. Validated 8x8 (90%) AND 28x28 (90%+) MNIST.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1: X = X[None, :]
        if self._pca is not None:
            X = self._pca.transform(X).astype(np.float32)
        in_dim = X.shape[1]
        if self._proj is None or self._proj.shape[1] != in_dim:
            rng = np.random.default_rng(self.enc_seed)
            self._proj = (rng.standard_normal((self.d, in_dim))
                          .astype(np.float32) * bandwidth)
        # L2-normalize per-row -> unit norm; then projection gives phase std
        # exactly equal to `bandwidth`, regardless of input dim or pixel range.
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms > 1e-9, norms, 1.0)
        Xn = X / norms
        phases = Xn @ self._proj.T
        return np.exp(1j * phases).astype(np.complex64)

    def _role(self):
        return self.mem.roles[f'class_T{self.task}']

    def train(self, X, y):
        # Auto-fit PCA on first train() call if configured (frozen thereafter)
        if self.pca_components and self._pca is None:
            self.fit_encoder(X)
        hvs = self._encode(X)
        ROLE = self._role()
        for hv, lbl in zip(hvs, y):
            addr = self.mem._bind(hv, ROLE)
            self.mem.sdm_rel.write(addr, self._class_keys[int(lbl)])

    def predict(self, X):
        hvs = self._encode(X)
        ROLE = self._role()
        preds = np.empty(len(X), dtype=np.int64)
        for i, hv in enumerate(hvs):
            addr = self.mem._bind(hv, ROLE)
            out = self.mem.sdm_rel.read(addr)
            best, bscore = 0, -9.0
            for c in range(self.n_classes):
                s = float(np.real(np.vdot(out, self._class_keys[c]))) / self.d
                if s > bscore: bscore, best = s, c
            preds[i] = best
        return preds

    def reset(self):
        self.task = 0
        self._proj = None
        self._build()


# ── sklearn MLP baseline ────────────────────────────────────────────────────

class SklearnMlpAdapter(BenchmarkAdapter):
    """
    sklearn MLPClassifier trained via partial_fit (online SGD).  Single
    network shared across all tasks -> catastrophic forgetting appears
    when tasks have non-overlapping input distributions.
    """
    name = "sklearn-mlp"

    def __init__(self, hidden=(128,), n_classes=10, seed=42,
                 lr=0.05, epochs_per_task=50):
        from sklearn.neural_network import MLPClassifier
        self.n_classes = n_classes
        self.epochs_per_task = int(epochs_per_task)
        self._init_args = dict(hidden_layer_sizes=hidden, max_iter=1,
                               learning_rate_init=lr, random_state=seed,
                               solver='sgd', warm_start=True)
        self._classes = np.arange(n_classes)
        self.reset()

    def train(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        # sklearn MLPClassifier.partial_fit rejects subsequent y subsets when
        # classes argument is fixed; pad each task with one zero-vector dummy
        # per missing class so y always contains all 10 labels.
        present = set(np.unique(y).tolist())
        missing = [c for c in self._classes if c not in present]
        if missing:
            dummies_X = np.zeros((len(missing), X.shape[1]), dtype=X.dtype)
            dummies_y = np.asarray(missing, dtype=y.dtype)
            X = np.vstack([X, dummies_X])
            y = np.concatenate([y, dummies_y])
        for _ in range(self.epochs_per_task):
            self.clf.partial_fit(X, y, classes=self._classes)

    def predict(self, X):
        return self.clf.predict(np.asarray(X))

    def reset(self):
        from sklearn.neural_network import MLPClassifier
        self.clf = MLPClassifier(**self._init_args)
