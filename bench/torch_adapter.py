"""
PyTorch CNN adapter for the bench harness (lazy torch import).

Small 2-conv + 2-FC MNIST CNN. Sequential SGD training across continual
tasks; classical catastrophic-forgetting baseline.
"""
import numpy as np

from .interface import BenchmarkAdapter


class TorchCnnAdapter(BenchmarkAdapter):
    """
    Tiny CNN trained sequentially via SGD. Standard continual-learning
    baseline. Catastrophically forgets prior tasks without replay/EWC/LoRA.
    """
    name = "torch-cnn"

    def __init__(self, n_classes=10, seed=42, lr=0.01, momentum=0.9,
                 epochs_per_task=20, batch_size=128, input_shape=(28, 28)):
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for TorchCnnAdapter. "
                              "pip install torch")
        self.n_classes = n_classes
        self.seed = seed
        self.lr = lr; self.momentum = momentum
        self.epochs_per_task = epochs_per_task
        self.batch_size = batch_size
        self.input_shape = input_shape   # (H, W)
        self.reset()

    def _build(self):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        H, W = self.input_shape
        # final conv map: (B, 32, H/4, W/4) after 2 maxpool(2)s
        fc_in = 32 * (H // 4) * (W // 4)
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(fc_in, 64), nn.ReLU(),
            nn.Linear(64, self.n_classes),
        )
        return model

    def reset(self):
        import torch
        self.model = self._build()
        self.optim = torch.optim.SGD(self.model.parameters(),
                                     lr=self.lr, momentum=self.momentum)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, X, y):
        import torch
        H, W = self.input_shape
        X = np.asarray(X, dtype=np.float32).reshape(-1, 1, H, W)
        y = np.asarray(y, dtype=np.int64)
        Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
        n = len(X)
        self.model.train()
        for _ in range(self.epochs_per_task):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb, yb = Xt[idx], yt[idx]
                self.optim.zero_grad()
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optim.step()

    def predict(self, X):
        import torch
        H, W = self.input_shape
        X = np.asarray(X, dtype=np.float32).reshape(-1, 1, H, W)
        Xt = torch.from_numpy(X)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(Xt)
            return logits.argmax(dim=1).numpy()
