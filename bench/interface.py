"""
Organism-agnostic interface for continual / online benchmarks.

Every adapter (flatmem, sklearn MLP, HuggingFace transformer, etc.) implements
this contract.  Benchmarks loop over tasks and call the same methods.
"""
from abc import ABC, abstractmethod


class BenchmarkAdapter(ABC):
    """
    Adapter contract:
      train(X, y)                fit/update on a batch (online or full)
      predict(X)                 -> array of predicted class indices
      reset()                    -> erase all learned state (between independent runs)
      name                       -> human-readable identifier
    """

    name = "unnamed"

    @abstractmethod
    def train(self, X, y):
        """Train (or incrementally update) on examples X with labels y."""
        ...

    @abstractmethod
    def predict(self, X):
        """Return predicted class indices for examples X (numpy array)."""
        ...

    @abstractmethod
    def reset(self):
        """Erase all learned state; restore to fresh-init condition."""
        ...
