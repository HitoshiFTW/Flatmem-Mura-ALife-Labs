# flatmem.bench

Unified benchmark harness. Adapter-pattern comparison of flat memory vs
neural-network baselines on continual-learning tasks where transformers
structurally lose.

## Quick start

```bash
pip install flatmem scikit-learn matplotlib
python -m bench.run 5 42                  # 5 tasks, seed 42
python -m bench.plot permuted_digits_results.json
```

## Permuted-Digits (5 sequential tasks, seed 42)

```
                  final avg     BWT     sec
flatmem               65%      -10%    28
sklearn-mlp           43%      -67%     2
```

**MLP**: T0 accuracy 97% → 11% after 4 distractor tasks (catastrophic forgetting).
**Flatmem**: T0 accuracy 83% → 65% (substantial retention via per-task role-binding).

BWT (Backward Transfer): mean drop on prior tasks after later training.
Strongly negative = forgetting prior tasks. Near zero = no forgetting.

![forgetting matrix](forgetting_matrix.png)

## Architecture

```
bench/
├── interface.py            BenchmarkAdapter ABC (train/predict/reset)
├── adapters.py             FlatmemAdapter, SklearnMlpAdapter
├── permuted_mnist.py       sequential-task benchmark + forgetting metrics
├── plot.py                 matplotlib forgetting-matrix heatmap
└── run.py                  CLI entry point
```

## Extending

Add a new organism by subclassing `BenchmarkAdapter`:

```python
from flatmem.bench import BenchmarkAdapter

class MyAdapter(BenchmarkAdapter):
    name = "my-organism"
    def train(self, X, y): ...
    def predict(self, X): ...
    def reset(self): ...
```

Plug it into `permuted_mnist.run_permuted_mnist`'s adapter list.

Add a new benchmark by writing a runner that calls `adapter.train` per task,
`adapter.predict` per evaluation, and collects an accuracy matrix.

## Roadmap

- Pack 132: Split-MNIST (10 classes -> 5 binary tasks)
- Pack 133: full 28x28 MNIST scale + CIFAR-100 sequential
- Pack 134: GSM8K, HumanEval adapters
