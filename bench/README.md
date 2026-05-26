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

## Results

### Permuted-Digits (5 sequential tasks)

```
                  final avg     BWT     sec
flatmem               65%      -10%    28
sklearn-mlp           43%      -67%     2
```

MLP T0 accuracy 97% → 11% after 4 tasks. Flatmem 83% → 65%.

![permuted](permuted_digits_matrix.png)

### Split-Digits (5 binary tasks: {0,1}, {2,3}, {4,5}, {6,7}, {8,9})

```
                  final avg     BWT     sec
flatmem               87%       -3%    5.7
sklearn-mlp           20%     -100%    0.6
```

**MLP catastrophically forgets every prior task**: T0 99% → 0% after T1.
After all 5 tasks, MLP only knows T4 (the most recent). All others = 0%.

**Flatmem holds all 5**: T0 97% → 93%, final-avg 87%, BWT -3%.

![split](split_digits_matrix.png)

BWT (Backward Transfer): mean drop on prior tasks after later training.
Strongly negative = forgetting. Near zero = no forgetting.

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
