# elasticneuralnetwork (DIPRECATED)

A **PyTorch C++ extension** for dynamic neural networks — networks
that grow and prune themselves during training. Built on libtorch via
[`torch.utils.cpp_extension`](https://pytorch.org/docs/stable/cpp_extension.html);
the C++ side derives from `torch::nn::Module` and the Python side
exposes the standard `forward`, `parameters`, and `buffers` surface
so the modules drop into a normal `torch.optim.*` +
`torch.utils.data.DataLoader` training loop.

## What this is

Three modules exposed from C++ (each is a `torch::nn::Module` on the
C++ side; on the Python side they appear as pybind11 classes with a
PyTorch-compatible `forward` / `parameters` / `buffers` API):

- **`ReversibleLinear(in, out)`** — like `torch.nn.Linear`, plus an
  `active_mask` buffer that lets you soft-prune output nodes
  mid-training without erasing their weights. `compact()` hard-removes
  pruned rows at the end of training.
- **`ReversibleNetwork(config)`** — a stack of `ReversibleLinear`
  layers with the same insert/remove/compact API at the network level.
- **`AdaptiveConv2d(in_ch, out_ch, kernel_sizes)`** — differentiable
  architecture search over `{3, 5, 7}` kernel sizes plus a
  `grow_active_kernel()` mutation that expands the live kernel by one
  ring without changing forward output at the moment of growth.

Three observer classes (no anthropomorphic naming — see the project
plan):

- **`PlateauDetector`** — slope + curvature gate on a rolling
  scalar history.
- **`StabilityMonitor`** — emits `Stable` /
  `ExcessiveGrowthRisk` / `PathologicalGrowth` /
  `ExcessivePruningRisk` / `PathologicalPruning` / `Critical` based on
  the running counts of topology mutations.
- **`AdaptiveLRController`** — reward / penalty / reset over the cost
  and utilization histories.

And a `PhaseController` that drives a four-phase schedule
(`TopologyDiscovery` → `ConvergenceRateEstimation` → `AdaptiveTraining`
→ `FrozenArchitectureFinetuning`) using `torch::optim::Adam` under the
hood. Every threshold (LR floor, plateau window, growth-anomaly
threshold, …) is derived at the start of `fit()` from
`DatasetStatistics` (`feature_variance_mean`, `signal_to_noise_ratio`,
`label_entropy`, `effective_rank`, `fisher_separability`). No magic
numbers in the code.

## Install

```bash
pip install torch
pip install -e .
```

`setup.py` uses `torch.utils.cpp_extension.CUDAExtension` when CUDA is
present (`_can_use_cuda()` checks `cpp_extension.CUDA_HOME`) and falls
back to `CppExtension` otherwise. Headers under `csrc/include/enn/` are
on the include path so external C++ code can link against the
extension too.

## Quickstart — standard PyTorch training loop

`net.parameters()` returns ordinary `torch.Tensor` parameters, which
`torch.optim.Adam` accepts directly. `forward()` is the standard call.
A normal mini-batch loop with `torch.utils.data.DataLoader` works as
written.

```python
import torch
from elasticneuralnetwork import ReversibleNetwork, ReversibleNetworkConfig

cfg = ReversibleNetworkConfig()
cfg.input_features = 28 * 28
cfg.output_features = 10
cfg.hidden_sizes = [64, 32]
net = ReversibleNetwork(cfg)

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

X = torch.randn(256, 28 * 28)
y = torch.randint(0, 10, (256,))

for epoch in range(20):
    optimizer.zero_grad()
    logits = net.forward(X)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

# Soft-prune nodes 3 and 17 of the first layer (weights preserved,
# active_mask buffer flipped to zero).
net.layer(0).prune_nodes([3, 17])

# End-of-training: hard-erase pruned rows.
net.compact()
```

Use a `DataLoader` instead of full-batch input:

```python
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y), batch_size=32, shuffle=True
)
for x_batch, y_batch in loader:
    optimizer.zero_grad()
    loss = torch.nn.functional.cross_entropy(net.forward(x_batch), y_batch)
    loss.backward()
    optimizer.step()
```

Save and load the raw parameter tensors with `torch.save` /
`torch.load`:

```python
torch.save([p.detach().clone() for p in net.parameters()], "model.pt")
```

(The pybind11-exposed classes do not yet implement the full
`torch.nn.Module` Python API — `state_dict()` / `.to(device)` /
`eval()` / `train()` are not bound. Those are recorded as future
work; see `docs/developer-guide.md`.)

## Optional — the `ElasticNetwork.fit()` wrapper

If you don't want to write the training loop yourself,
`ElasticNetwork` composes a `ReversibleNetwork` and runs the four
phases for you:

```python
import torch
from elasticneuralnetwork import ElasticNetwork

X = torch.randn(64, 16)
y = torch.eye(4)[torch.randint(0, 4, (64,))]

model = ElasticNetwork(input_shape=(16,), output_size=4, seed=42)
result = model.fit(X, y)
print(result.epochs_completed, result.parameter_count_final)
preds = model.predict(X)
```

Under the hood this is the same `ReversibleNetwork` you'd build by
hand, driven by `PhaseController.fit(X, y)`. Every threshold the
controllers use is populated by the `derive_*` functions in
`csrc/training/data_driven_config.cpp` from
`compute_dataset_statistics(X, y)`.

## Layout

| Path | Role |
|---|---|
| `csrc/` | C++17 sources for the torch extension. |
| `csrc/include/enn/` | Public headers (`modules`, `controllers`, `training`). |
| `csrc/bindings/` | pybind11 bindings that expose the C++ `torch::nn::Module` subclasses to Python. |
| `elasticneuralnetwork/` | Python package: re-exports the C++ types and adds the `ElasticNetwork` convenience class. |
| `elasticneuralnetwork/benchmarks/` | MNIST + CIFAR-10 + UCI Adult / Covertype / Higgs benchmark CLIs and a `report.py` aggregator. |
| `tests/cpp/` | gtest unit tests for the controllers, modules, and `PhaseController`. |
| `tests/python/` | pytest tests for the bindings and the end-to-end fit. |

See each directory's `CLAUDE.md` for invariants and cross-references.

## Benchmarks

```bash
for d in mnist cifar10 tabular_adult tabular_covertype tabular_higgs; do
    python -m elasticneuralnetwork.benchmarks.$d --seed 42 --device cpu
done
python -m elasticneuralnetwork.benchmarks.report
```

Each script writes `results/<dataset>_seed<N>.csv` plus a curves PNG.
`report.py` aggregates them into a markdown summary and evaluates the
"ENN beats baseline on ≥ 2 of 5 datasets with ≥ 20% parameter
reduction" verification criterion.

## Out of scope

Distributed training, mixed precision (fp16 / bf16), quantization,
mobile deployment, pretrained model loading, ONNX export, recurrent /
attention layers.
