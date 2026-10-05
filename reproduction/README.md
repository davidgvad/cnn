# Reproduction commands

Install the environment as described in the [main README](../README.md). Run the commands below from the repository root. Use separate output directories for the fixed paper parameters and fresh selection searches.

## Scientific settings

KDDTrain+ has 125,973 rows and KDDTest+ has 22,544. Class order is `[DoS, Probe, R2L, U2R, Normal]`. Encoding produces 121 features. Each OOF preprocessor is fitted on its training fold only. Final preprocessing is fitted on all KDDTrain+ and only transforms KDDTest+.

Four folds use split seed 0. Training uses seeds 0, 1, and 2, 25 epochs, and batch size 256. Guaranteed batches reserve one example each from R2L and U2R. There is no synthetic data or test-based parameter selection.

| Backbone | Paper β | Paper γ | Parameters |
|---|---:|---:|---:|
| Conv2D | 0.99 | 0.50 | 109381 |
| Conv1D | 0.99 | 0.25 | 109797 |
| Transformer | 0.99 | 0.75 | 110661 |
| MLP | 0.99 | 0.25 | 99845 |

## Fixed paper parameters

```bash
python reproduction/reproduce.py --stage all --device cpu --workers 2 --output-dir runs/paper
```

Or run the stages separately:

```bash
python reproduction/reproduce.py --stage oof --device cpu --workers 2 --output-dir runs/paper
python reproduction/reproduce.py --stage final --device cpu --workers 2 --output-dir runs/paper
python reproduction/reproduce.py --stage tables --output-dir runs/paper
python reproduction/reproduce.py --stage compare --output-dir runs/paper
```

These commands use `paper_config.json`, including its sixteen score-coefficient pairs. There are four training regimes per backbone: baseline, focal-only, batching-only, and focal-plus-batching. Scaling reuses their probabilities, so all eight factorial cells require 192 OOF fits and 48 final fits. Scaling divides R2L/U2R scores by the frozen coefficients before argmax.

## Full parameter search

```bash
python reproduction/reproduce.py --stage search-all --device cpu --workers 2 --output-dir runs/search
```

This runs the following stages in order:

1. `search-focal` evaluates β ∈ {0.99, 0.999, 0.9999} and γ ∈ {0.25, 0.5, 0.75, 1, 1.5, 2} for each backbone. There are 18 candidates × 3 seeds × 4 folds × 4 backbones, or 864 fits. Candidates rank by mean OOF Rare Macro-F1, then Macro-F1, lower γ, and lower β. The selected settings go into `focal_config.json`.
2. `search-oof` reuses the winning focal-only predictions and trains baseline, batching-only, and focal-plus-batching for another 144 fits.
3. `search-scaling` evaluates 576 coefficient pairs for each backbone/training regime using the three seeds' pooled OOF predictions. It writes the frozen `selected_config.json`. This step trains no models.

Each coefficient uses the grid `{0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45, 1.60, 1.75, 1.90, 2.20, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00, 6.00, 7.00, 8.00, 10.00}`. Pairs rank by mean Rare Macro-F1, mean Macro-F1, lower `|log(k_R2L)| + |log(k_U2R)|`, lower Rare Macro-F1 SD, then lower R2L and U2R coefficients. Retention indicators are diagnostic and exclude no candidates.

The stages can also be run separately with the same output directory:

```bash
python reproduction/reproduce.py --stage search-focal --device cpu --workers 2 --output-dir runs/search
python reproduction/reproduce.py --stage search-oof --device cpu --workers 2 --output-dir runs/search
python reproduction/reproduce.py --stage search-scaling --output-dir runs/search
```

After selection, use the new configuration for final training and tables:

```bash
python reproduction/reproduce.py --stage final --config runs/search/selected_config.json --device cpu --workers 2 --output-dir runs/search
python reproduction/reproduce.py --stage tables --config runs/search/selected_config.json --output-dir runs/search
python reproduction/reproduce.py --stage compare --output-dir runs/search
```

Final training adds 48 fits, giving 1,056 fits for the complete search-and-evaluation workflow. KDDTest+ is used after selection is frozen. Fresh winners do not overwrite the paper configuration.

## Devices, checks, and resume

Replace `--device cpu --workers 2` with `--device gpu --gpus 0 1` for two allocated GPUs. Set `--workers` to limit concurrency. Historical `_4gpu.py` filenames do not require four GPUs. Use `--data-dir PATH`, `--output-dir PATH`, and `--config PATH` to change locations.

`--stage plan` prints the fixed workflow without fitting. Add `--dry-run` to a search stage to inspect its plan. Downstream plans use paper focal placeholders until the search winners exist.

For a small execution check:

```bash
python reproduction/reproduce.py --stage smoke --device cpu --output-dir runs/smoke
python -m unittest discover -s tests
```

The smoke stage uses sampled data and one epoch. It checks execution, not paper performance. GPU detection alone does not verify kernel compatibility. The pinned environment successfully trained on Quadro RTX8000 GPUs, but failed on RTX PRO 6000 Blackwell GPUs during testing.

To resume, repeat the command with the same settings and output directory. Completed fits are validated before reuse. An `all` run with a freshly selected configuration reuses focal-search predictions only when their matching files are present in that run directory. In a new directory, it trains all four OOF regimes with the selected parameters.

Run one launcher at a time per output directory. Use a new directory after changing source or scientific settings, since source hashes contribute to experiment IDs. Use a persistent terminal or your scheduler for long runs.

For Slurm, submit from the repository root with the environment active:

```bash
sbatch --partition=compute slurm/reproduce.sbatch --stage all --device cpu --workers 2 --output-dir runs/paper
sbatch --partition=gpu --gres=gpu:2 slurm/reproduce.sbatch --stage all --device gpu --gpus 0 1 --output-dir runs/paper_gpu
```

Adjust the partition, GPU type, CPU count, memory, and time limit for your cluster. `CNN_PYTHON` can specify the environment's Python executable. The batch job keeps running after the terminal disconnects.

## Outputs

`paper_tables/` contains CSV and LaTeX for class counts, architectures, focal parameters, score coefficients, OOF results, test results, architecture summaries, secondary metrics, and factorial interactions. Seed metrics, unrounded summaries, and provenance are also saved.

Means use equal seed/backbone weights and SD uses the sample denominator n−1. Aggregates use unrounded values. `compare` writes differences against `reference_tables/summary_unrounded.csv` without imposing an arbitrary pass tolerance. Invocation records contain commands, settings, source/data hashes, environment versions, and runtime allocation.

The historical runs did not enable deterministic TensorFlow operations. Fixed seeds alone do not guarantee identical training across devices or software versions. [TensorFlow documents the additional conditions for determinism](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism). See [VALIDATION.md](VALIDATION.md) for the original checks and the completed MLP comparison.
