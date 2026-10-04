# Reproducing the paper with chosen parameters fixed

This workflow rebuilds all nine paper tables from fresh OOF and KDDTest+ probabilities. It fixes the previously chosen focal/scaling parameters. It does not repeat their selection searches. All scientific settings are in `paper_config.json`; the neural architectures, feature ordering, losses and guaranteed-batch implementation remain in the original model modules.

## Environment

Use Python 3.12; the recorded runs used Python 3.12.7. Create a clean environment and activate it using the command appropriate for your operating system. For example, on Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r reproduction/requirements.txt
```

For NVIDIA GPUs on Linux or WSL, install `reproduction/requirements-gpu.txt` instead. GPU execution requires a compatible NVIDIA driver. CPU execution requires no CUDA allocation. Platform installation guidance is available in the [official TensorFlow installation instructions](https://www.tensorflow.org/install/pip).

The requirements pin TensorFlow 2.16.2, Keras 3.13.2 and the recorded numerical/plotting dependencies. They are scientific dependency pins, not a complete operating-system or transitive-environment lock. CTGAN, PyTorch and XGBoost are not needed for this paper workflow. TensorFlow's platform support determines which machines can install this environment; the code does not require a specific GPU model or cluster scheduler.

## Inputs

Place the raw `KDDTrain+.txt` and `KDDTest+.txt` files under `data/`, or pass `--data-dir PATH`. The review archive built with `--include-data` supplies both files. The entry point verifies these SHA-256 values before fitting:

| Input | Rows | SHA-256 |
|---|---:|---|
| KDDTrain+.txt | 125973 | `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95` |
| KDDTest+.txt | 22544 | `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84` |

Class order is `[DoS, Probe, R2L, U2R, Normal]`. The existing taxonomy maps `httptunnel` and `worm` to R2L. The existing canonical encoder produces 121 features. Each OOF preprocessor is fitted on its training fold only; final preprocessing is fitted on all KDDTrain+ and only transforms KDDTest+.

## Commands

Run from the repository root. The script also works from another current directory when invoked by its absolute path; all defaults resolve relative to the checkout.

```bash
# Print the full plan without fitting models
python reproduction/reproduce.py --stage plan --device cpu

# Automatically use available TensorFlow GPUs, falling back to CPU
python reproduction/reproduce.py --stage all --device auto --output-dir runs/paper

# Use two logical GPUs from your allocated CUDA_VISIBLE_DEVICES
python reproduction/reproduce.py --stage all --device gpu --gpus 0 1 --output-dir runs/paper_two_gpu

# Explicit CPU execution; one independent fit at a time by default
python reproduction/reproduce.py --stage all --device cpu --output-dir runs/paper_cpu
```

`--gpus` accepts logical IDs starting at zero within the inherited allocation. `--workers N` limits GPU concurrency or requests N CPU workers. There is no four-GPU requirement or cap. Each GPU runs one independent fit at a time, rather than splitting a model's batch across GPUs. The global batch size remains 256; four folds, three seeds, 25 epochs and the one-example-per-rare-class guarantee remain unchanged.

Historical `_4gpu.py` filenames are retained for source lineage; worker counts and device choices come from the runtime arguments.

Run individual stages when convenient:

```bash
python reproduction/reproduce.py --stage oof --device gpu --gpus 0 1 --output-dir runs/paper
python reproduction/reproduce.py --stage final --device gpu --gpus 0 1 --output-dir runs/paper
python reproduction/reproduce.py --stage tables --output-dir runs/paper
python reproduction/reproduce.py --stage compare --output-dir runs/paper
```

The native controllers validate completed fits before resuming them. Run one launcher at a time against each output directory. Use separate output directories for independent repetitions and CPU/GPU comparisons. Outputs are isolated from the repository's historical `results/`. Changing source code or scientific settings creates new experiment identities; choose a new output directory to keep these experiments separate.

The entry point records invocation commands, Python/platform/package versions, runtime allocation, source hashes, configuration hash and dataset hashes. Native fit records also record completed epochs, model parameter counts and prediction hashes. `tables` requires complete probabilities and matching scientific settings, and refuses ambiguous distinct test artifacts rather than choosing the best test result.

## Small execution check

```bash
python reproduction/reproduce.py --stage smoke --device cpu --output-dir runs/smoke_cpu
```

This samples 20 real training examples and five real test examples per class, then performs sixteen one-epoch fits: Conv2D focal-only and focal-plus-batching OOF workers, plus baseline/focal-plus-batching final workers for all four backbones. It checks worker execution and parameter counts. A successful smoke check is not a reproduction of the reported paper numbers.

Unit checks, which do not train models:

```bash
python -m unittest discover -s tests -p 'test_experiment_runtime.py'
python -m unittest discover -s tests -p 'test_paper_reproduction.py'
python -m unittest discover -s tests -p 'test_variant_specific_score_scaling.py'
```

The checks already completed for this package, and their limits, are recorded in [VALIDATION.md](VALIDATION.md).

## Fixed scientific protocol and outputs

There are four underlying training regimes per backbone: baseline, focal-only, batching-only and focal-plus-batching. Four folds × three seeds × four regimes × four backbones yield 192 OOF fits. Training those regimes on all KDDTrain+ yields another 48 fits. Raw and scaled decisions share the same networks, producing all eight factorial cells. Scaling divides R2L/U2R scores by their respective frozen coefficients and then takes argmax.

| Backbone | Beta | Gamma | Parameters |
|---|---:|---:|---:|
| Conv2D | 0.99 | 0.50 | 109381 |
| Conv1D | 0.99 | 0.25 | 109797 |
| Transformer | 0.99 | 0.75 | 110661 |
| MLP | 0.99 | 0.25 | 99845 |

The sixteen coefficient pairs in `paper_config.json` come from audited selection `ea63ca7e718d`. The OOF fitting commands use only the chosen beta/gamma and unit coefficients; the table exporter applies the fixed pairs without searching new values. The final runner also reads the same coefficient configuration for its own scaled summaries.

`OUTPUT/paper_tables/` contains these tables in CSV and LaTeX:

1. Five-class composition.
2. Backbone parameter counts and inductive biases.
3. Chosen focal parameters.
4. Chosen score coefficients.
5. OOF Rare Macro-F1 by architecture/configuration.
6. KDDTest+ Rare Macro-F1 by architecture/configuration.
7. Test architecture mean, minimum, range and SD.
8. Secondary test metrics.
9. Pairwise and three-way factorial interactions.

The parameter tables describe the fixed settings; they do not independently establish the validity of their selection. `seed_metrics.csv`, `summary_unrounded.csv` and `provenance.json` preserve calculations and source hashes. Seed and architecture SD use the sample denominator n−1, and aggregates use unrounded values with equal seed/backbone weights. Preserve the manuscript captions and notes when using generated LaTeX fragments.

`reference_tables/` contains tables and unrounded summaries independently verified against the original server probabilities. `compare` produces 1,536 metric mean/SD differences against those summaries. It reports differences rather than imposing an arbitrary pass tolerance.

## What reproducibility establishes

Fixed seeds and pinned scientific dependencies preserve the experimental protocol. The original runs did not enable deterministic TensorFlow operations, so this configuration preserves that choice. Fresh fits can differ across CPU/GPU kernels, devices and environments; this package does not promise bitwise-identical predictions or exact fresh-run table values. Saved original predictions reproduced the verified reference metrics exactly. Full fresh-run agreement must be assessed from the new per-seed values and variation.

The original audit found two table-range rounding corrections: focal-plus-scaling is 10.45, and all-three-controls is 4.61 when computed from unrounded architecture means. Transformer batching's marginal contrast is 9.16 percentage points. The supplied references use the unrounded calculations. Evidence for the MLP focal choice shows a one-candidate rerun; an earlier full MLP search has not been established. Therefore this package makes no claim that it has reproduced the full hyperparameter-selection history. Negative pairwise test interactions are architecture-averaged; they are not negative in every backbone.

## Review archive

```bash
python reproduction/package_review.py --include-data --output dist/cnn_paper_reproduction.zip
```

The archive includes the paper entry point, required source modules, fixed settings, dependency requirements, tests, reference tables and optionally the two raw datasets. It excludes local audit files, server exports, caches, historical results, pretrained models, synthetic data, and source-copy adapters. `MANIFEST.json` gives SHA-256 and byte counts for every included file. Existing archives are not overwritten. Original thesis attribution is preserved in `LEGACY_README.md`.
