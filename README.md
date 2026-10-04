# Rare-class intrusion detection: paper reproduction

This repository evaluates a complete 2×2×2 factorial of minority-guaranteed mini-batches, Class-Balanced Focal Loss, and class-specific score scaling on NSL-KDD. The backbones are Conv2D, Conv1D, a feature-token Transformer, and MLP. The primary endpoint is Rare Macro-F1, the mean of R2L and U2R F1.

The reviewer workflow is [reproduction/README.md](reproduction/README.md). It uses checked-in chosen parameters, four OOF folds, three training seeds, and an untouched KDDTest+ evaluation. It supports CPU execution and any available GPU count. Hardware changes concurrency; each worker fits one complete model with the same scientific settings.

Using Python 3.12 (the recorded experiment used 3.12.7):

```bash
python -m venv .venv
# Activate the environment using your operating system's activation command.
python -m pip install -r reproduction/requirements.txt
python reproduction/reproduce.py --stage plan --device cpu
python reproduction/reproduce.py --stage all --device auto --output-dir runs/paper
```

For NVIDIA GPU installation, use `reproduction/requirements-gpu.txt`. To select two allocated GPUs, pass `--device gpu --gpus 0 1`. Paths are configurable through `--data-dir`, `--output-dir`, and `--config`; no cluster account, server directory or source-copy patch is required.

The `all` stage performs 192 OOF fits and 48 final fits with chosen parameters fixed, then exports all nine tables as CSV and LaTeX. Parameters are recorded in [reproduction/paper_config.json](reproduction/paper_config.json). Regenerating the hyperparameter searches is a separate check.

Fresh fitting may differ numerically across hardware and libraries. The workflow records input/source hashes, settings, package versions and hardware allocation; verified original-run reference summaries are supplied for comparison. A smaller `--stage smoke` checks execution with sampled data and one epoch; its outputs are not paper results.

The paper workflow does not use CTGAN, synthetic data, saved image files, or the legacy pretrained model. The original thesis README, including its author attribution, is preserved in [reproduction/LEGACY_README.md](reproduction/LEGACY_README.md). Older exploratory programs remain in this repository; the review archive contains the files used by the documented paper workflow.
