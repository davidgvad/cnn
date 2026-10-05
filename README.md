# Rare-class intrusion detection on NSL-KDD

This repository studies minority-guaranteed mini-batches, Class-Balanced Focal Loss, and class-specific score scaling across Conv2D, Conv1D, a feature-token Transformer, and MLP. The eight combinations form a complete 2×2×2 experiment. Rare Macro-F1 is the mean of R2L and U2R F1.

## Setup

Run commands from the repository root. Use Python 3.12, preferably the tested version 3.12.7.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproduction/requirements.txt
```

For NVIDIA GPUs on Linux, install `reproduction/requirements-gpu.txt` instead. The pinned environment uses TensorFlow 2.16.2 and Keras 3.13.2. GPU execution needs hardware and drivers compatible with that environment. CPU execution is also supported. See [TensorFlow's installation guide](https://www.tensorflow.org/install/pip).

Both raw datasets are included in `data/`. The launcher checks their hashes before training. Paths, devices, and worker counts are command-line arguments.

## Reproduce the tables with the paper's parameters

```bash
python reproduction/reproduce.py --stage plan --device cpu
python reproduction/reproduce.py --stage all --device cpu --workers 2 --output-dir runs/paper
```

This performs 192 OOF fits and 48 final fits, then exports all nine tables and compares them with the supplied references. Settings are in [reproduction/paper_config.json](reproduction/paper_config.json). Each fit uses 25 epochs and batch size 256, with four OOF folds and training seeds 0, 1, and 2.

To use two allocated GPUs, replace `--device cpu --workers 2` with `--device gpu --gpus 0 1`. GPU IDs are logical IDs within `CUDA_VISIBLE_DEVICES`. Each worker trains one model. Use `--device auto` to detect available TensorFlow GPUs or fall back to CPU.

## Repeat parameter selection

```bash
python reproduction/reproduce.py --stage search-all --device cpu --workers 2 --output-dir runs/search
python reproduction/reproduce.py --stage final --config runs/search/selected_config.json --device cpu --workers 2 --output-dir runs/search
python reproduction/reproduce.py --stage tables --config runs/search/selected_config.json --output-dir runs/search
python reproduction/reproduce.py --stage compare --output-dir runs/search
```

This repeats the focal grid, trains the remaining OOF regimes, selects score coefficients from OOF predictions, freezes the new configuration, and evaluates KDDTest+. New selections can differ from the historical choices. The [reproduction guide](reproduction/README.md) gives the search spaces, individual stages, outputs, and resume instructions.

Fresh training does not guarantee identical scores across hardware and software. Original saved predictions reproduced the reference metrics, and the historical full MLP search has been recovered and checked. A complete fresh repetition of all paper fits remains pending. [VALIDATION.md](reproduction/VALIDATION.md) records the completed checks and numerical corrections.

## Files

- `src/` contains the models, preprocessing, training, searches, and table exporter.
- `reproduction/` contains the launcher, fixed settings, requirements, and reference tables.
- `tests/` checks hardware selection, metrics, parameter selection, and artifact handling.
- `data/` contains KDDTrain+ and KDDTest+.

New outputs go under the chosen run directory. Original server prediction files are not bundled.

Shared code originated in **Network Intrusion Detection with CNN and CTGAN-Synthetic Data**, credited to **Leo Martinez III**, Fall 2024–Spring 2025. The present study uses four backbones without synthetic data.
