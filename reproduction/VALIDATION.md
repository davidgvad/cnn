# Checks completed

This file separates verification of saved results from fresh training. Original server prediction arrays and per-fit artifacts are not bundled. The compact reference tables are included for comparison.

## Original paper results

Frozen OOF selection `ea63ca7e718d` and test evaluation `d5c1c2bb3050` were checked against the supplied manuscript. Independent recalculation decoded 96 prediction files, recomputed 192 configuration-by-seed evaluations, and matched all 2,304 metric values against the saved seed CSVs. All sixteen score-scaling grids were checked independently, with 576 coefficient pairs per family. Main result checks passed.

The KDDTest+ architecture mean is 21.46097664% for baseline and 31.00392951% for focal plus batching, a gain of 9.54295287 percentage points. The table exporter regenerated all nine tables from the original predictions, matching all 1,536 unrounded metric mean/SD comparisons.

Three manuscript entries need rounding corrections when calculated from unrounded means:

- Focal plus scaling architecture range: **10.45**, rather than 10.44.
- All-three-controls architecture range: **4.61**, rather than 4.60.
- Transformer batching marginal contrast: **9.16** percentage points, rather than 9.15.

The supplied references use the corrected arithmetic. Negative pairwise test interactions describe the architecture average, not every backbone.

## Historical MLP focal search

The full Firebird search `3cd4803ca07a` was recovered. All 216 completed fit records and their fold-prediction hashes passed checks. All 54 pooled OOF arrays were reconstructed exactly from fold predictions, 6,264 metric comparisons matched saved records, and all eighteen candidate ranks were recomputed independently.

The winner was **β = 0.99, γ = 0.25**, with OOF Rare Macro-F1 **73.167367% ± 1.175557 percentage points** across the three seeds. Recorded completion times span August 4–5, 2026. The original table uses a later one-candidate rerun, `0d0a1b34a405`, at the same parameter pair, rather than the full-search predictions. This explains why the table's referenced protocol alone showed only twelve fits.

The full search recorded TensorFlow 2.20.0, Keras 3.13.1, NumPy 2.0.2, and scikit-learn 1.7.2. The table-producing rerun recorded TensorFlow 2.16.2, Keras 3.13.2, NumPy 1.26.4, and scikit-learn 1.5.1. The requirements target the latter environment.

## Fresh CPU/GPU comparison

Firebird job `916187` completed 24 CPU fits and 24 Quadro RTX8000 GPU fits on the same node in 13 minutes 40 seconds. Both phases used Python 3.12.7, TensorFlow 2.16.2, Keras 3.13.2, the same code snapshot, four folds, and three seeds. β was fixed at 0.99 and only γ = 0.25 and 0.50 were compared.

| Execution | γ = 0.25 | γ = 0.50 | Higher mean |
|---|---:|---:|---:|
| CPU | 72.27% ± 1.27 pp | 72.82% ± 0.31 pp | 0.50 |
| GPU | 72.89% ± 0.54 pp | 71.81% ± 2.45 pp | 0.25 |

Entries are OOF Rare Macro-F1 mean ± sample SD from the completed run summaries. Independent rescoring of these new prediction arrays remains pending. The GPU ranking agrees with the historical choice. This two-candidate comparison is not a new full search or an exact numerical reproduction. It supports sensitivity to execution conditions, without isolating the cause or establishing statistical superiority from three seeds. KDDTest+ was not accessed.

## Portable workflow

The cleaned repository passed 26 unit tests in Python 3.12.7 with the pinned dependencies. Checks cover independent metric calculations, search-to-evaluation handoff, artifact hashes, fresh-directory execution, and resume. CPU and two-GPU dry plans passed, including a copied checkout whose path contains spaces.

Sixteen one-epoch CPU smoke fits passed. They covered all four model parameter counts and both focal-only and guaranteed-batch worker execution. The retained model, loss, batching, and preprocessing definitions match their pre-cleanup syntax trees. Both datasets, the paper configuration, and all ten reference CSVs remain byte-for-byte unchanged.

The pinned environment passed real training steps on two Quadro RTX8000 GPUs. On RTX PRO 6000 Blackwell Server Edition GPUs, TensorFlow 2.16.2 detected the devices but failed while creating the model with `CUDA_ERROR_INVALID_HANDLE`. GPU-count configuration does not remove software/hardware compatibility requirements.

A complete fresh repetition of all 240 fixed-parameter paper fits and the new full-search launcher has not been completed. Original runs did not enable deterministic operations, and fixed seeds do not ensure bitwise equality across hardware or library versions. [TensorFlow's determinism documentation](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism) explains these limits. Fresh runs should be compared using unrounded per-seed values and their variation.
