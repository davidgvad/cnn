# Fresh reproduction of all nine tables with parameters fixed

**Reviewer workflow update:** The portable pipeline is now implemented directly in the codebase. Use [../reproduction/README.md](../reproduction/README.md) and `reproduction/reproduce.py` for new runs and reviewer packaging. The server-copy adapter and commands below document the earlier preparation and are not included in the review archive.

These commands retrain the networks while keeping the paper's chosen parameters fixed. They do not retest the hyperparameter choices. The original source and datasets were copied into a separate reproduction workspace; original results are not used as completed fits in that workspace.

## Commands on turing501

Run inside your allocation with two L40S GPUs. Pass the allocated logical GPU IDs as parameters; the launcher supports one to four GPUs, with two as its default. Each GPU runs one independent fit at a time; all four folds still run. The checked Python environment contains TensorFlow 2.16.2, Keras 3.13.2, NumPy 1.26.4, pandas 2.2.2, and scikit-learn 1.5.1, matching the recorded experiment versions.

```bash
export REPRO_PYTHON=/home/gvadzabd/anaconda3/bin/python3
REPRO_RUNNER=/home/gvadzabd/cnn/analysis/fixed_parameter_reproduction_20261003_2d17e4d4/reproduce_fixed_parameters.sh
REPRO_DIR=/home/gvadzabd/cnn_fixed_reproduction_01

# 192 fits: 4 backbones x 4 training regimes x 4 folds x 3 seeds
bash "$REPRO_RUNNER" "$REPRO_DIR" oof 0 1

# 48 fits: 4 backbones x 4 training regimes x 3 seeds
bash "$REPRO_RUNNER" "$REPRO_DIR" final 0 1

# Apply the fixed coefficients and export all nine tables
bash "$REPRO_RUNNER" "$REPRO_DIR" tables
```

Run the stages in order, proceeding after each succeeds. Alternatively, one command executes all three stages and stops if a stage fails:

```bash
bash "$REPRO_RUNNER" "$REPRO_DIR" all 0 1
```

The same `oof`, `final`, or `all` command resumes verified completed fits after an interruption. No `--rerun` flag is used. To start another completely fresh repetition, choose a new sibling directory such as `/home/gvadzabd/cnn_fixed_reproduction_02`.

To inspect every training plan without training:

```bash
bash "$REPRO_RUNNER" "$REPRO_DIR" plan 0 1
```

Do not launch concurrent copies against the same reproduction directory.

## Hardware options and original-code preservation

The original project's `src/` files remain unchanged. Most controllers already accept fewer GPUs. The Conv2D focal-only controller requires exactly four, so the launcher adapts its scheduling in the isolated copy to queue all four folds across the requested workers. A source-text check confirms that its fitting, preprocessing, metric and worker-command functions remain identical to the original.

For one GPU, use `all 0`; for four, use `all 0 1 2 3`. These change concurrency, without changing the batch size, epochs, folds or seeds. Logical IDs are resolved within inherited `CUDA_VISIBLE_DEVICES`, including scheduler-provided device UUIDs.

CPU mode is optional and uses a separate workspace:

```bash
REPRO_CPU_DIR=/home/gvadzabd/cnn_fixed_reproduction_cpu_01
bash "$REPRO_RUNNER" "$REPRO_CPU_DIR" plan cpu
bash "$REPRO_RUNNER" "$REPRO_CPU_DIR" all cpu
```

CPU mode hides all CUDA devices and runs one fit at a time. Its copied OOF controllers additionally permit zero visible GPUs and resolve the CPU worker to an empty CUDA device list. Only those device checks/resolvers change; fitting code remains identical. The final-fitting controller already has `--allow-cpu`, so it needs no source change. A TensorFlow installation and the experiment dependencies are still required. CPU execution can take substantially longer. Use a new workspace when switching between CPU and GPU to keep their artifacts separate.

Every adaptation records original/adapted hashes, preserved functions, and the exact source diff in the reproduction workspace's `analysis/gpu_scheduling_adaptation.json` and `analysis/hardware_adaptation.patch`. These files make the changes to copied controllers reviewable. GPU mode adapts one copied source file; CPU mode adapts five. No original model-building code is edited.

## Fixed settings

| Backbone | Beta | Gamma |
|---|---:|---:|
| Conv2D | 0.99 | 0.50 |
| Conv1D | 0.99 | 0.25 |
| Transformer | 0.99 | 0.75 |
| MLP | 0.99 | 0.25 |

All backbone settings remain those in the original code. The fitting budget is 25 epochs, batch size 256, training seeds 0–2, four stratified OOF folds with split seed zero, and one reserved example per minority class in guaranteed batches. CTGAN and synthetic observations are excluded. Deterministic-operations flags are omitted, matching the recorded experiment.

Focal-only OOF commands pass exactly one beta and one gamma to each Stage-1 driver. Baseline, batching-only, and focal-plus-batching OOF commands pass `--coefficient-values 1.0`, so they only produce raw predictions. The table exporter then applies the sixteen already-chosen coefficient pairs from the frozen selection `ea63ca7e718d`; it performs no scaling search. Final fitting trains only `baseline`, `focal_only`, `batch_only`, and `full`, totaling four networks per backbone/seed. Each network supplies both raw and scaled table entries.

The final fitting driver's own `full` summaries use older embedded scaling coefficients for some backbones. Use the exported `paper_tables/` tables for the paper comparison: the exporter reads the raw probabilities and applies the correct per-regime paper coefficients.

## Output tables

All tables are written in both CSV and LaTeX fragment formats beneath:

```text
/home/gvadzabd/cnn_fixed_reproduction_01/paper_tables/
```

| File stem | Paper table |
|---|---|
| `table_01_class_counts` | Five-class composition |
| `table_02_architectures` | Backbone parameter counts and inductive biases |
| `table_03_focal_parameters` | Chosen focal parameters |
| `table_04_score_parameters` | Chosen score-scaling pairs |
| `table_05_oof_results` | OOF Rare Macro-F1, seed mean/SD and architecture summaries |
| `table_06_kddtest_results` | Test Rare Macro-F1, seed mean/SD and architecture summaries |
| `table_07_architecture_summary` | Test mean, minimum, range and Architecture SD |
| `table_08_secondary_test` | MCC, Macro-F1, rare and class-specific F1/recall |
| `table_09_factorial_interactions` | OOF/test pairwise and three-way contrasts |

`seed_metrics.csv`, `summary_unrounded.csv`, and `provenance.json` preserve the underlying calculations and source hashes. Calculations use unrounded metrics, sample SD with the n−1 denominator, equal weighting of seeds/backbones, and division of rare-class scores by their fixed coefficients. Preserve the original manuscript captions and explanatory notes when using the LaTeX fragments.

The focal/scaling parameter tables record the chosen settings; their selection procedure is intentionally deferred. The results tables are recalculated from the fresh predictions. Original training was not deterministic, so fresh fitted values may differ; compare the full per-seed results and SDs when assessing reproduction.

## Checks completed before providing these commands

- All sixteen OOF-controller commands and the final-controller command passed actual server dry runs for two-GPU and CPU modes, without model fitting.
- Mocked scheduling checks cover one, two, three and four GPU workers, all twelve fold/seed fits, one concurrent fit per GPU, and failure handling. CPU device guards and CUDA allocation mapping are checked separately.
- The fixed-parameter exporter regenerated nine CSV tables and nine LaTeX fragments from the original frozen predictions.
- All 1,536 unrounded metric mean/SD values matched the earlier independent prediction verification.
- The deployed script hashes match the local source files.
- Fresh training has not been started by this preparation.

Dry runs validate command parsing and experiment plans. Full CPU/GPU training has not been performed by this preparation; successful model fitting must still be verified in your allocation.

[Launcher source](reproduce_fixed_parameters.sh), [hardware adapter](adapt_reproduction_gpu_workers.py), [fixed-parameter exporter](export_fixed_parameter_tables.py), [server setup](fixed_reproduction_server_setup.json), [scheduling checks](hardware_scheduling_checks.json), [hardware preflight evidence](fixed_reproduction_preflight/hardware_preflight.json), and [original-table preflight evidence](fixed_reproduction_preflight/validation_summary.json).
