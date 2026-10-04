# Audit of the complete server metadata export

Follow-up: the saved prediction arrays have now been independently re-scored on the server, and all 16 scaling searches repeated. See [PAPER_CLAIMS_AND_REVERIFICATION.md](PAPER_CLAIMS_AND_REVERIFICATION.md). The report below records the earlier audit stage.

The paper's primary results, secondary metrics, selected parameters, and final training records form a consistent recorded pipeline. The eleven-file source core identified earlier is supported by the server evidence. There is one unresolved methods claim: the MLP focal OOF source used by the frozen selection evaluates a single setting, while the manuscript describes an eighteen-candidate search for every backbone. A direct read-only server search found no earlier full MLP focal-search protocol in the current repository's results.

The complete supplied export is preserved in [server_metadata_bundle.json](server_metadata_bundle.json). Its transcript path/hash, comparisons, checked settings and discrepancies are in [server_export_audit.json](server_export_audit.json). The seven CSV payloads are preserved byte-for-byte under [server_result_tables](server_result_tables). Existing project source, datasets and original results were not changed. No model training or inference was run.

## What the saved evidence establishes

| Evidence | Result |
|---|---|
| Requested metadata | All 80 records supplied; none missing |
| Metadata JSON bodies | All 80 reconstructed bodies match their exported SHA-256 hashes |
| Metadata hashes recorded by the frozen manifests | All 75 match the supplied records |
| Result CSV payloads | All seven match their exported hashes |
| OOF and test per-seed metric rows | 96 rows per partition, 192 total; source paths/hashes and coefficient choices agree with manifests |
| Per-architecture metric means and sample SDs | All 768 metric mean/SD pairs agree with the three-seed calculations |
| Raw test metric values | All 576 values agree between the 48 training records and final evaluation seed rows |
| Primary manuscript results | All 64 Rare Macro-F1 mean/SD pairs agree |
| Secondary manuscript metrics | All 56 entries agree after rounding to manuscript precision |
| Backbone parameter counts and selected focal/scaling parameters | Agree with the manuscript and saved records |

The headline test means are 21.460976641215694% for baseline and 31.00392950976021% for focal plus batching. Their difference is 9.542952868544516 percentage points. All published pairwise and three-way interaction-table entries agree with calculations from the unrounded saved means.

The mean/SD checks cover accuracy, MCC, Macro-F1, Macro-Recall, Rare Macro-F1, minimum minority recall, and R2L/U2R precision, recall and F1. Rare Macro-F1 also agrees with the mean of the two minority F1 scores in every seed row, and the two class F1 scores agree with their recorded precision/recall values.

These checks validate saved measurements and aggregation. They do not independently recompute metrics from the prediction arrays. A separate direct read-only SSH check confirmed that all 96 referenced probability files exist and that every file's current SHA-256 matches its frozen record. It read 143,230,648 bytes for checksumming, without decoding predictions or running a model. No probability payloads were copied into the local repository.

## Actual recorded training and preprocessing

All 48 final run records report:

- 25 requested and 25 completed epochs, batch size 256, and the recorded Adam defaults.
- Training on all KDDTrain+, with no validation during fitting, no checkpoint selection, and no KDDTest-based selection.
- No CTGAN and zero synthetic rows.
- The intended CE/focal loss and ordinary/guaranteed batching for their regime.
- Beta 0.99 and the paper's architecture-specific gamma whenever focal loss is active.
- One reserved example per minority class whenever guaranteed batching is active.
- The manuscript's total model parameter counts and the local frozen backbone configurations.

All five final cache records describe 125,973 training rows, 22,544 test rows, 121 predictors, the same feature order, and the adopted five-class counts. They record preprocessing fitted only on KDDTrain+, with test data transformed only. Their cached-array hashes are identical across the five final experiment groups.

All sixteen OOF producer protocols record four folds, split seed 0, training seeds 0–2, 25 epochs and batch size 256. Their fold index hashes agree across every architecture/regime. Their four per-fold feature-cache, feature-order and scaler-state signatures also agree across groups, providing evidence for common preprocessing and partitions. Each held-out fold has thirteen U2R records, and each corresponding training partition has thirty-nine.

The recorded software versions are TensorFlow 2.16.2, Keras 3.13.2, NumPy 1.26.4, pandas 2.2.2 and scikit-learn 1.5.1. The OOF protocols additionally record Matplotlib 3.9.2 and seaborn 0.13.2 where applicable. These recorded versions are more relevant to this experiment than the older repository dependency notes. The complete Python, CUDA, cuDNN and GPU hardware environment is not established by this export.

Deterministic operations are recorded as disabled in all final runs and OOF protocols. The supplied evidence supports reconstruction of the procedure and exact aggregation of saved outputs; it does not establish that newly trained networks would give bitwise-identical results.

## Which source files are supported by the evidence

All eleven core source files match the current server copy by full SHA-256. In total, 56 of the server's 57 source files match locally. The sole difference is `src/audit_validation_filtered_fusion_kddtest.py`, which belongs to the separate fusion work and is outside this paper's dependency closure. Its differing content has not been retrieved in this audit.

The core to preserve is:

| Stage | Files |
|---|---|
| Shared preprocessing, metrics and artifacts | `run_no_ctgan_model_ablation_4gpu.py` |
| Shared focal loss, sampler and model builders | `cnn_gan_foc.py`, `cnn_opt.py`, `cnn_opt_1d_4gpu.py` |
| Focal OOF candidates / selected-setting OOF generation | `tune_conv2d_focal_cv_4gpu.py`, `tune_conv1d_focal_cv_4gpu.py`, `tune_transformer_focal_cv_4gpu.py`, `tune_mlp_focal_cv_4gpu.py` |
| Shared baseline, batching and focal-plus-batching OOF engine | `tune_conv2d_score_scaling_cv_4gpu.py` |
| Final network fits | `run_final_baseline_vs_full_kddtest_4gpu.py` |
| Frozen per-regime scaling and final eight-cell tables | `tune_variant_specific_score_scaling.py` |

The full path/function mapping remains in [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md), and all 57 source classifications remain in [PAPER_SOURCE_FILES.md](PAPER_SOURCE_FILES.md). `run_final_single_enhancements_kddtest_4gpu.py` should also be retained to preserve the launcher associated with the combined final experiment family. The equivalent explicit shared-runner invocation cannot be distinguished from this wrapper by the saved protocol alone.

The OOF protocols provide historical source evidence beyond current-file similarity. I recomputed all sixteen dependency fingerprints using local file bytes and their original recorded server paths. Every fingerprint matches. These cover the training data and the relevant focal, preprocessing, loss, sampler and backbone modules in the recorded order.

The shared OOF engine additionally records a full hash of its training-worker and worker-command function source. Those function hashes identify older versions of the focal-plus-batching engine:

| Referenced F+B OOF groups | Matching worker function source in local history |
|---|---|
| Conv2D and Conv1D | `2f62108` |
| Transformer | `9ffbee4` |
| MLP | `03aa910` |

Baseline and batching-only OOF worker hashes match the current implementation. Every recorded OOF scoring-function hash also matches the current scoring functions. The older focal-plus-batching workers always used focal loss and guaranteed batching; the current worker adds conditional branches supporting ordinary CE and CE with batching. The recorded function hashes cover those functions, rather than attesting to every byte of a historical controller or an exact whole-repository commit.

The four server root shell scripts are older launchers. `file.sh` invokes `src/cnn_fin.py`; its two saved copies contain the incomplete command `src/cnn_fi`; `gan.sh` invokes CTGAN-only generation. Their contents do not launch the factorial pipeline. This does not establish which interactive or other scheduler commands launched the paper runs. Unused historical entry points can coexist with actively imported helpers in the same Python file; in particular, the focal loss in `cnn_gan_foc.py` remains required.

## MLP search provenance gap

The referenced focal protocols record:

| Backbone | Candidate pairs | Expected fits |
|---|---:|---:|
| Conv2D | 18 | 216 |
| Conv1D | 18 | 216 |
| Transformer | 18 | 216 |
| MLP | **1** | **12** |

The MLP protocol is `mlp_focal_stage1_0d0a1b34a405_protocol.json`. It explicitly records `betas: [0.99]`, `focal_gammas: [0.25]`, `expected_configurations: 1`, and `expected_fits: 12`. Its best-config JSON ranks that single candidate first. This record establishes evaluation of a fixed setting, not selection against eighteen alternatives.

I directly scanned all 43 files matching `*protocol*.json` recursively under `/home/gvadzabd/cnn/results`. Only this one MLP focal-search protocol was found, alongside its best-config and summary files. The older `/home/gvadzabd-laf/cnn/results` path referenced by the shell templates does not exist on this server. No full MLP focal-search record was found in those locations. An earlier search could have occurred elsewhere or its records could have been removed; the current evidence cannot establish that it did or did not happen.

The four referenced focal protocols therefore describe 660 planned fits, rather than the 864 associated with four eighteen-candidate searches. This is a methods/provenance discrepancy, not a discrepancy in the supplied final F1 values. If an earlier MLP search selected the fixed setting, its provenance should be recovered and linked. If no earlier search was performed, the manuscript should describe the MLP setting as fixed and state the different tuning budget, rather than claim the same eighteen-candidate search for every backbone. The full eight-cell MLP factorial results remain present.

## Three small rounding corrections

| Location | Manuscript | From unrounded saved means | Correct two-decimal display |
|---|---:|---:|---:|
| Architecture-summary range, Focal + scaling | 10.44 | 10.4450700890 | **10.45** |
| Architecture-summary range, All three controls | 4.60 | 4.6071129121 | **4.61** |
| Results narrative, Transformer B marginal contrast | +9.15 | +9.1556277760 | **+9.16** |

The two printed ranges agree with subtraction of the rounded primary-table cells. Using unrounded architecture seed means yields the values above. None of these rounding corrections changes the headline test result, the positive batching contrast in each backbone, or the reported aggregate interaction signs. No manuscript file was edited.

## Scope of the conclusion

There is high confidence in the core file mapping, the exact saved artifacts used for the manuscript's primary/secondary results, and the recorded final settings. Exact convenience-wrapper commands remain unproven, and the eighteen-candidate MLP tuning claim is not supported by the recovered records. Fresh-training reproduction has not been attempted. No files have been deleted, moved, or classified as never used during all prior development.
