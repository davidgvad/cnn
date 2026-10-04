# Verification of server tables against the paper

Follow-up: the saved prediction arrays have now been independently re-scored on the server, and all 16 scaling searches repeated. See [PAPER_CLAIMS_AND_REVERIFICATION.md](PAPER_CLAIMS_AND_REVERIFICATION.md). The report below records the earlier audit stage.

The later complete metadata/seed-metric inspection is documented in [SERVER_EXPORT_AUDIT.md](SERVER_EXPORT_AUDIT.md). It confirms the primary and secondary results while identifying an MLP tuning-provenance gap and three small rounding corrections outside the primary result tables.

The user supplied terminal output from `/home/gvadzabd/cnn` containing three CSV files associated with frozen selection **`ea63ca7e718d`** and test evaluation **`d5c1c2bb3050`**. These pasted tables match the supplied manuscript's primary results and score-scaling parameters. The comparison found no mismatches.

The input transcript, its hash, all extracted rows, individual comparisons and calculated contrasts are recorded in [server_result_table_verification.json](server_result_table_verification.json). The paper values were previously extracted into [paper_rare_f1_cells.csv](paper_rare_f1_cells.csv), [paper_scaling_coefficients.csv](paper_scaling_coefficients.csv), and [paper_code_lineage.json](paper_code_lineage.json).

| Comparison | Result |
|---|---|
| OOF: 8 configurations × 4 backbones | All 32 Rare Macro-F1 means and all 32 seed SDs match |
| KDDTest+: 8 configurations × 4 backbones | All 32 Rare Macro-F1 means and all 32 seed SDs match |
| Architecture means and SDs in both primary tables | All 16 pairs match after rounding to paper precision |
| Selected R2L/U2R scaling coefficients | All 16 pairs match exactly as numeric values |
| Selected-coefficient CSV's OOF Rare-F1 means and SDs | All 16 pairs agree with the corresponding scaled OOF cells |
| Architecture-mean pairwise and three-way interactions | All 8 values match after rounding to paper precision |

The pasted test table reports baseline architecture-mean Rare Macro-F1 of **21.460976641215694%** and focal-plus-batching of **31.00392950976021%**. The difference is **9.542952868544516 percentage points**. These round to the abstract's 21.46% and 31.00%.

Calculations using the unrounded pasted architecture means give test marginal contrasts F = +1.8326526 pp, B = +6.1269087 pp, and S = +0.5759273 pp. Test pairwise interactions are F×B = −2.5102852 pp, F×S = −2.2772764 pp, and B×S = −0.8895066 pp. The three-way interaction is +1.8079692 pp. These support the paper's printed aggregate contrasts. The three negative pairwise values refer to architecture-mean interactions, rather than a claim that every backbone has negative interactions.

The unrounded OOF aggregates also resolve the earlier rounding ambiguity: F×B = −6.5340772 pp, F×S = +1.0943521 pp, B×S = +4.8080242 pp, and F×B×S = +10.1358434 pp. They round to the manuscript's −6.53, +1.09, +4.81, and +10.14.

The coefficient CSV explicitly records these OOF source-pointer families for all four architectures:

- Baseline: `<architecture>_baseline_cv_latest.json`.
- Focal only: `<architecture>_focal_stage1_latest.json`.
- Batching only: `<architecture>_batch_baseline_cv_latest.json`.
- Focal plus batching: `<architecture>_balanced_score_scaling_latest.json`.

For the selected coefficient rows, all four focal-plus-batching sources record the balanced-score-scaling pointer family. They do not record the alternative focal-batch pointer family. Following the manifests is still necessary to identify the actual OOF files and their producer protocols.

The filenames and CSV schema match the `select` and `evaluate` output paths in [tune_variant_specific_score_scaling.py](../src/tune_variant_specific_score_scaling.py). Together with the numeric agreement, this provides strong evidence for the final table-generation stage identified in [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md). It does not independently establish the exact historical script version or wrapper commands used for the underlying training runs.

This verification compares existing pasted results with the manuscript. No neural-network training or inference was run. Original server file bytes, recorded hashes, per-seed metric calculations, raw predictions, training settings and secondary metric tables remain to be checked. The pasted CSVs alone do not prove that a fresh training run would reproduce the same numbers.

**Subsequent provenance inspection:** the user supplied both top-level JSON manifests. Their exported hashes match the pointers, the evaluation links to the supplied selection, and their recorded scoring-script hash matches the local script exactly. The dataset hashes also match. The manifests identify forty-eight OOF and forty-eight final prediction sources. See [SERVER_RUN_LINEAGE.md](SERVER_RUN_LINEAGE.md) for the complete trace and the remaining individual producer/run metadata to inspect. The numerical table checks above remain unchanged.
