# Paper claims and independent server reverification

**Portable reviewer workflow:** Use [../reproduction/README.md](../reproduction/README.md) for the hardware-configurable entry point and packaging. Source inventories and server audits below describe the original snapshot. [reviewer_portability_source_checks.json](reviewer_portability_source_checks.json) records the later controller/path changes and paper coefficient corrections; model building, preprocessing and fitting remain unchanged.

The supplied paper claims a descriptive comparison of three imbalance controls across four neural backbones on NSL-KDD. Its primary outcome is Rare Macro-F1: the equal-weight mean of R2L F1 and U2R F1. This is a rare-class detection score, not overall classification accuracy.

## What the paper claims

1. **Focal loss plus minority-guaranteed batching has the highest mean held-out Rare Macro-F1 across the four backbones.** The baseline is 21.46%, the combination is 31.00%, and the improvement is 9.54 percentage points.
2. **Batching is the most consistent component on KDDTest+.** Its marginal contrast is positive in each backbone, every backbone’s best configuration includes it, and the four highest architecture means are the four batching-enabled configurations. This claim concerns the observed test benchmark; batching is not uniformly beneficial in the OOF selection results.
3. **The best additional control depends on the backbone and distribution.** Conv2D and Transformer lead with focal plus batching; Conv1D leads with batching only; MLP leads with all three. The OOF winners differ.
4. **The leading test configurations have less between-backbone variation.** Focal plus batching reduces Architecture SD from 7.22 to 1.71 points and has the smallest range. Batching plus scaling has the highest minimum-backbone score, 28.85%.
5. **The rare-class gains mainly come from U2R detection.** For focal plus batching versus baseline, architecture-averaged U2R F1 rises from 21.80% to 38.99%, while R2L F1 rises from 21.12% to 23.02%. Macro-F1 and MCC also exceed baseline for that combination.
6. **The controls are not additive on average across architectures.** The test pairwise interaction contrasts are negative, while the three-way contrast is positive. The MLP has positive pairwise contrasts, so an unqualified statement that every backbone has negative interactions would be incorrect.

The reported means and SDs describe three training seeds and four architectures. They do not establish statistical significance, confidence intervals, superiority on every attack class, or generalization to other datasets. The OOF figures were also used for parameter selection and are not a nested-CV estimate.

## Recomputed architecture means

All entries below were calculated directly from saved prediction arrays on the server.

| Configuration | OOF Rare Macro-F1 (%) | Test Rare Macro-F1 (%) | Test Architecture SD (points) |
|---|---:|---:|---:|
| Baseline | 65.91 | 21.46 | 7.22 |
| Focal only | 70.36 | 26.14 | 3.07 |
| Batching only | 67.54 | 29.74 | 2.48 |
| Scaling only | 70.54 | 24.07 | 6.62 |
| Focal + batching | 60.39 | 31.00 | 1.71 |
| Focal + scaling | 71.02 | 25.57 | 4.95 |
| Batching + scaling | 71.91 | 30.56 | 2.00 |
| All three controls | 70.92 | 30.45 | 2.14 |

Architecture SD is the sample SD of the four architecture-specific seed means. It differs from the sample SD across seeds printed beside each individual backbone result.

## Best test configurations and batching contrasts

| Backbone | Best test configuration | Rare Macro-F1, mean ± seed SD (%) | Batching marginal contrast (points) |
|---|---|---:|---:|
| Conv2D | Focal + batching | 31.72 ± 3.79 | +9.69 |
| Conv1D | Batching only | 32.77 ± 2.92 | +5.20 |
| Transformer | Focal + batching | 32.23 ± 3.73 | +9.16 |
| MLP | All three controls | 31.86 ± 3.06 | +0.46 |

A positive marginal batching contrast averages four matched on/off differences across the other controls. It does not mean batching alone improves on baseline in every backbone: the MLP batching-only mean is 30.74%, below its 31.52% baseline.

The MLP all-three result is only about 0.02 points above batching plus scaling. The winner is an observed ranking of seed means, not evidence of a statistically established advantage.

## What the server run verified

- Decoded all **96 frozen prediction files**, using `allow_pickle=False`: 48 pooled OOF files and 48 final test files. All hashes match their frozen records.
- Independently reconstructed labels from the original KDDTrain+/KDDTest+ raw CSVs under the paper’s taxonomy. Every prediction file’s labels match the original row order.
- Checked complete OOF row indices, four fold IDs, common fold assignments, finite probabilities, probability row sums, and saved raw predictions against argmax.
- Recomputed **192 configuration-by-seed evaluations** and their five-class confusion matrices. All **2,304 metric values** agree with the original seed CSVs within a numerical tolerance of 1e-12.
- Recomputed all **64 primary mean/SD pairs**, the OOF/test architecture means and Architecture SDs, all **56 secondary table entries**, all **eight interaction-table values**, backbone winners and other qualitative rankings.
- Independently evaluated every one of the **576 scaling pairs** for each of **16 architecture/training-regime families**, across the three seeds. All **16 selected coefficient pairs** match the frozen selection. These searches use OOF data only, never test scores.
- Checked the 48 final run metadata records against fixed epochs, batch size, model parameter counts, selected focal parameters, batching quotas and absence of synthetic rows or test-based selection. Raw test metrics also agree with the final training records.

The verification imports NumPy and the Python standard library. It does not import experiment modules, TensorFlow, train a network or run model inference. It uses explicit frozen manifests, rather than current latest pointers. Original datasets, prediction files and experiment outputs are read only.

## Findings that still require attention

| Finding | Paper | Recomputed / recorded value |
|---|---|---|
| MLP focal search provenance | 18 candidates / 216 planned fits | Current referenced protocol: one candidate / 12 planned fits |
| Focal + scaling architecture range | 10.44 points | 10.45 points from unrounded means |
| All-three architecture range | 4.60 points | 4.61 points from unrounded means |
| Transformer batching marginal contrast | +9.15 points | +9.16 points from unrounded means |

The MLP final performance values and focal settings match. Its missing eighteen-candidate search evidence concerns selection history. The four referenced focal protocols describe 660 planned fits, compared with 864 stated in the manuscript. The earlier server audit found no full MLP search in the current results protocols; it cannot rule out removed files or a search elsewhere.

The two printed ranges can be recovered by subtracting already-rounded primary-table cells. Calculating from the unrounded seed means yields the corrections above.

For the abstract and conclusion, specify **“architecture-averaged pairwise interactions are negative on KDDTest+.”** The aggregate table is correct; all three MLP pairwise contrasts are positive.

## How to repeat the check on the server

The completed run is stored at:

```text
/home/gvadzabd/cnn/analysis/paper_reverification_20261003_866aebf4/results
```

After logging into `turing501`, run:

```bash
cd /home/gvadzabd/cnn
python3 analysis/paper_reverification_20261003_866aebf4/reverify_paper_results.py \
  --repo /home/gvadzabd/cnn \
  --output-dir analysis/paper_reverification_repeat_01
```

Use a new output-directory name for each repeat. The adjacent claims JSON is loaded automatically. This claims snapshot corresponds to the supplied manuscript SHA-256 `9ea3bcad10db05b45f6b599c4f51c393b90a5eb602dd5b971882306b129c2a68`; a revised manuscript requires a revised claims snapshot.

Exit status **2** means the verification completed and reported the four documented manuscript differences. Exit status **1** means an unexpected verification failure. Exit status **0** means all checked claims agree. The completed run has no unexpected failures.

For a faster check that applies the frozen coefficients without repeating their OOF searches, add `--skip-coefficient-grid`. No GPU is required.

## Local evidence files

- [Independent verifier](reverify_paper_results.py)
- [Claims extracted from the supplied paper](paper_verification_claims.json)
- [Machine-readable server verification](server_prediction_reverification/verification.json)
- [Recomputed seed metrics and confusion matrices](server_prediction_reverification/recomputed_seed_metrics.csv)
- [Recomputed per-backbone summaries](server_prediction_reverification/recomputed_summary.csv)
- [Recomputed architecture summaries](server_prediction_reverification/recomputed_architecture_summary.csv)
- [All independently ranked scaling candidates](server_prediction_reverification/recomputed_coefficient_rankings.csv)
- [Download hashes](server_prediction_reverification/download_integrity.json)
- [Server launch details](server_reverification_launch.json)

Fresh-training reproducibility remains a separate task. This completed run establishes that the saved predictions, frozen coefficient choices, scoring formulas and the principal manuscript results agree.
