# Repository and research audit

Reviewed 2026-10-03. Checkout: `5ce5872` (`Support legacy focal batch OOF protocols`). Scope: the supplied abstract and all 220 existing files outside `.git`, totaling 31,528,664 bytes. Review-generated files in `analysis/` are excluded from that original scope.

## Assessment

The newer code implements a coherent rare-class intervention study, with substantially better experimental discipline than the original CNN/CTGAN thesis code. Its most valuable contribution is the controlled comparison of selected exposure, loss, and decision policies across heterogeneous backbones. The implementation supports a complete eight-cell experiment by training four regimes and applying independently selected score scaling to each regime's saved probabilities.

The checkout is **not yet a self-contained reproduction of the abstract's numerical results**. It contains the experiment machinery, raw benchmark files, legacy results, and tests, but no completed modern OOF prediction bundles, frozen variant-specific coefficient manifest, or corresponding final eight-cell prediction/result tables. Consequently, I cannot independently confirm 31.00%, 21.46%, the four positive batching contrasts, the four architecture winners, or the signs of the reported test interactions. These are unverified here, not disproved.

The most consequential implementation issue is incomplete semantic validation between selected OOF models and final test artifacts. The most consequential interpretation issue is that the batching intervention changes replacement sampling and epoch coverage as well as guaranteeing minority presence. The most consequential statistical issue is that negative interactions in a nonlinear, bounded F1 metric do not establish a mechanistic conflict between stages.

The project also contains several distinct historical protocols. Their results and figures should be labeled separately; the README currently describes the original CTGAN thesis rather than the supplied factorial study.

The contribution is experimental organization and controlled cross-architecture evidence, rather than a new CNN architecture or a newly invented loss/sampling algorithm. That is a worthwhile empirical contribution if the complete tables and provenance support the claims; presenting it as algorithmic novelty or universal superiority of one stage would overstate what this design can establish.

## What was inspected and what was executed

All files were read as bytes and individually hashed. Every Python file was read, parsed into an AST, and syntax checked without importing training code. Critical paths were examined in depth: raw data preprocessing, class mapping, model definitions, focal loss, guaranteed batches, OOF reconstruction, coefficient selection, final inference, artifact reuse, stacking, and test-aware audits. All CSV rows and both raw benchmark datasets were read. All 73 PNGs passed chunk CRC checks; all 63 sample images were decoded and viewed in a contact sheet, and all ten methodology/result figures were visually inspected. The HDF5 signature and embedded model/training configuration were inspected; stored weights were not executed. The 68 bytecode files were inventoried and their headers inspected rather than executed as opaque code.

The existing 62 Python files contain 38,783 source lines and 37 test methods across five test modules. Syntax compilation passed for all 62. An attempted test discovery could not import any of the five modules because NumPy is absent from the available Python environment. TensorFlow, pandas, scikit-learn, h5py, matplotlib, and other numerical packages are also unavailable. **This is not a passing test run and not evidence that the test logic fails.** No GPU training, model inference, full numerical test suite, or fresh reproduction was performed.

Two dependency-free audit scripts are supplied. `audit_repository.py` reproduces file, data, syntax, and image checks. `metadata_lineage_probe.py` executes only an AST-extracted metadata validator against a deliberately inconsistent fixture and confirms its semantic validation gap. Its placeholder is not a probability array; the separate NPZ array checker is deliberately outside that probe's scope. No real experimental artifact is alleged invalid on that basis. Production source, data, and results were not modified.

See [the complete file review](FILE_BY_FILE_REVIEW.md), [inventory](file_inventory.csv), [source map](source_map.json), [dataset audit](dataset_audit.json), [audit summary](audit_summary.json), and [metadata probe](metadata_lineage_probe.json).

## Repository structure and protocol generations

| Generation | Principal entry points | Experimental meaning |
|---|---|---|
| Original thesis | `cnn_baseline.py`, `cnn_focal.py`, `cnn_gan_foc.py`, `cnn_fin.py` | Smaller Conv2D models; original full-data preprocessing, synthetic data and/or undersampling; legacy plots and model |
| Optimized comparisons | `cnn_opt.py`, `cnn_opt_1d_4gpu.py`, `compare_*`, `add_transformer_to_model_comparison.py` | Residual/grouped CNN variants, CTGAN comparisons, searches, multiple architectures; validation protocols differ between paths |
| Clean split-based ablations | `run_no_ctgan_model_ablation_4gpu.py`, `sweep_fixed_backbone_imbalance_4gpu.py`, `run_ctgan_amount_sweep_4gpu.py` | Real validation folds, training-only preprocessing; older 80/20 holdout selection rather than the abstract's four-fold protocol |
| Current factorial study | four `tune_*_focal_cv_4gpu.py` files, shared `tune_conv2d_score_scaling_cv_4gpu.py`, CV wrappers, final runner, `tune_variant_specific_score_scaling.py` | Four-fold OOF selection, three seeds, fixed epochs, four backbones, four training regimes, eight scored configurations |
| Subsequent fusion work | safe stack, robust calibrated super stack, validation-baseline selector, final fusion evaluators and audit | Additional ensemble research; some modules explicitly acknowledge post-test development and conditional rather than full nesting |

The generations reuse helper modules. Importing a legacy module's loss or backbone builder does not by itself reproduce that module's legacy `main()` protocol. Conversely, similarly named scripts do not necessarily implement the same split, selection objective, or training data policy.

## Current pipeline, from data to factorial cells

1. Load the 43-column NSL-KDD text files: 41 predictors, raw attack label, and difficulty. Remove difficulty and constant `num_outbound_cmds`; normalize `su_attempted=2` to 1; collapse attack names into five classes.
2. Use a fixed schema for three categorical fields and 37 remaining numeric fields. Encoding yields 121 scalar features: 37 numeric plus 3 protocol, 11 flag, and 70 service indicators. Sixteen numeric columns receive MinMax scaling. Remaining rate/binary fields retain their original representation.
3. Arrange the encoded fields in the prescribed numeric/basic/content/traffic/host order followed by protocol, flag, and service groups. Conv2D reshapes these 121 values to 11×11×1; Conv1D uses 121×1; MLP uses 121 values; Transformer treats them as 121 scalar feature tokens.
4. Create one fixed shuffled, stratified four-fold partition of KDDTrain+ with fold seed 0. Fit preprocessing on the three training folds only. For each candidate and training seed 0, 1, or 2, train for a fixed 25 epochs and predict the held-out fold. Held-out data are not passed to `fit()` for the modern focal/score-scaling CV protocol.
5. Reconstruct a full OOF prediction array in original row order for each seed. Calculate pooled OOF metrics per seed and then mean/sample SD across seeds. This is different from averaging four fold-level F1 scores.
6. Select focal parameters by OOF Rare Macro-F1, with the stated tie rules. The default focal grid is three beta values × six gamma values = 18 candidates, requiring 216 fold fits per architecture.
7. Produce OOF arrays for ordinary cross-entropy, focal only, guaranteed batching only, and focal plus batching. Independently search two positive score divisors for each architecture/training regime, using a shared pair across the three seeds and the same Rare-F1-first selection rule.
8. Freeze the selected coefficient manifest. Train four corresponding regimes on all KDDTrain+ using a preprocessor fitted only there. Save raw KDDTest+ probabilities. Apply both raw argmax and the regime-specific frozen scaling to each probability array.

| Focal F | Batching B | Scaling S | Configuration | Distinct fitted model? |
|---:|---:|---:|---|---|
| 0 | 0 | 0 | Baseline | Ordinary CE |
| 0 | 0 | 1 | Scaling only | Reuse ordinary CE |
| 1 | 0 | 0 | Focal only | Focal, ordinary shuffled batches |
| 1 | 0 | 1 | Focal + scaling | Reuse focal only |
| 0 | 1 | 0 | Batching only | CE, guaranteed batches |
| 0 | 1 | 1 | Batching + scaling | Reuse batching only |
| 1 | 1 | 0 | Focal + batching | Focal, guaranteed batches |
| 1 | 1 | 1 | Focal + batching + scaling | Reuse focal + batching |

The complete final design therefore requires 48 distinct fits at four architectures × four training regimes × three seeds and produces 96 seed-level scored cells, then 32 architecture/configuration means. Historical final-runner configurations can create additional redundant scaled-only fits, but they are unnecessary when raw probabilities are reused correctly.

The final runner's older five-variant table is not the complete factorial table. In particular, its `scaling_only` uses the old hardcoded full-model coefficients. The later variant-specific script is the relevant implementation because it retunes S independently for all four regimes, and uses the `full` artifact's **raw** probabilities as the F+B source.

## Data findings and cybersecurity meaning

| Collapsed class | ID | KDDTrain+ | KDDTest+ under repo mapping |
|---|---:|---:|---:|
| DoS | 0 | 45,927 | 7,458 |
| Probe | 1 | 11,656 | 2,421 |
| R2L | 2 | 995 | 2,887 |
| U2R | 3 | 52 | 67 |
| Normal | 4 | 67,343 | 9,711 |
| Total | | 125,973 | 22,544 |

All raw rows have 43 columns, and all numeric predictors examined are finite. The only constant raw numeric predictor in either file is index 19, `num_outbound_cmds`, which the current loader drops. Train categorical cardinalities are 3/70/11; test cardinalities are 3/64/11. Label CSV class counts agree with the raw-data class mapping.

U2R is 0.0413% of training data. A stratified four-fold holdout contains approximately 13 U2R validation examples and 39 training examples. This is a very small evidence base: a few changed predictions can materially move U2R F1. Batching repeatedly exposes existing examples; it cannot introduce a missing attack mechanism or subtype.

The benchmark shift is substantial. Training R2L is dominated by 890 `warezclient` rows, about 89.45% of that class, while `warezclient` is absent from KDDTest+. Test R2L is dominated by `guess_passwd` and `warezmaster`, together about 75.34%. Seventeen raw attack subtypes occur in test but not train. R2L prevalence rises from about 0.79% to 12.81%, and U2R prevalence from about 0.041% to 0.297% under this mapping. Generalization difficulties involve subtype and prevalence shifts in addition to training imbalance. Calling NSL-KDD a controlled generalization benchmark is reasonable; treating it as evidence of performance on today's arbitrary live networks is not supported by this repository. The [official NSL-KDD description](https://www.unb.ca/cic/datasets/nsl.html) explains its relationship to the older KDD/DARPA benchmark.

### Label taxonomy requires explicit documentation

The code places the 133 `httptunnel` test records in R2L. That is defensible: the [original MIT Lincoln Laboratory attack taxonomy](https://archive.ll.mit.edu/ideval/docs/attackDB.html) lists HttpTunnel among Remote-to-Local attacks. Some NSL-KDD literature instead assigns it to U2R, producing test supports of R2L=2,754 and U2R=200. This is **not an established mapping bug**. It is an important comparability issue because the paper's primary endpoint gives U2R half its weight. Publish the full raw-label mapping and supports, cite the adopted convention, and calculate a clearly labeled alternative-mapping sensitivity analysis from the same predictions. Do not silently compare Rare-F1 numbers across incompatible mappings.

### Feature overlap and ambiguity

Using equality of the 41 raw predictor fields, excluding attack label and difficulty, I found 16 extra duplicate-feature rows in train and 57 in test. Seven train feature groups and 54 test groups have conflicting **collapsed class labels**. There are 654 unique train/test shared feature vectors, occurring in 664 test rows (about 2.95%). These are precisely defined feature-level overlaps, not a claim that all those labeled complete records are exact duplicates or that the benchmark files are corrupted.

Random fold assignment can put identical predictor vectors in both training and validation. Conflicting labels imply some irreducible ambiguity for deterministic classifiers on those predictors. Retain the official split for the primary standard benchmark comparison, but document these findings and consider an auxiliary grouped-feature OOF analysis and test analysis excluding train-overlapping vectors. Any such sensitivity analysis changes the evaluation population and must be labeled accordingly.

### What the model sees

The CNN sees a reshaped table row, not a packet image. The Conv1D axis is feature order, not connection time. The Transformer models scalar encoded feature positions, not a sequence of packets and not one token per original categorical field. The sparse black/white sample images are rendered feature grids. Current training works directly with float arrays and does not load those PNGs. The available image directories contain only 21 illustrations each, although the label CSVs reference the entire historical image collection.

The repo is an offline five-class benchmark workflow. It does not provide a packet capture/feature extraction service, deployed inference API, alarm handling, or validation of latency and false alarms per traffic volume. Those would be separate development and evaluation tasks.

## Backbone audit

| Backbone | Main structure under paper defaults | Total model parameters | Trainable parameters |
|---|---|---:|---:|
| Conv2D | 11×11 grid; 64-filter 3×3 stem; pooling; 64-filter residual 3×3/1×1 block; pooling; dense 256; five-way softmax | 109,381 | 108,997 |
| Conv1D | 121 positions; 64-filter width-3 stem; pooling; residual width-3/1 block; pooling; dense 48; softmax | 109,797 | 109,413 |
| Feature Transformer | shared scalar embedding width 64 + learned position embedding; two blocks, four heads, feed-forward width 128; average pool; dense 512; softmax | 110,661 | 110,661 |
| MLP | 121 inputs; two width-256 dense/BN/ReLU/dropout layers; softmax | 99,845 | 98,821 |

These counts are analytically reconstructed from layer definitions and checked against the code's count formulas, not runtime `count_params()` in this environment. The Transformer-to-MLP total-parameter ratio is about 1.108, so “approximately parameter-matched” is reasonable if the actual counts are reported. “Exactly matched” would be false. Batch normalization accounts for the nontrainable state in the CNNs and MLP.

Matching parameters does not match FLOPs, activation memory, training time, depth, or inductive bias. A 121-token attention model and a small convolutional grid use very different computation. Report training time/compute if available and avoid attributing architecture differences only to architecture family. The 25-epoch budget and Adam defaults are held constant; this supports a fixed-budget comparison but does not prove that every architecture has converged equally.

The Conv2D grid introduces adjacency relationships from tabular ordering and row wrapping. The optimized feature order supplies a useful semantic grouping, but it remains a chosen representation. A feature-order sensitivity experiment would test whether a purported Conv2D advantage depends on this layout. MLP is a particularly useful control because it does not impose local adjacency. The feature-token Transformer should be described with its precise embedding/tokenization rather than equated automatically with a standard named tabular Transformer architecture.

## Intervention audit

### B: guaranteed minority exposure is a replacement-sampling policy

`BalancedBatchSequence` in `cnn_opt.py:518` guarantees at least one R2L and one U2R item when each class is present. It then fills the remaining 254 positions of a batch of 256 by sampling from **all training indices with replacement**. It has `ceil(N/256)` batches per epoch and a deterministic seed derived from training seed, epoch, and batch index.

At full-training defaults, there are 493 batches. Expected U2R draws per epoch are:

`493 × [1 + 254 × (52 / 125973)] = 544.69`.

Ordinary shuffled training sees each of the 52 U2R rows once per epoch. Thus the expected exposure multiplier is about 10.47. A randomly formed ordinary batch has about a 90% chance of containing no U2R record. This makes the intervention's motivation concrete.

The general replacement fill also means an ordinary nonquota row has probability approximately `1 − (1 − 1/N)^(493×254) ≈ 63%` of being sampled at least once in an epoch. Repetition and omitted majority rows are deliberate consequences of the policy. B changes rare exposure, overall sampling noise, duplicate presence within batches, majority coverage, and batch-normalization statistics. Its marginal contrast cannot isolate only the guarantee.

The same calculation gives about 1,482 expected R2L draws versus 995 baseline rows, an approximately 1.49-fold exposure multiplier. Thus the two rare classes receive very different relative exposure changes. The 63% figure is per-epoch majority coverage; it does not imply that 37% of majority rows remain unseen across the entire 25-epoch run, because the sampler changes its draws each epoch.

The full epoch's nominal sample draws remain close to baseline, so this is not simply an increase in optimizer steps. However, the final replacement batch is full sized whereas the ordinary last batch can be partial. In older mirrored multi-GPU paths the guarantee is global to a batch, not necessarily satisfied on every replica; modern independent one-GPU workers avoid that particular ambiguity.

For the paper, specify the complete policy, not only “minority-guaranteed mini-batches.” To isolate mechanisms in future work, compare this implementation against (a) replacement sampling without the quotas, (b) quotas plus majority traversal without replacement, and (c) ordinary sampling with a comparable rare-class sampling rate. These are mechanism controls, beyond the current factorial's scope.

### F: the implemented objective is mathematically coherent

`ClassBalancedFocalLoss` in `cnn_gan_foc.py:298` uses the true class's clipped softmax probability:

`L = −alpha_y (1 − p_y)^gamma log(p_y)`.

The class-balanced weights are proportional to `(1−beta)/(1−beta^n_c)` and normalized to sum to five. Modern paths calculate counts on each training fold, then on the full training partition for final fitting. This is a reasonable multiclass softmax form combining [focal modulation](https://arxiv.org/abs/1708.02002) and [effective-number weighting](https://arxiv.org/abs/1901.05555). It should be described as the implemented formulation; citing the class-balanced paper does not imply an exact reproduction of every option in its original training setup.

At beta=0.99 on full training, approximate weights in DoS/Probe/R2L/U2R/Normal order are `[0.7744, 0.7744, 0.7744, 1.9025, 0.7744]`. R2L and majority weights are almost equal because beta^995 is already small. The effective-number component therefore primarily distinguishes U2R at the winning default beta. Gamma still changes weighting by example difficulty for all classes.

F bundles class weighting and focal modulation. Its effect is not the isolated effect of “focusing” alone. Normalization and focal modulation also change overall gradient magnitude under fixed Adam settings. A decomposition would add class-balanced CE (gamma=0) and unweighted focal loss (all alpha equal). Their absence does not invalidate this three-policy factorial, but limits mechanistic claims about why F works.

The hardcoded frozen defaults are beta=0.99 for all architectures, with gamma 0.5/0.25/0.75/0.25 for Conv2D/Conv1D/Transformer/MLP. These align with the intended architecture-specific policy, but the final runner does not automatically consume a verified focal-selection manifest. A new OOF winner can diverge from these constants unless manually updated and checked.

### S: score division is a decision offset, not calibrated probability

`apply_class_score_scaling` divides R2L and U2R probabilities by separate positive coefficients before argmax, leaving other classes at divisor 1. A smaller divisor promotes a class; a larger divisor suppresses it. For positive softmax probabilities this has the same argmax as subtracting `log(c_k)` from class logits. The modified scores need not sum to one.

The default common 24-value grid yields 576 coefficient pairs per architecture/training regime. The raw `(1,1)` candidate is included. Selection maximizes mean Rare-F1 across seeds, then macro-F1, then conservative coefficient/tie rules. The selection is separate for CE, focal, batching, and F+B. This is exactly the needed improvement over applying one full-model coefficient pair indiscriminately to every cell.

Macro-F1 and rare precision retention flags are calculated in this newer scorer as **diagnostics**, not mandatory eligibility constraints. A candidate can be selected despite failing them. The methods section must match that behavior; it should not claim a hard macro-F1/precision safeguard unless such a rule was actually used for the reported artifacts.

S is tuned to an F1 objective on OOF data, not fitted as a probability calibrator or derived from a known real-world cost matrix. Because raw scores are in the grid, selected OOF Rare-F1 cannot fall below the raw candidate's primary objective except for implementation/numerical issues. Apparent OOF gains from S therefore partly reflect selection on the same OOF labels. Test gains remain meaningful only after coefficients are frozen.

S is also a policy that adapts its coefficients to each upstream regime. A factorial S contrast measures the effect of this fitted decision policy, not the effect of one identical fixed offset in every cell. Likewise, choosing F without B and carrying it into F+B measures a frozen focal policy; retuning F independently in every B condition would answer a different question. State these estimands clearly.

## Findings that require attention

| Priority | Finding | Evidence and consequence |
|---|---|---|
| High | Numerical result bundle is missing | No modern OOF NPZs, selection manifest, or final factorial tables are present. Abstract claims cannot be recomputed from this checkout. |
| High | Final factorial artifact validation omits semantic configuration checks | `tune_variant_specific_score_scaling.py:854` checks architecture/variant/seed, file/source hashes and a no-test-selection flag, but does not require selected beta/gamma, loss, epochs, batch size, quota, parameter count, backbone/preprocessor protocol, or no CTGAN. Incorrectly labeled but hash-consistent runs can enter a cell. |
| High | Final-run resume identity omits code/data/environment/determinism | `run_final_baseline_vs_full_kddtest_4gpu.py:805` builds its key from protocol settings and constants, while `result_is_complete:528` does not compare a result's feature-cache identity with the current cache or source identity. Cached results can be reused after source/data or determinism changes. |
| High for interpretation | B changes replacement sampling and coverage | The observed contrast belongs to the complete sampling policy, not an isolated guarantee. |
| Medium | Row/label checks establish agreement across artifacts rather than independent benchmark truth | The variant-specific checker verifies shape, finite probabilities, range, row order, raw argmax and fold agreement, but does not require the canonical dataset row count and labels reloaded from current raw sources. An internally consistent partial or mislabeled bundle can pass those checks. |
| Medium | Frozen focal constants are not enforced against selection artifacts | Old constants can persist after a new focal search. The current factorial loader does not close this gap. |
| Medium | Hard safeguards can be inferred incorrectly from diagnostic flags | Retention values are not filtering constraints in the current S selector. |
| Medium | Reproducible environment is missing/inconsistent | The dependency file is a historical pair of long environment lists, not an installable minimal lock. Its second environment specifies TensorFlow 2.18 with NumPy 1.23.5, incompatible with TensorFlow 2.18's declared NumPy range. |
| Medium | Test-focused ranking exists in legacy/post hoc utilities | `summarize_trials.py` defaults to test `macro_f1`; the fusion audit uses KDDTest metrics to rank candidates. These paths must not be confused with pre-test factorial selection. |
| Lower | Documentation and repository hygiene are stale | Missing referenced LICENSE, tracked bytecode, unrelated installer log, old figures, no current study runbook, and repeated large code copies make provenance harder to follow. |

### Concrete metadata check

The supplied metadata-only probe presented a `focal_only` run with ordinary CE, beta 0.9, gamma 99, one epoch, batch size 8, a wrong batch policy, CTGAN enabled, and a one-parameter model. It supplied internally consistent file hashes and the expected architecture/variant/seed. The actual `load_test_run_metadata()` function accepted it. This proves those semantics are unchecked by that function; it does **not** prove any unavailable real run used those settings, and it does not bypass or test the separate probability validator.

The repair should compare final metadata with an explicit selected training specification: class order, feature order/schema, model structure/count, optimizer, epoch budget, batch policy/quota, loss formulation, beta/gamma, no synthetic data, and determinism/software identity. Check full expected labels and row counts against current source files as well. The stricter semantic checks already implemented in the final simple-average/safe-stack evaluator provide a useful local model for this repair.

### Resume behavior

The final feature cache validates its source hashes, which is good. The problem is the separate completion shortcut: it recognizes completed run JSONs by a protocol key that does not include current source/data fingerprints. Rebuilding a cache after a raw-data change does not automatically make old run JSONs fail that shortcut. A later evaluation may catch some mismatches, but the resume runner should reject stale results itself. Include source hashes, core dependency source fingerprints, library versions and deterministic settings in a training identity, and verify the result's cache identity against the newly expected cache. The modern OOF scorer already distinguishes training and scoring identities more carefully.

### Environment details

The first historical environment's Python `3.19.8` is evidently a documentation typo. The second says Python 3.9.21, TensorFlow 2.18.0, Keras 3.8.0 and NumPy 1.23.5. [TensorFlow 2.18's official package definition](https://github.com/tensorflow/tensorflow/blob/v2.18.0/tensorflow/tools/pip_package/setup.py) requires NumPy ≥1.26.0 and <2.1.0. Newer code also uses `zip(..., strict=True)`, which requires Python 3.10+, irrespective of successful parsing on 3.9. Use a documented modern Python version with a tested compatible minimal dependency lock; keep CTGAN dependencies separate if historical runs are retained. Capture CUDA/cuDNN, device, TensorFlow/Keras versions, and deterministic flags with each reproducible result bundle.

## Statistical audit of the abstract

The primary metric is implemented correctly as the unweighted mean of R2L and U2R class F1, with fixed five-class labels and zero-division handling. This endpoint makes rare classes visible when accuracy is dominated by Normal and DoS. It still averages two quite different supports and can conceal improvement in one rare class with deterioration in the other. Always publish both rare-class precision, recall, F1, support and true-positive counts beside the primary endpoint, plus macro-F1, MCC, and accuracy as secondary outcomes.

The stated 31.00% versus 21.46% difference is 9.54 percentage points, or approximately 44.45% relative improvement **if the supplied numbers are correct**. Percentage-point and relative improvement statements are different. Neither is recomputed here because the final artifacts are absent.

### Define the factorial contrasts precisely

Let `Y[f,b,s]` be Rare Macro-F1 for a particular architecture and seed. Use the eight matched cells for that same architecture/seed:

`M_B = (1/4) Σ[f,s] {Y[f,1,s] − Y[f,0,s]}`.

For a pairwise focal/batching interaction averaged over scaling:

`I_FB = (1/2) Σ[s] {Y[1,1,s] − Y[1,0,s] − Y[0,1,s] + Y[0,0,s]}`.

Define `I_FS` and `I_BS` analogously. The three-way contrast is the difference between the F×B difference-of-differences at S=1 and S=0. Give formulas and units; ±1 regression coefficients use different numerical factors from these differences.

There is no dedicated marginal/interaction calculation or bootstrap confidence-analysis module in the checked-in current pipeline. The eight-cell tables are the necessary inputs, but they are absent here. Add a small reproducible analysis that recomputes all contrasts directly from the seed-level metrics and explicitly verifies the abstract's statements.

“All pairwise interactions are negative” is ambiguous without specifying whether it means three pooled architecture-mean interactions, all twelve architecture-specific averaged interactions, or every interaction conditional on the third factor. These are different statements. A negative averaged interaction does not ensure every conditional interaction is negative, particularly when the three-way contrast is substantial.

A negative interaction on F1 means the observed joint improvement is less than the sum of separate improvements on that metric scale. It does not necessarily mean the combination is worse than baseline or worse than either constituent. F1 is bounded and nonlinear in true/false positives; diminishing returns, shared corrections and ceiling effects can yield negative interactions without antagonistic learning mechanisms. Interpretation should remain descriptive unless additional mechanism evidence is provided.

### Uncertainty and selection

The same fold partition is used across the three training seeds. The reported seed SD captures training randomness under fixed folds; it is not the variability of independently resampled datasets or fold assignments. The twelve fold runs per configuration are not twelve independent statistical replicates. The four architecture means are four deliberately chosen backbones, not a random sample from all possible architectures. An SD across architectures describes heterogeneity, not a confidence interval for a universal architectural population.

Use paired seed-level contrasts and, where prediction arrays are available, paired class-stratified resampling of the same test rows across configurations. Preserve the rare-class supports in the resampling and state the resulting conditional estimand. With only 67 U2R test rows, include actual TP counts and avoid overly precise significance claims. Bootstrap intervals cannot establish generalization to new datasets or attack families.

The OOF metrics used to choose 18 focal settings and 576 coefficient pairs are development performance, not unbiased performance of the entire selection procedure. A frozen KDDTest evaluation can still legitimately assess the selected pipeline. Full nesting would be needed to interpret selected OOF performance as unbiased procedure-level validation. Search budget differs between toggles, so this is a comparison of selected intervention policies, not simply three untuned algorithms with identical selection complexity.

Flags such as `kddtest_accessed=False` and `kddtest_used_for_selection=False` express protocol intent. They do not prove chronological precommitment. Publish the frozen manifest/hash, selection timestamps or run history, exact final source IDs, and any prior test-driven development. Reporting the test's architecture-wise winner is a descriptive result; selecting it for a subsequent deployment claim consumes the test as development evidence.

### Scope of the scientific conclusion

If supported by the missing tables, a defensible conclusion is: **among these three selected policies, with this preprocessing, budget and benchmark, guaranteed replacement-based rare exposure has the most consistent positive marginal association across the four backbones**. That is useful cross-architecture evidence. One representative per stage on one dataset cannot establish that exposure interventions generally dominate all optimization or decision techniques. “Foundation” is best treated as the study's practical recommendation, bounded by these settings.

Do not conflate the highest architecture-mean configuration with the best configuration in each architecture. The abstract already recognizes variation in complements to B; show the full eight-cell matrix so readers can assess that heterogeneity and potential validation-to-test rank reversals.

## Legacy leakage and result comparability

The original and optimized legacy scripts commonly fit one-hot encoding/scaling on all training rows before making an internal validation split. That exposes validation feature distributions to preprocessing. Some CTGAN paths fit the generator before the validation split and/or split augmented data so synthetic rows occur in validation. These are real weaknesses of those historical validation estimates.

`cnn_opt.py` has a real-only validation split before adding synthetic rows in its updated path, but historical synthetic data may already have been generated from all real training rows. `cnn_opt_1d_4gpu.py` uses a different old augmented split. Consequently, legacy paired architecture comparisons can differ in more than backbone and parameter count even when they share seeds. Neural CTGAN pipelines versus real-only weighted XGBoost also compare whole pipelines, not just architecture families.

These weaknesses do **not** automatically apply to the modern factorial path: it uses the clean raw-fold preprocessor, no CTGAN, and fixed-epoch OOF fitting. The modern CTGAN amount sweep also makes the real split first, trains its generator only on the training portion, and keeps validation real only, which is a significant improvement over the original GAN scripts.

The legacy conditional generation helper merits repair if retained: it calculates a conditional label-match rate but can overwrite the generated class column with the requested class. A condition request is not a substitute for verifying the returned synthetic label. Relabeling mismatched generated rows can introduce label noise. Count/integer/binary and domain-range validation of generated predictors would also be needed to establish usable synthetic examples. This helper is outside the supplied no-CTGAN factorial's active data path.

`summarize_trials.py` defaults to ranking by `macro_f1`, which is the test metric in its historical result schema; callers must select the explicit validation metric for development use. Avoid choosing a trial from that default leaderboard for a claim of untouched test evaluation. The script also accepts partial metric groups, so an apparently strong mean can reflect incomplete seeds without an explicit completeness rule.

## Ensemble and post hoc research

The safe-stack engine combines ordinary CE, focal-only and batching-only OOF experts. It uses a selected base expert, rare-score mixtures, top-two agreement support, and margins before overriding majority predictions. It searches validation data only. It is a separate decision procedure, not one of the abstract's three independent factorial toggles.

The robust calibrated super-stack adds training-fold temperature fitting, log-probability and disagreement/interactions features, standardized logistic meta-models, rare-class sample weights, blend ratios, score offsets, and subtype-weighted robustness objectives. Its feature/refit/inference and cache tests are more developed than those of the training primitives. It correctly warns that the expert OOF predictions are frozen: nesting the meta-model is conditional on those experts, not a full re-fit of every backbone inside every outer split. Outer held-out labels can have influenced expert models that generate other meta-training rows. Do not describe this as fully nested end-to-end CV.

The validation-baseline selector deliberately changes the selection rule to absolute Rare-F1 gains against all stand-alone baselines and refits meta-models on appropriate complements. This is carefully documented, but it still shares the frozen-expert limitation and needs explicit adaptation history.

The final natural-rare stack evaluator acknowledges that earlier KDDTest results motivated later fusion work. That transparency is valuable: its numbers are post hoc generalization checks, not fresh untouched-test confirmation. The validation-filtered test audit then evaluates multiple OOF-surviving candidates on KDDTest and ranks with test-derived criteria. This is test-informed selection and requires an independent evaluation dataset for a subsequent confirmatory claim. These scripts are not dependencies of the original variant-specific factorial evaluation, so their existence alone does not prove the abstract's original experiment leaked test labels.

## Existing artifacts and documentation

The saved HDF5 model is the old approximately 55,493-weight Conv2D architecture with valid convolutions and a leaky-ReLU second convolution, trained with categorical CE. It is not one of the current approximately 100k-parameter matched backbones. The checkout does not include its original fitted encoders/scaler/order manifest as an inseparable inference bundle. A model file alone is insufficient to reproduce the required preprocessing for arbitrary new rows.

The legacy confusion matrix gives approximately 87.00% accuracy, R2L F1=74.32%, U2R F1=16.00%, and Rare Macro-F1≈45.16%. Only 6 of 67 U2R examples are correctly classified. These outputs illustrate why accuracy and ROC-AUC alone can conceal poor rare-class decisions: the U2R ROC curve reports AUC≈0.93 despite recall≈0.0896 at the selected argmax operating point.

That older 45.16% Rare-F1 must not be compared directly with the abstract's 31.00% as though it were an identical-protocol baseline. It uses a different model, data intervention and provenance. It nevertheless makes protocol separation important: readers will naturally notice the larger older value if the README foregrounds it without explanation.

`results/results.txt` says 25 epochs, while the available training curve and training log cover 100. They are not linked by a run ID/protocol manifest. The documented ten-fold score likewise lacks the current fold/prediction evidence needed to connect it to the new study. The six methodology figures depict the old GAN/undersampling or educational flows, not the modern factorial protocol. The grid-plan CSV contains 324 planned trial rows, not 324 completed result artifacts.

The README's MIT License link targets an absent file. Add the intended license with appropriate authorship authorization rather than treating the broken link as a license grant. Remove tracked interpreter bytecode and the unrelated Malwarebytes installer log, add a `.gitignore`, and move historical artifacts into an explicitly labeled legacy area. The existing `.vscode/extensions.json` simply has empty recommendations and has no scientific role.

## Recommended work in order

1. **Make the reported experiment reviewable.** Provide the selected focal manifests, four-regime OOF artifacts or documented download locations with hashes, frozen per-regime scaling manifest, final run metadata and raw probability files, eight-cell seed table, architecture means, and a script producing all abstract contrasts. Include class mapping and exact protocol/version IDs.
2. **Close silent lineage and reuse gaps.** Tie final runs to the selected OOF training specification, independently verify labels/counts, and use code/data/environment/determinism-aware training identities in the final resume logic. Add negative tests for mismatched loss/gamma/quota, stale cache/source identities, swapped labels, and truncated arrays.
3. **Align manuscript wording with implemented controls.** Describe B's replacement fill and expected exposure; distinguish CB weighting from focusing; state score division and diagnostic-only guards; define which pairwise contrasts are negative; report approximate counts and seed uncertainty accurately.
4. **Provide statistical evidence at the primary endpoint.** Publish rare-class P/R/F1 and TP counts, per-architecture and paired-seed marginal/interaction contrasts, three-way interaction, appropriately scoped intervals, and mapping/overlap sensitivities. Keep exploratory post-test fusion separate.
5. **Create a minimal reproducible environment and runbook.** Specify supported Python, compatible pinned packages, GPU requirements, commands from fresh checkout to frozen selection to test tables, expected output schemas and storage needs. Use one tiny end-to-end fixture for smoke validation and meaningful primitive tests for focal/batching/contrast correctness.
6. **Reduce duplicated orchestration after the study is stable.** Extract common CV scheduling/OOF reconstruction from the four large focal scripts; retain architecture-specific builders and explicit configuration. Preserve current manifests and hashes during refactoring so scientific artifacts remain traceable.

This audit adds review material only. It neither changes the experimental implementation nor silently changes the dataset mapping or reported results.
