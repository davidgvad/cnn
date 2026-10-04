# Paper run provenance from the server manifests

Follow-up: the saved prediction arrays have now been independently re-scored on the server, and all 16 scaling searches repeated. See [PAPER_CLAIMS_AND_REVERIFICATION.md](PAPER_CLAIMS_AND_REVERIFICATION.md). The report below records the earlier audit stage.

**Subsequent complete export:** all 80 metadata records and seven CSVs have now been inspected, and all 96 prediction-file hashes have been checked directly on the server. The actual final run settings and all primary/secondary saved-metric aggregates agree with the manuscript. The MLP source contains only a single focal candidate, leaving the manuscript's eighteen-candidate MLP search claim unsupported by the recovered records. See [SERVER_EXPORT_AUDIT.md](SERVER_EXPORT_AUDIT.md) for the completed inspection and rounding corrections. Statements below about unread producer/run metadata describe the earlier two-manifest checkpoint.

The supplied server manifests establish a consistent chain from frozen OOF selection **`ea63ca7e718d`** to KDDTest evaluation **`d5c1c2bb3050`**. The associated primary tables and all sixteen coefficient pairs already matched the manuscript in [SERVER_RESULT_VERIFICATION.md](SERVER_RESULT_VERIFICATION.md).

Both JSON payloads were extracted from the pasted terminal transcripts with their final newline preserved. Their full SHA-256 hashes exactly match the hashes in the previously supplied latest pointers. The evaluation references the supplied selection by both its ID and its full manifest hash. Independently rebuilding the identity hashes from the documented fields reproduces both IDs.

Both manifests record scoring-script SHA-256 `61fc8dafaa58d75c4f6e31ddbfbf4dc7b5dbee485762383ec41284f96bcd2091`. This exactly matches the current local [tune_variant_specific_score_scaling.py](../src/tune_variant_specific_score_scaling.py). Thus the final selection/evaluation script's recorded version is identified at byte level. This hash does not cover its imported helper modules.

The KDDTrain+ hash recorded for all sixteen OOF source groups and all forty-eight final prediction sources matches the local `data/KDDTrain+.txt`. All forty-eight test sources also record the local `data/KDDTest+.txt` hash. The selection records four architectures, three seeds, four underlying training regimes and the same 24-value coefficient grid as the manuscript, giving 576 pairs per architecture/regime. Its sixteen selected rows also agree with the previously pasted coefficient CSV, including the unrounded OOF score values.

## Exact OOF source groups

Each cell below gives the experiment ID embedded in the recorded source directory. The full directory, producer protocol, selected-config and prediction paths are preserved in [server_manifest_verification.json](server_manifest_verification.json) and [server_prediction_lineage.csv](server_prediction_lineage.csv).

| Architecture | Baseline CV | Selected focal Stage 1 | Batching-only CV | Focal + batching CV |
|---|---|---|---|---|
| Conv2D | `a546e5aa0cc7` | `e85df7f4e441` | `920c07cf4023` | `fd9c49713960` |
| Conv1D | `3791c3caba4e` | `e384e5a86c3f` | `71dbe8728273` | `da3be350aa19` |
| Transformer | `318ef61e45ed` | `f39c10180e9b` | `af6b6d968387` | `ec9f0f1b0ed6` |
| MLP | `756465e5abd4` | `0d0a1b34a405` | `69b405b1ec20` | `0dce0ce7d82b` |

Each group contributes three complete OOF prediction files, one for each seed: forty-eight OOF sources in total. The focal Stage-1 selected configuration IDs encode beta 0.99 and gamma 0.50, 0.25, 0.75 and 0.25 respectively. The original four best-config JSONs are identified by path and hash but have not yet been supplied for direct inspection.

All four focal-plus-batching OOF groups use the `balanced_score_scaling` source family. The manifest records `legacy_focal_batch_fields` as their training-mode evidence. This is the older metadata format explicitly supported by the current scorer: it identifies focal loss and guaranteed batching through model, loss hyperparameters, batching policy and quota fields rather than the newer explicit `training_mode` field. The underlying producer protocols still need inspection against the full paper settings.

These sources correspond to the previously identified four `tune_<architecture>_focal_cv_4gpu.py` engines and the generic [tune_conv2d_score_scaling_cv_4gpu.py](../src/tune_conv2d_score_scaling_cv_4gpu.py) engine. The generic engine's filename covers all four backbones. Which interchangeable baseline/batching/architecture wrappers were invoked remains a separate execution-history question.

## Exact final test source groups

| Architecture | Baseline and F+B experiment family | Focal-only and batching-only family |
|---|---|---|
| Conv2D | `final_baseline_vs_full_kddtest_f1fce59e8498` | `final_single_enhancement_kddtest_3ec1e4e235cb` |
| Conv1D | `final_baseline_vs_full_kddtest_8cd98c1c3a8c` | Same combined family |
| Transformer | `final_baseline_vs_full_kddtest_5f655d7ccb1f` | Same combined family |
| MLP | `final_baseline_vs_full_kddtest_1e17a659cef6` | Same combined family |

The four baseline/full families each contribute six sources: two underlying regimes × three seeds. The single-enhancement family contributes twenty-four sources: four architectures × two underlying regimes × three seeds. Together these are forty-eight distinct test probability files, with forty-eight corresponding run-metadata JSONs and five feature-cache metadata JSONs.

The evaluator calls the raw regime `focal_batch` but reads files stored under variant name `full`. It uses their raw probabilities and raw predictions to build the F+B and newly scaled F+B+S cells. The older built-in scaled decisions in these files are not the manuscript's newly selected decisions. Similarly, the manuscript's scaling-only cells reuse baseline probabilities. The evaluator's source records contain no `scaling_only` final-fit files.

The fitting implementation is [run_final_baseline_vs_full_kddtest_4gpu.py](../src/run_final_baseline_vs_full_kddtest_4gpu.py). [run_final_single_enhancements_kddtest_4gpu.py](../src/run_final_single_enhancements_kddtest_4gpu.py) is the convenience launcher consistent with the combined family. An explicit call to the shared implementation with the same arguments would create the same family, so these records alone cannot distinguish the launcher command.

## Why the final experiment IDs differ between source versions

I reconstructed the five observed final experiment keys without running the fitting code. Four baseline/full keys exactly match the protocol schema and frozen configuration in historical commit `06db6f1`, with one backbone per group, seeds 0–2, 25 epochs, batch size 256 and one reserved example per minority class. The combined single-enhancement key exactly matches the newer protocol schema with all four backbones and variants `focal_only`, `batch_only`, `scaling_only`.

The baseline/full schema uses purpose string `final baseline versus full comparison on KDDTest+`; the newer schema uses `final frozen neural-configuration evaluation on KDDTest+`. This explains why reconstructing the four old keys directly with the newer purpose string fails even with the same numerical settings. Both historical and current versions have the same frozen backbone/focal constants.

This is evidence consistent with the original baseline/full protocol and a later combined single-enhancement protocol. An experiment key is a twelve-character hash of protocol settings, not a source-code hash. Its reconstruction does not establish an exact historical commit, software environment, completed epoch count or command line. The forty-eight original run JSONs are needed to check the actual recorded settings and completion details. The combined protocol includes an older scaling-only variant, but the current evaluation only references its focal-only and batching-only sources.

## Confidence and remaining evidence

The recorded final scoring-script version, the two top-level manifest identities, their cross-link, the displayed primary results, the selected coefficients and the exact ninety-six source prediction references are established. All source-reference hashes are internally consistent between the identity fields and detailed records. The underlying probability-file bytes have not yet been supplied or independently checksummed in this audit.

The eleven-file dependency core in [PAPER_CODE_MAP.md](PAPER_CODE_MAP.md) remains the appropriate core to retain. The single-enhancement launcher is now specifically associated with an observed final experiment family and should also be retained when preserving the historical workflow. Other interchangeable launchers can only be classified as historically used or unused when their producer metadata or command logs distinguish them. Files outside the paper's numerical pipeline are archive candidates, not proof of never having been used during development.

For the next inspection, [server_metadata_requested.csv](server_metadata_requested.csv) lists seventy-five metadata files with recorded full hashes; two are the supplied top-level manifests, and seventy-three remain unread. Five additional final producer protocols can be located by replacing `_feature_cache.json` with `_protocol.json` in the recorded cache paths. The per-seed and summary CSVs are also needed to verify secondary metrics and aggregation directly.

[server_metadata_export_command.txt](server_metadata_export_command.txt) contains a read-only command to print these records together for pasting into the conversation. It also prints current source-file hashes and the four server launch scripts, when present. Alternatively, [collect_server_paper_metadata.py](collect_server_paper_metadata.py) can create one metadata/source/table archive and optionally checksum the ninety-six prediction files. Neither method trains models or performs inference. Existing experiment files remain untouched; the archive collector creates only its archive.

All local work in this step was confined to `analysis/`, including the added audit collector and export instructions. No training or inference was run, and no existing project source, dataset or experiment-result files were modified or deleted.
