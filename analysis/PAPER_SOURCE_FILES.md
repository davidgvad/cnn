# Individual source classifications for the supplied paper

All 57 current src/ files. The classification concerns the numerical pipeline in the supplied manuscript, not proof that an unrelated file was never executed historically. See [the detailed mapping](PAPER_CODE_MAP.md).

## paper_core_or_shared_dependency (11)

| File | Role / basis |
|---|---|
| [src/cnn_gan_foc.py](../src/cnn_gan_foc.py) | Shared ClassBalancedFocalLoss; GAN generation/main is outside this paper. |
| [src/cnn_opt.py](../src/cnn_opt.py) | Shared Conv2D builder, BalancedBatchSequence and score helpers; legacy main is outside this paper. |
| [src/cnn_opt_1d_4gpu.py](../src/cnn_opt_1d_4gpu.py) | Shared Conv1D, feature Transformer and MLP builders; legacy main is outside this paper. |
| [src/run_final_baseline_vs_full_kddtest_4gpu.py](../src/run_final_baseline_vs_full_kddtest_4gpu.py) | Four-regime final network fits and saved raw test probabilities; own older scaled results do not match all paper coefficients. |
| [src/run_no_ctgan_model_ablation_4gpu.py](../src/run_no_ctgan_model_ablation_4gpu.py) | Shared raw mapping, fold preprocessor, class weights, metrics and artifact utilities; its holdout main is not the paper CV protocol. |
| [src/tune_conv1d_focal_cv_4gpu.py](../src/tune_conv1d_focal_cv_4gpu.py) | Conv1D Stage-1 focal selection and selected focal-only OOF arrays. |
| [src/tune_conv2d_focal_cv_4gpu.py](../src/tune_conv2d_focal_cv_4gpu.py) | Conv2D Stage-1 18-candidate four-fold focal selection and selected focal-only OOF arrays. |
| [src/tune_conv2d_score_scaling_cv_4gpu.py](../src/tune_conv2d_score_scaling_cv_4gpu.py) | Generic four-backbone OOF engine for CE, CE+B and F+B, plus shared 576-pair scoring machinery. |
| [src/tune_mlp_focal_cv_4gpu.py](../src/tune_mlp_focal_cv_4gpu.py) | MLP Stage-1 focal selection and selected focal-only OOF arrays. |
| [src/tune_transformer_focal_cv_4gpu.py](../src/tune_transformer_focal_cv_4gpu.py) | Transformer Stage-1 focal selection and selected focal-only OOF arrays. |
| [src/tune_variant_specific_score_scaling.py](../src/tune_variant_specific_score_scaling.py) | Independent per-regime scaling selection, frozen manifest, complete OOF/test eight-cell tables and secondary metrics. |

## paper_optional_or_alternative_launcher (20)

| File | Role / basis |
|---|---|
| [src/run_conv1d_baseline_cv_4gpu.py](../src/run_conv1d_baseline_cv_4gpu.py) | Convenience OOF launcher for conv1d/ordinary CE; equivalent generic-engine CLI exists. |
| [src/run_conv1d_batch_baseline_cv_4gpu.py](../src/run_conv1d_batch_baseline_cv_4gpu.py) | Convenience OOF launcher for conv1d/CE+B; equivalent generic-engine CLI exists. |
| [src/run_conv1d_focal_batch_cv_4gpu.py](../src/run_conv1d_focal_batch_cv_4gpu.py) | Convenience OOF launcher for conv1d/F+B fallback; equivalent generic-engine CLI exists. |
| [src/run_conv2d_baseline_cv_4gpu.py](../src/run_conv2d_baseline_cv_4gpu.py) | Convenience OOF launcher for conv2d/ordinary CE; equivalent generic-engine CLI exists. |
| [src/run_conv2d_batch_baseline_cv_4gpu.py](../src/run_conv2d_batch_baseline_cv_4gpu.py) | Convenience OOF launcher for conv2d/CE+B; equivalent generic-engine CLI exists. |
| [src/run_conv2d_focal_batch_cv_4gpu.py](../src/run_conv2d_focal_batch_cv_4gpu.py) | Convenience OOF launcher for conv2d/F+B fallback; equivalent generic-engine CLI exists. |
| [src/run_final_single_enhancements_kddtest_4gpu.py](../src/run_final_single_enhancements_kddtest_4gpu.py) | Convenience final launcher for focal_only/batch_only/scaling_only; raw focal/batch arrays feed the paper, old scaling_only array is unnecessary. |
| [src/run_mlp_baseline_cv_4gpu.py](../src/run_mlp_baseline_cv_4gpu.py) | Convenience OOF launcher for mlp/ordinary CE; equivalent generic-engine CLI exists. |
| [src/run_mlp_batch_baseline_cv_4gpu.py](../src/run_mlp_batch_baseline_cv_4gpu.py) | Convenience OOF launcher for mlp/CE+B; equivalent generic-engine CLI exists. |
| [src/run_mlp_focal_batch_cv_4gpu.py](../src/run_mlp_focal_batch_cv_4gpu.py) | Convenience OOF launcher for mlp/F+B fallback; equivalent generic-engine CLI exists. |
| [src/run_transformer_baseline_cv_4gpu.py](../src/run_transformer_baseline_cv_4gpu.py) | Convenience OOF launcher for transformer/ordinary CE; equivalent generic-engine CLI exists. |
| [src/run_transformer_batch_baseline_cv_4gpu.py](../src/run_transformer_batch_baseline_cv_4gpu.py) | Convenience OOF launcher for transformer/CE+B; equivalent generic-engine CLI exists. |
| [src/run_transformer_focal_batch_cv_4gpu.py](../src/run_transformer_focal_batch_cv_4gpu.py) | Convenience OOF launcher for transformer/F+B fallback; equivalent generic-engine CLI exists. |
| [src/tune_conv1d_baseline_score_scaling_cv_4gpu.py](../src/tune_conv1d_baseline_score_scaling_cv_4gpu.py) | Alternative conv1d ordinary-CE OOF launcher with baseline scaling grid; produces the same baseline pointer family. |
| [src/tune_conv1d_score_scaling_cv_4gpu.py](../src/tune_conv1d_score_scaling_cv_4gpu.py) | Preferred focal+batching OOF source launcher for conv1d; generic Conv2D-named engine handles this architecture. |
| [src/tune_conv2d_baseline_score_scaling_cv_4gpu.py](../src/tune_conv2d_baseline_score_scaling_cv_4gpu.py) | Alternative conv2d ordinary-CE OOF launcher with baseline scaling grid; produces the same baseline pointer family. |
| [src/tune_mlp_baseline_score_scaling_cv_4gpu.py](../src/tune_mlp_baseline_score_scaling_cv_4gpu.py) | Alternative mlp ordinary-CE OOF launcher with baseline scaling grid; produces the same baseline pointer family. |
| [src/tune_mlp_score_scaling_cv_4gpu.py](../src/tune_mlp_score_scaling_cv_4gpu.py) | Preferred focal+batching OOF source launcher for mlp; generic Conv2D-named engine handles this architecture. |
| [src/tune_transformer_baseline_score_scaling_cv_4gpu.py](../src/tune_transformer_baseline_score_scaling_cv_4gpu.py) | Alternative transformer ordinary-CE OOF launcher with baseline scaling grid; produces the same baseline pointer family. |
| [src/tune_transformer_score_scaling_cv_4gpu.py](../src/tune_transformer_score_scaling_cv_4gpu.py) | Preferred focal+batching OOF source launcher for transformer; generic Conv2D-named engine handles this architecture. |

## outside_paper_numerical_pipeline (26)

| File | Role / basis |
|---|---|
| [src/add_transformer_to_model_comparison.py](../src/add_transformer_to_model_comparison.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/audit_validation_filtered_fusion_kddtest.py](../src/audit_validation_filtered_fusion_kddtest.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/cnn_baseline.py](../src/cnn_baseline.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/cnn_fin.py](../src/cnn_fin.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/cnn_focal.py](../src/cnn_focal.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/compare_all_models.py](../src/compare_all_models.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/compare_cnn_1d_2d.py](../src/compare_cnn_1d_2d.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/compare_cnn_opt_xgboost.py](../src/compare_cnn_opt_xgboost.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/cost_sensitive_xgboost.py](../src/cost_sensitive_xgboost.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/evaluate_cnn_grid_winner.py](../src/evaluate_cnn_grid_winner.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/evaluate_final_natural_rare_super_stack_kddtest.py](../src/evaluate_final_natural_rare_super_stack_kddtest.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/evaluate_final_simple_average_vs_safe_stack_kddtest.py](../src/evaluate_final_simple_average_vs_safe_stack_kddtest.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/grid_search_cnn_opt_4gpu.py](../src/grid_search_cnn_opt_4gpu.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/random_search_cnn_opt_1d.py](../src/random_search_cnn_opt_1d.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/run_cnn_ablation_4gpu.py](../src/run_cnn_ablation_4gpu.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/run_ctgan_amount_sweep_4gpu.py](../src/run_ctgan_amount_sweep_4gpu.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/select_fusion_against_all_validation_baselines.py](../src/select_fusion_against_all_validation_baselines.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/summarize_trials.py](../src/summarize_trials.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/sweep_fixed_backbone_imbalance_4gpu.py](../src/sweep_fixed_backbone_imbalance_4gpu.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/transformer_baseline.py](../src/transformer_baseline.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/tune_cnn_opt_thresholds.py](../src/tune_cnn_opt_thresholds.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/tune_conv1d_safe_stack_fusion.py](../src/tune_conv1d_safe_stack_fusion.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/tune_conv2d_safe_stack_fusion.py](../src/tune_conv2d_safe_stack_fusion.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/tune_mlp_safe_stack_fusion.py](../src/tune_mlp_safe_stack_fusion.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/tune_robust_calibrated_super_stack_all.py](../src/tune_robust_calibrated_super_stack_all.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |
| [src/tune_transformer_safe_stack_fusion.py](../src/tune_transformer_safe_stack_fusion.py) | No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run. |

