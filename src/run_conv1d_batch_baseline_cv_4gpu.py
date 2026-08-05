"""Evaluate Conv1D with cross-entropy plus minority-guaranteed batches only."""

from tune_conv2d_score_scaling_cv_4gpu import main


if __name__ == "__main__":
    main(
        default_architecture="conv1d",
        default_training_mode="baseline_batch",
        default_coefficient_values=[1.0],
        default_name_prefix="conv1d_batch_baseline_cv",
    )
