"""Tune post-hoc score scaling on pure cross-entropy Conv1D OOF predictions."""

from tune_conv2d_score_scaling_cv_4gpu import (
    BASELINE_SCALING_COEFFICIENTS,
    main,
)


if __name__ == "__main__":
    main(
        default_architecture="conv1d",
        default_training_mode="baseline_ce",
        default_coefficient_values=BASELINE_SCALING_COEFFICIENTS,
        default_name_prefix="conv1d_baseline_cv",
    )
