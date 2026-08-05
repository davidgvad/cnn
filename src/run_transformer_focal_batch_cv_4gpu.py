"""Evaluate Transformer with focal loss plus minority-guaranteed batches."""

from tune_conv2d_score_scaling_cv_4gpu import main


if __name__ == "__main__":
    main(
        default_architecture="transformer",
        default_training_mode="focal_balanced",
        default_coefficient_values=[1.0],
        default_name_prefix="transformer_focal_batch_cv",
    )
