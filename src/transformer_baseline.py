"""
Run the vanilla feature-token Transformer with the cnn_opt pipeline.

The launcher defaults to raw argmax and a 512-unit classifier head. Together
with the default encoder settings, this gives 110,661 parameters versus
109,381 in the default Conv2D model.
"""

from cnn_opt_1d_4gpu import main


if __name__ == "__main__":
    main(
        default_architecture="transformer",
        default_run_name="transformer_baseline",
        default_no_thresholds=True,
    )
