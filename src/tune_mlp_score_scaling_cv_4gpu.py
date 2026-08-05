"""Run the shared balanced-batch OOF score-scaling search for MLP."""

from tune_conv2d_score_scaling_cv_4gpu import main


if __name__ == "__main__":
    main(default_architecture="mlp")
