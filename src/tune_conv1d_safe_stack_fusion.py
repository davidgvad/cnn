"""Tune Conv1D SAFE-Stack fusion from saved out-of-fold probabilities."""

from tune_conv2d_safe_stack_fusion import main


if __name__ == "__main__":
    main(default_architecture="conv1d")
