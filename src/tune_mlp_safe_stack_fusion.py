"""Tune MLP SAFE-Stack fusion from saved OOF probabilities."""

from tune_conv2d_safe_stack_fusion import main


if __name__ == "__main__":
    main(default_architecture="mlp")
