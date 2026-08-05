"""Run the pure MLP baseline with four-fold, three-seed OOF evaluation.

The fixed parameter-matched MLP and training budget remain unchanged.  No
imbalance enhancement is used: training uses ordinary sparse cross-entropy,
shuffled mini-batches, no synthetic data, and raw multiclass argmax.
"""

from tune_conv2d_score_scaling_cv_4gpu import main


if __name__ == "__main__":
    main(default_architecture="mlp", default_training_mode="baseline_ce")
