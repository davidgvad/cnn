"""Run the pure Conv2D baseline with four-fold, three-seed OOF evaluation.

The Conv2D backbone and training budget stay fixed for a fair comparison.  This
baseline deliberately uses no imbalance enhancement: ordinary sparse
cross-entropy, shuffled mini-batches, no synthetic data, and raw argmax.
"""

from tune_conv2d_score_scaling_cv_4gpu import main


if __name__ == "__main__":
    main(default_architecture="conv2d", default_training_mode="baseline_ce")
