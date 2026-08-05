"""Run all single-enhancement neural ablations on final KDDTest+.

This is the simple entry point for the paper experiment. It evaluates, for
Conv2D, Conv1D, Transformer, and MLP across seeds 0, 1, and 2:

1. model + class-balanced focal loss only;
2. model + minority-guaranteed mini-batches only;
3. model + frozen class-score scaling only.

The implementation and frozen settings live in the shared final-test runner.
"""

from __future__ import annotations

import sys

from run_final_baseline_vs_full_kddtest_4gpu import main


if __name__ == "__main__":
    # Keep this entry point deliberately simple: users only need to specify the
    # allocated GPUs. Additional shared-runner options still work normally.
    sys.argv[1:1] = [
        "--variants",
        "focal_only",
        "batch_only",
        "scaling_only",
    ]
    main()
