# Validation of the portable reviewer workflow

The package's execution code was tested in an isolated Linux checkout using Python 3.12.7 and the versions pinned in `requirements.txt`. The checkout path contained spaces, and commands ran from a different working directory. All included source and test files were checked against the local working copy by SHA-256.

Completed checks:

- 18 unit tests passed: hardware selection/allocation, fixed scientific command settings, metric formulas, factorial contrasts, scaling ties, native artifact validation, and moved-artifact resolution.
- Complete plans passed on CPU and for explicitly requested two- and eight-GPU workers. Each retained 192 OOF fits and 48 final fits. GPU planning used dry runs; fitting on two/eight GPUs was not performed during this check.
- The table exporter regenerated all nine tables from the original server probabilities. All 1,536 unrounded metric mean/SD values matched the independently verified reference summaries.
- The CPU smoke stage passed sixteen one-epoch fits on sampled data: 100 training and 25 test rows, Conv2D focal-only/focal-plus-batching OOF workers, and baseline/focal-plus-batching final workers for all four backbones. Parameter counts and completed epochs matched. The test limited TensorFlow intra/inter-op threads to two to limit resource use on the shared machine.
- The original server source, raw datasets and result pointers remained unchanged during the isolated checks.

The reviewer workflow modifies the repository's controllers for portable hardware/path configuration and consistent paper score coefficients. Neural model builders, preprocessing, losses, guaranteed-batch sampling and fitting remain unchanged. Source comparisons checked this separately from the smoke execution.

The full 240-fit paper repetition with chosen parameters fixed has not been performed by this package preparation. CPU smoke results are execution checks, not paper results. Full hyperparameter-selection history, including the unresolved earlier MLP search, is outside this fixed-parameter workflow. Cross-hardware numerical agreement must be evaluated after full fresh fitting; bitwise equality is not promised.
