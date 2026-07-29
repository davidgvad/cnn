"""
Compare five fixed models on the same untouched KDDTest+ set.

Models:
  - MLP using the cnn_opt training pipeline
  - standard XGBoost
  - cost-sensitive XGBoost
  - Conv1D using the cnn_opt training pipeline
  - Conv2D from cnn_opt

All five use argmax predictions. Multiple paired seeds are summarized with
mean and standard deviation. Conv1D's dense width is automatically adjusted
to match Conv2D's parameter budget. Each GPU runs one child process at a time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Sequence

import pandas as pd


MODELS = [
    "mlp",
    "xgboost_standard",
    "xgboost_cost_sensitive",
    "conv1d",
    "conv2d",
]

METRICS = [
    "accuracy",
    "mcc",
    "macro_f1",
    "macro_recall",
    "r2l_recall",
    "u2r_recall",
]

RESULT_KEYS = {
    "Test Accuracy (sklearn)": "accuracy",
    "MCC": "mcc",
    "Test Macro F1": "macro_f1",
    "Test Macro Recall": "macro_recall",
    "R2L Recall": "r2l_recall",
    "U2R Recall": "u2r_recall",
    "Model Parameters": "model_parameters",
    "Best Validation Macro F1": "val_macro_f1",
    "best_iteration": "best_iteration",
}


def conv2d_parameter_count(
    base_filters: int,
    groups: int,
    dense_units: int,
) -> int:
    """Parameter count for cnn_opt with batch normalization enabled."""
    filters = int(base_filters)
    dense = int(dense_units)
    convolution_block = (
        21 * filters
        + (9 * filters * filters) // groups
        + filters * filters
    )
    dense_head = (4 * filters + 6) * dense + 5
    return convolution_block + dense_head


def conv1d_parameter_count(
    base_filters: int,
    groups: int,
    dense_units: int,
) -> int:
    """Parameter count for Conv1D with batch normalization enabled."""
    filters = int(base_filters)
    dense = int(dense_units)
    convolution_block = (
        15 * filters
        + (3 * filters * filters) // groups
        + filters * filters
    )
    dense_head = (30 * filters + 6) * dense + 5
    return convolution_block + dense_head


def matched_conv1d_dense_units(
    base_filters: int,
    groups: int,
    conv2d_dense_units: int,
) -> int:
    """Choose the integer Conv1D width nearest to Conv2D's parameter count."""
    target = conv2d_parameter_count(
        base_filters,
        groups,
        conv2d_dense_units,
    )
    filters = int(base_filters)
    conv1d_block_and_output_bias = (
        15 * filters
        + (3 * filters * filters) // groups
        + filters * filters
        + 5
    )
    parameters_per_dense_unit = 30 * filters + 6
    return max(
        1,
        round(
            (target - conv1d_block_and_output_bias)
            / parameters_per_dense_unit
        ),
    )


def parse_gpus(values: Sequence[str]) -> List[str]:
    gpus: List[str] = []
    for value in values:
        gpus.extend(part.strip() for part in value.split(",") if part.strip())
    if not gpus:
        raise ValueError("Provide at least one GPU ID.")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU IDs must be unique, got: {gpus}")
    return gpus


def fingerprint_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def read_metrics(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        output_name = RESULT_KEYS.get(key.strip())
        if output_name is None:
            continue
        try:
            values[output_name] = float(raw_value.strip())
        except ValueError:
            continue

    missing = [metric for metric in METRICS if metric not in values]
    if missing:
        raise ValueError(f"Missing metrics in {path}: {missing}")
    return values


def result_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        read_metrics(path)
        return True
    except (OSError, ValueError):
        return False


def add_shared_neural_arguments(
    command: List[str],
    args: argparse.Namespace,
    seed: int,
    synth_path: Path,
    dense_units: int,
) -> None:
    command.extend(
        [
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--seed",
            str(seed),
            "--val-split",
            str(args.val_split),
            "--synth-path",
            str(synth_path),
            "--feature-layout",
            "optimized",
            "--focal-gamma",
            str(args.focal_gamma),
            "--cb-beta",
            str(args.cb_beta),
            "--minority-per-batch",
            str(args.minority_per_batch),
            "--groups",
            str(args.groups),
            "--base-filters",
            str(args.base_filters),
            "--dense-units",
            str(dense_units),
            "--dropout1",
            str(args.dropout1),
            "--dropout2",
            str(args.dropout2),
            "--no-thresholds",
        ]
    )


def build_command(
    repo_root: Path,
    model_name: str,
    run_name: str,
    seed: int,
    synth_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    if model_name in {"mlp", "conv1d"}:
        command = [
            sys.executable,
            "-u",
            str(repo_root / "src" / "cnn_opt_1d_4gpu.py"),
            "--run-name",
            run_name,
            "--architecture",
            model_name,
            "--num-gpus",
            "1",
        ]
        dense_units = (
            args.conv1d_dense_units
            if model_name == "conv1d"
            else args.dense_units
        )
        add_shared_neural_arguments(
            command,
            args,
            seed,
            synth_path,
            dense_units,
        )
        return command

    if model_name == "conv2d":
        command = [
            sys.executable,
            "-u",
            str(repo_root / "src" / "cnn_opt.py"),
            "--run-name",
            run_name,
        ]
        add_shared_neural_arguments(
            command,
            args,
            seed,
            synth_path,
            args.dense_units,
        )
        return command

    weighting = (
        "none" if model_name == "xgboost_standard" else "balanced"
    )
    return [
        sys.executable,
        "-u",
        str(repo_root / "src" / "cost_sensitive_xgboost.py"),
        "--run-name",
        run_name,
        "--seed",
        str(seed),
        "--val-split",
        str(args.val_split),
        "--class-weighting",
        weighting,
        "--device",
        args.xgb_device,
        "--n-estimators",
        str(args.xgb_n_estimators),
        "--early-stopping-rounds",
        str(args.xgb_early_stopping_rounds),
    ]


def training_description(
    model_name: str,
    minority_per_batch: int,
) -> tuple[str, str]:
    if model_name in {"mlp", "conv1d", "conv2d"}:
        imbalance_method = "class-balanced focal loss"
        if minority_per_batch > 0:
            imbalance_method += " + minority-guaranteed batches"
        return (
            "KDDTrain+ + cached CTGAN CSV",
            imbalance_method,
        )
    if model_name == "xgboost_cost_sensitive":
        return ("real KDDTrain+", "balanced per-class sample weights")
    return ("real KDDTrain+", "none")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    parser = argparse.ArgumentParser(
        description="Paired multi-seed comparison of five models on KDDTest+."
    )
    parser.add_argument("--name-prefix", default="all_models")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--gpus", nargs="+", default=["0"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument(
        "--synth-path",
        default="data/synth_ctgan_5class.csv",
    )

    # Current cnn_opt defaults. Training settings are shared; Conv1D's dense
    # width is adjusted separately to match Conv2D's parameter budget.
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--cb-beta", type=float, default=0.9999)
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument(
        "--conv1d-dense-units",
        type=int,
        default=None,
        help=(
            "Conv1D dense width. By default it is calculated automatically "
            "to match the Conv2D parameter count."
        ),
    )
    parser.add_argument("--dropout1", type=float, default=0.25)
    parser.add_argument("--dropout2", type=float, default=0.30)

    parser.add_argument(
        "--xgb-device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument("--xgb-n-estimators", type=int, default=1000)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=50)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run again even when a complete result already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique seed values.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be greater than 0.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1.")
    if args.minority_per_batch < 0:
        parser.error("--minority-per-batch must be 0 or greater.")
    if args.groups <= 0 or args.base_filters % args.groups != 0:
        parser.error("--groups must be positive and divide --base-filters.")
    if args.dense_units <= 0:
        parser.error("--dense-units must be greater than 0.")
    if args.conv1d_dense_units is not None and args.conv1d_dense_units <= 0:
        parser.error("--conv1d-dense-units must be greater than 0.")
    if args.xgb_n_estimators <= 0:
        parser.error("--xgb-n-estimators must be greater than 0.")
    if args.xgb_early_stopping_rounds < 0:
        parser.error("--xgb-early-stopping-rounds must be 0 or greater.")
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))

    if args.conv1d_dense_units is None:
        args.conv1d_dense_units = matched_conv1d_dense_units(
            args.base_filters,
            args.groups,
            args.dense_units,
        )
    expected_conv2d_parameters = conv2d_parameter_count(
        args.base_filters,
        args.groups,
        args.dense_units,
    )
    expected_conv1d_parameters = conv1d_parameter_count(
        args.base_filters,
        args.groups,
        args.conv1d_dense_units,
    )

    synth_path = Path(args.synth_path).expanduser()
    if not synth_path.is_absolute():
        synth_path = repo_root / synth_path
    synth_path = synth_path.resolve()

    if not args.dry_run and not synth_path.exists():
        raise SystemExit(f"Cached CTGAN file not found: {synth_path}")
    if not args.dry_run and importlib.util.find_spec("xgboost") is None:
        raise SystemExit(
            "XGBoost is not installed. Run:\n"
            "  python -m pip install xgboost==2.1.3"
        )

    synth_class_counts: Dict[int, int] = {}
    if not args.dry_run:
        try:
            synth_labels = pd.to_numeric(
                pd.read_csv(synth_path, usecols=["class"])["class"],
                errors="raise",
            )
        except (KeyError, OSError, ValueError) as error:
            raise SystemExit(
                f"Invalid CTGAN CSV at {synth_path}: {error}"
            ) from error
        if bool(synth_labels.isna().any()):
            raise SystemExit(f"CTGAN class labels contain missing values: {synth_path}")
        try:
            integer_labels = synth_labels.astype(int)
        except (OverflowError, ValueError) as error:
            raise SystemExit(
                f"CTGAN class labels must be integers in 0..4: {synth_path}"
            ) from error
        if not bool((synth_labels == integer_labels).all()):
            raise SystemExit(
                f"CTGAN class labels must be integers in 0..4: {synth_path}"
            )
        invalid_classes = sorted(
            set(integer_labels.tolist()) - set(range(5))
        )
        if invalid_classes:
            raise SystemExit(
                f"Invalid CTGAN class IDs {invalid_classes}: {synth_path}"
            )
        synth_class_counts = {
            int(class_id): int(count)
            for class_id, count in integer_labels.value_counts()
            .sort_index()
            .items()
        }

    files_key = fingerprint_files(
        [
            repo_root / "src" / "compare_all_models.py",
            repo_root / "src" / "cnn_opt.py",
            repo_root / "src" / "cnn_opt_1d_4gpu.py",
            repo_root / "src" / "cost_sensitive_xgboost.py",
            repo_root / "src" / "cnn_gan_foc.py",
            repo_root / "data" / "KDDTrain+.txt",
            repo_root / "data" / "KDDTest+.txt",
            synth_path,
        ]
    )
    experiment_settings = (
        files_key,
        args.epochs,
        args.batch_size,
        args.val_split,
        args.focal_gamma,
        args.cb_beta,
        args.minority_per_batch,
        args.groups,
        args.base_filters,
        args.dense_units,
        args.conv1d_dense_units,
        args.dropout1,
        args.dropout2,
        args.xgb_device,
        args.xgb_n_estimators,
        args.xgb_early_stopping_rounds,
    )
    experiment_key = hashlib.sha256(
        repr(experiment_settings).encode("utf-8")
    ).hexdigest()[:12]
    prefix = args.name_prefix.strip() or "all_models"
    log_dir = results_dir / f"{prefix}_logs"

    plans: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for model_name in MODELS:
            run_name = (
                f"{prefix}_{experiment_key}_{model_name}_s{seed}"
            )
            plans.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "run_name": run_name,
                    "result_path": (
                        results_dir / f"{run_name}_results.txt"
                    ),
                }
            )

    plan_path = results_dir / f"{prefix}_plan.csv"
    print(f"Models: {MODELS}")
    print(f"Paired seeds: {args.seeds}")
    print(f"Total runs: {len(plans)}")
    print(f"GPUs: {gpus}")
    print("Decision policy: raw argmax for every model")
    print(
        "CNN parameter match: "
        f"Conv2D dense={args.dense_units}, params={expected_conv2d_parameters}; "
        f"Conv1D dense={args.conv1d_dense_units}, "
        f"params={expected_conv1d_parameters} "
        f"(difference={expected_conv1d_parameters - expected_conv2d_parameters:+d})"
    )
    if not args.dry_run:
        print(f"Synthetic class counts: {synth_class_counts}")
    print(f"Experiment key: {experiment_key}")
    print(f"Plan output: {plan_path}")

    if args.dry_run:
        for plan in plans:
            command = build_command(
                repo_root,
                plan["model"],
                plan["run_name"],
                plan["seed"],
                synth_path,
                args,
            )
            print(
                f"[{plan['model']} seed={plan['seed']}] "
                f"{shlex.join(command)}"
            )
        print("Dry run complete; nothing was trained.")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "experiment_key": experiment_key,
                "model": plan["model"],
                "seed": plan["seed"],
                "run_name": plan["run_name"],
                "conv2d_dense_units": args.dense_units,
                "conv1d_dense_units": args.conv1d_dense_units,
                "expected_conv2d_parameters": expected_conv2d_parameters,
                "expected_conv1d_parameters": expected_conv1d_parameters,
                "synth_class_counts": synth_class_counts,
            }
            for plan in plans
        ]
    ).to_csv(plan_path, index=False)

    assignments: Dict[str, List[Dict[str, Any]]] = {
        gpu: [] for gpu in gpus
    }
    for index, plan in enumerate(plans):
        assignments[gpus[index % len(gpus)]].append(plan)

    print_lock = threading.Lock()
    data_lock = threading.Lock()
    runtimes: Dict[str, float] = {}
    failures: List[str] = []
    failed_run_names: set[str] = set()

    def gpu_worker(gpu: str, gpu_plans: List[Dict[str, Any]]) -> None:
        for plan in gpu_plans:
            result_path = plan["result_path"]
            if not args.rerun and result_is_complete(result_path):
                with print_lock:
                    print(
                        f"[GPU {gpu}] SKIP {plan['run_name']}",
                        flush=True,
                    )
                continue

            if args.rerun and result_path.exists():
                backup_path = result_path.with_name(
                    f"{result_path.name}.previous-{int(time.time())}"
                )
                result_path.replace(backup_path)

            command = build_command(
                repo_root,
                plan["model"],
                plan["run_name"],
                plan["seed"],
                synth_path,
                args,
            )
            log_path = log_dir / f"{plan['run_name']}.log"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONHASHSEED"] = str(plan["seed"])

            with print_lock:
                print(
                    f"[GPU {gpu}] START {plan['run_name']}",
                    flush=True,
                )

            start = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(shlex.join(command) + "\n\n")
                log_file.flush()
                completed = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            runtime = time.perf_counter() - start
            complete = result_is_complete(result_path)

            with data_lock:
                runtimes[plan["run_name"]] = runtime
                if completed.returncode != 0 or not complete:
                    failed_run_names.add(plan["run_name"])
                    failures.append(
                        f"{plan['run_name']}: exit={completed.returncode}, "
                        f"log={log_path}"
                    )

            with print_lock:
                status = (
                    "DONE"
                    if completed.returncode == 0 and complete
                    else "FAILED"
                )
                print(
                    f"[GPU {gpu}] {status} {plan['run_name']} "
                    f"({runtime / 60.0:.1f} min)",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(gpu_worker, gpu, gpu_plans)
            for gpu, gpu_plans in assignments.items()
        ]
        for future in futures:
            future.result()

    completed_rows: List[Dict[str, Any]] = []
    for plan in plans:
        result_path = plan["result_path"]
        if (
            plan["run_name"] in failed_run_names
            or not result_is_complete(result_path)
        ):
            continue
        train_data, imbalance_method = training_description(
            plan["model"],
            args.minority_per_batch,
        )
        completed_rows.append(
            {
                "experiment_key": experiment_key,
                "model": plan["model"],
                "seed": plan["seed"],
                "run_name": plan["run_name"],
                "dense_units": (
                    args.conv1d_dense_units
                    if plan["model"] == "conv1d"
                    else (
                        args.dense_units
                        if plan["model"] in {"mlp", "conv2d"}
                        else None
                    )
                ),
                "synth_class_counts": (
                    synth_class_counts
                    if plan["model"] in {"mlp", "conv1d", "conv2d"}
                    else None
                ),
                "train_data": train_data,
                "imbalance_method": imbalance_method,
                "decision_policy": "argmax",
                "test_data": "KDDTest+",
                **read_metrics(result_path),
                "runtime_seconds": runtimes.get(plan["run_name"]),
                "result_path": str(result_path),
            }
        )

    if not completed_rows:
        raise SystemExit("No completed runs were found. Check the log files.")

    all_runs = pd.DataFrame(completed_rows)
    all_runs_path = results_dir / f"{prefix}_all_runs.csv"
    complete_seeds = [
        seed
        for seed in args.seeds
        if set(
            all_runs.loc[all_runs["seed"] == seed, "model"].tolist()
        )
        == set(MODELS)
    ]
    all_runs["paired_complete"] = all_runs["seed"].isin(complete_seeds)
    all_runs.to_csv(all_runs_path, index=False)

    paired_runs = all_runs[all_runs["seed"].isin(complete_seeds)].copy()
    if paired_runs.empty:
        raise SystemExit(
            f"No seed completed all five models. Partial results: {all_runs_path}"
        )
    paired_runs_path = results_dir / f"{prefix}_paired_runs.csv"
    paired_runs.to_csv(paired_runs_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for model_name in MODELS:
        model_runs = paired_runs[paired_runs["model"] == model_name]
        summary: Dict[str, Any] = {
            "model": model_name,
            "paired_seeds": len(model_runs),
        }
        for metric in METRICS:
            numeric = pd.to_numeric(
                model_runs[metric],
                errors="coerce",
            ).dropna()
            summary[f"{metric}_mean"] = float(numeric.mean())
            summary[f"{metric}_std"] = (
                float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
            )
        parameters = pd.to_numeric(
            model_runs.get("model_parameters"),
            errors="coerce",
        ).dropna()
        summary["model_parameters_mean"] = (
            float(parameters.mean()) if len(parameters) else None
        )
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = results_dir / f"{prefix}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Five-model KDDTest+ comparison ===")
    print(f"Complete paired seeds: {complete_seeds}")
    print(
        f"{'Model':<27} {'Accuracy':>9} {'MCC':>9} {'Macro-F1':>10} "
        f"{'Macro Rec':>10} {'R2L Rec':>9} {'U2R Rec':>9}"
    )
    for row in summary_rows:
        print(
            f"{row['model']:<27} "
            f"{row['accuracy_mean']:>9.4f} "
            f"{row['mcc_mean']:>9.4f} "
            f"{row['macro_f1_mean']:>10.4f} "
            f"{row['macro_recall_mean']:>10.4f} "
            f"{row['r2l_recall_mean']:>9.4f} "
            f"{row['u2r_recall_mean']:>9.4f}"
        )
    print(f"\nPer-seed results: {all_runs_path}")
    print(f"Complete paired runs: {paired_runs_path}")
    print(f"Mean/std summary: {summary_path}")

    if failures:
        print(f"\nFailed runs: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
