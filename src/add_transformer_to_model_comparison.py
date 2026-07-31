"""
Add a five-seed Transformer baseline to the existing five-model comparison.

The original model5_param_matched experiment already contains five paired
KDDTest+ runs for MLP, standard XGBoost, cost-sensitive XGBoost, Conv1D, and
Conv2D. This script:

1. validates those original per-seed results;
2. trains the parameter-matched Transformer for the same seeds;
3. combines all six models from raw per-seed results; and
4. writes a fresh mean +/- sample-standard-deviation table.

The original five-model files are never modified.

For comparability, this intentionally reproduces the historical neural
protocol in which the cached synthetic rows were added before the 80/20
train/validation split. It should not be mixed with newer real-only-validation
runs.
"""

from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


BASE_MODELS = [
    "mlp",
    "xgboost_standard",
    "xgboost_cost_sensitive",
    "conv1d",
    "conv2d",
]
ALL_MODELS = [*BASE_MODELS, "transformer"]
NEURAL_BASE_MODELS = {"mlp", "conv1d", "conv2d"}

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
}

DISPLAY_NAMES = {
    "mlp": "MLP",
    "xgboost_standard": "XGBoost standard",
    "xgboost_cost_sensitive": "XGBoost cost-sensitive",
    "conv1d": "Conv1D",
    "conv2d": "Conv2D",
    "transformer": "Transformer",
}


def parse_gpus(values: Sequence[str]) -> List[str]:
    gpus: List[str] = []
    for value in values:
        gpus.extend(
            part.strip() for part in value.split(",") if part.strip()
        )
    if not gpus:
        raise ValueError("Provide at least one GPU ID.")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU IDs must be unique, got: {gpus}")
    return gpus


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def fingerprint_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as input_file:
            for chunk in iter(
                lambda: input_file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def read_metadata(path: Path) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def read_metrics(path: Path) -> Dict[str, float]:
    metadata = read_metadata(path)
    values: Dict[str, float] = {}
    for source_name, output_name in RESULT_KEYS.items():
        raw_value = metadata.get(source_name)
        if raw_value is None:
            continue
        try:
            values[output_name] = float(raw_value)
        except ValueError:
            continue

    missing = [metric for metric in METRICS if metric not in values]
    if missing:
        raise ValueError(f"Missing metrics in {path}: {missing}")
    return values


def resolve_recorded_result_path(
    repo_root: Path,
    results_dir: Path,
    raw_path: str,
) -> Path:
    recorded = Path(raw_path).expanduser()
    candidates = [
        recorded,
        repo_root / recorded,
        results_dir / recorded.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find recorded result file: {raw_path}"
    )


def require_numeric_metadata(
    metadata: Dict[str, str],
    key: str,
    expected: float,
    path: Path,
) -> None:
    raw_value = metadata.get(key)
    if raw_value is None:
        raise ValueError(f"{path} is missing metadata field '{key}'.")
    try:
        actual = float(raw_value)
    except ValueError as error:
        raise ValueError(
            f"{path} has non-numeric {key}: {raw_value}"
        ) from error
    if not np.isclose(actual, float(expected)):
        raise ValueError(
            f"{path} has {key}={actual}, expected {expected}."
        )


def parse_count_list(
    raw_value: str,
    source: Path,
    field_name: str = "synth_counts",
) -> List[int]:
    try:
        values = [int(value) for value in ast.literal_eval(raw_value)]
    except (SyntaxError, TypeError, ValueError) as error:
        raise ValueError(
            f"Could not parse {field_name} from {source}: {raw_value}"
        ) from error
    if len(values) != 5:
        raise ValueError(
            f"Expected five {field_name} values in {source}, got {values}."
        )
    return values


def validate_base_runs(
    base_runs_path: Path,
    repo_root: Path,
    results_dir: Path,
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    List[int],
    Dict[int, Dict[str, List[int]]],
]:
    base_runs = pd.read_csv(base_runs_path)
    required_columns = {
        "model",
        "seed",
        "experiment_key",
        "result_path",
        "decision_policy",
        "test_data",
        *METRICS,
    }
    missing_columns = sorted(
        required_columns - set(base_runs.columns)
    )
    if missing_columns:
        raise ValueError(
            f"{base_runs_path} is missing columns: {missing_columns}"
        )

    base_runs["seed"] = pd.to_numeric(
        base_runs["seed"],
        errors="raise",
    ).astype(int)
    selected = base_runs[
        base_runs["seed"].isin([int(seed) for seed in seeds])
    ].copy()
    expected_seeds = {int(seed) for seed in seeds}

    unexpected_models = sorted(
        set(selected["model"].astype(str)) - set(BASE_MODELS)
    )
    if unexpected_models:
        raise ValueError(
            f"Unexpected models in {base_runs_path}: {unexpected_models}"
        )

    for model_name in BASE_MODELS:
        model_rows = selected[selected["model"] == model_name]
        observed_seeds = set(model_rows["seed"].tolist())
        if observed_seeds != expected_seeds or len(model_rows) != len(
            expected_seeds
        ):
            raise ValueError(
                f"{model_name} does not contain exactly the requested "
                f"paired seeds. Expected {sorted(expected_seeds)}, "
                f"got {sorted(observed_seeds)}."
            )

    if selected.duplicated(subset=["model", "seed"]).any():
        raise ValueError(
            f"{base_runs_path} contains duplicate model/seed rows."
        )
    experiment_keys = set(
        selected["experiment_key"].fillna("").astype(str)
    )
    if len(experiment_keys) != 1 or "" in experiment_keys:
        raise ValueError(
            "The original rows must come from one non-empty experiment key; "
            f"found: {sorted(experiment_keys)}"
        )

    for metric in METRICS:
        selected[metric] = pd.to_numeric(
            selected[metric],
            errors="raise",
        )
        if not np.isfinite(selected[metric].to_numpy()).all():
            raise ValueError(
                f"{base_runs_path} contains non-finite {metric} values."
            )

    policies = set(
        selected["decision_policy"].fillna("").astype(str).str.lower()
    )
    if policies != {"argmax"}:
        raise ValueError(
            "The original comparison must use raw argmax for every "
            f"model; found decision policies: {sorted(policies)}"
        )
    test_sets = set(
        selected["test_data"].fillna("").astype(str)
    )
    if test_sets != {"KDDTest+"}:
        raise ValueError(
            f"Expected only KDDTest+ results, found: {sorted(test_sets)}"
        )

    expected_synth_counts: List[int] | None = None
    conv1d_protocol_by_seed: Dict[int, Dict[str, List[int]]] = {}
    normalized_rows: List[Dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        result_path = resolve_recorded_result_path(
            repo_root,
            results_dir,
            str(row["result_path"]),
        )
        recorded_metrics = read_metrics(result_path)
        for metric in METRICS:
            if not np.isclose(
                float(row[metric]),
                recorded_metrics[metric],
            ):
                raise ValueError(
                    f"{base_runs_path} disagrees with {result_path} for "
                    f"{row['model']} seed {row['seed']} metric {metric}: "
                    f"{row[metric]} vs {recorded_metrics[metric]}."
                )
        normalized_rows.append(
            {
                **row,
                **recorded_metrics,
                "result_path": str(result_path),
            }
        )
        metadata = read_metadata(result_path)

        if row["model"] not in NEURAL_BASE_MODELS:
            require_numeric_metadata(
                metadata,
                "seed",
                int(row["seed"]),
                result_path,
            )
            require_numeric_metadata(
                metadata,
                "val_split",
                args.val_split,
                result_path,
            )
            if metadata.get("test_data") != "KDDTest+":
                raise ValueError(
                    f"{result_path} was not evaluated on KDDTest+."
                )
            if metadata.get("decision_policy", "").lower() != "argmax":
                raise ValueError(
                    f"{result_path} did not use raw argmax predictions."
                )
            expected_weighting = (
                "balanced"
                if row["model"] == "xgboost_cost_sensitive"
                else "none"
            )
            if metadata.get("class_weighting") != expected_weighting:
                raise ValueError(
                    f"{result_path} has class_weighting="
                    f"{metadata.get('class_weighting')!r}, expected "
                    f"{expected_weighting!r}."
                )
            continue
        require_numeric_metadata(
            metadata,
            "seed",
            int(row["seed"]),
            result_path,
        )
        require_numeric_metadata(
            metadata,
            "epochs",
            args.epochs,
            result_path,
        )
        require_numeric_metadata(
            metadata,
            "val_split",
            args.val_split,
            result_path,
        )
        require_numeric_metadata(
            metadata,
            "focal_gamma",
            args.focal_gamma,
            result_path,
        )
        require_numeric_metadata(
            metadata,
            "cb_beta",
            args.cb_beta,
            result_path,
        )
        require_numeric_metadata(
            metadata,
            "minority_per_batch",
            args.minority_per_batch,
            result_path,
        )
        batch_key = (
            "global_batch_size"
            if "global_batch_size" in metadata
            else "batch_size"
        )
        require_numeric_metadata(
            metadata,
            batch_key,
            args.batch_size,
            result_path,
        )
        if metadata.get("feature_layout") != "optimized":
            raise ValueError(
                f"{result_path} did not use the optimized feature layout."
            )
        if metadata.get("thresholds_applied", "").lower() != "false":
            raise ValueError(
                f"{result_path} did not use raw argmax predictions."
            )

        synth_counts = parse_count_list(
            metadata.get("synth_counts", ""),
            result_path,
        )
        if expected_synth_counts is None:
            expected_synth_counts = synth_counts
        elif synth_counts != expected_synth_counts:
            raise ValueError(
                "The original neural models did not use identical cached "
                f"synthetic counts: {expected_synth_counts} vs "
                f"{synth_counts} in {result_path}."
            )

        if row["model"] == "conv1d":
            conv1d_protocol_by_seed[int(row["seed"])] = {
                field_name: parse_count_list(
                    metadata.get(field_name, ""),
                    result_path,
                    field_name,
                )
                for field_name in (
                    "real_counts",
                    "synth_counts",
                    "augmented_counts",
                    "alpha_counts",
                )
            }

    if expected_synth_counts is None:
        raise ValueError(
            "Could not recover synthetic class counts from the original "
            "neural runs."
        )
    if set(conv1d_protocol_by_seed) != expected_seeds:
        raise ValueError(
            "Could not recover the original Conv1D data/split counts for "
            f"every seed. Found {sorted(conv1d_protocol_by_seed)}."
        )

    return (
        pd.DataFrame(normalized_rows),
        expected_synth_counts,
        conv1d_protocol_by_seed,
    )


def current_synth_counts(synth_path: Path) -> List[int]:
    labels = pd.to_numeric(
        pd.read_csv(synth_path, usecols=["class"])["class"],
        errors="raise",
    )
    if labels.isna().any():
        raise ValueError(f"Missing class labels in {synth_path}.")
    integer_labels = labels.astype(int)
    if not bool((labels == integer_labels).all()):
        raise ValueError(
            f"Synthetic class labels must be integers in {synth_path}."
        )
    invalid = sorted(set(integer_labels.tolist()) - set(range(5)))
    if invalid:
        raise ValueError(
            f"Invalid synthetic class IDs in {synth_path}: {invalid}"
        )
    return (
        np.bincount(integer_labels.to_numpy(), minlength=5)
        .astype(int)
        .tolist()
    )


def build_transformer_command(
    repo_root: Path,
    run_name: str,
    seed: int,
    synth_path: Path,
    args: argparse.Namespace,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(repo_root / "src" / "cnn_opt_1d_4gpu.py"),
        "--run-name",
        run_name,
        "--architecture",
        "transformer",
        "--num-gpus",
        "1",
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
        "--dense-units",
        str(args.transformer_dense_units),
        "--dropout2",
        str(args.transformer_head_dropout),
        "--d-model",
        str(args.d_model),
        "--num-heads",
        str(args.num_heads),
        "--transformer-blocks",
        str(args.transformer_blocks),
        "--ff-dim",
        str(args.ff_dim),
        "--transformer-dropout",
        str(args.transformer_dropout),
        "--no-thresholds",
    ]


def transformer_result_is_complete(
    result_path: Path,
    run_name: str,
    seed: int,
    expected_synth_counts: Sequence[int] | None,
    expected_protocol_counts: Dict[str, List[int]] | None,
    args: argparse.Namespace,
) -> bool:
    if not result_path.exists():
        return False
    try:
        metadata = read_metadata(result_path)
        if metadata.get("run_name") != run_name:
            return False
        if metadata.get("architecture") != "transformer":
            return False
        checks = {
            "seed": seed,
            "epochs": args.epochs,
            "val_split": args.val_split,
            "num_gpus": 1,
            "global_batch_size": args.batch_size,
            "focal_gamma": args.focal_gamma,
            "cb_beta": args.cb_beta,
            "minority_per_batch": args.minority_per_batch,
            "dense_units": args.transformer_dense_units,
            "dropout2": args.transformer_head_dropout,
            "d_model": args.d_model,
            "num_heads": args.num_heads,
            "transformer_blocks": args.transformer_blocks,
            "ff_dim": args.ff_dim,
            "transformer_dropout": args.transformer_dropout,
            "Model Parameters": args.expected_transformer_parameters,
        }
        for key, expected in checks.items():
            require_numeric_metadata(
                metadata,
                key,
                float(expected),
                result_path,
            )
        if metadata.get("feature_layout") != "optimized":
            return False
        if metadata.get("thresholds_applied", "").lower() != "false":
            return False
        if expected_synth_counts is not None:
            observed_counts = parse_count_list(
                metadata.get("synth_counts", ""),
                result_path,
            )
            if observed_counts != list(expected_synth_counts):
                return False
        if expected_protocol_counts is not None:
            for field_name, expected_counts in (
                expected_protocol_counts.items()
            ):
                observed_counts = parse_count_list(
                    metadata.get(field_name, ""),
                    result_path,
                    field_name,
                )
                if observed_counts != expected_counts:
                    return False
        read_metrics(result_path)
        return True
    except (OSError, TypeError, ValueError):
        return False


def format_metric(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}% +/- {100.0 * std:.2f}%"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    parser = argparse.ArgumentParser(
        description=(
            "Train five Transformer seeds and add them to the original "
            "parameter-matched five-model comparison."
        )
    )
    parser.add_argument(
        "--base-runs",
        default="results/model5_param_matched_paired_runs.csv",
        help=(
            "Original per-seed five-model CSV. The summary CSV is not used, "
            "so manual summary edits cannot affect the new table."
        ),
    )
    parser.add_argument("--name-prefix", default="model6_param_matched")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument(
        "--synth-path",
        default="data/synth_ctgan_5class.csv",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--cb-beta", type=float, default=0.9999)
    parser.add_argument("--minority-per-batch", type=int, default=1)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument(
        "--transformer-dense-units",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--transformer-dropout",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--transformer-head-dropout",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--expected-transformer-parameters",
        type=int,
        default=110_661,
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be greater than zero.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between zero and one.")
    if args.minority_per_batch <= 0:
        parser.error("--minority-per-batch must be greater than zero.")
    if args.d_model <= 0:
        parser.error("--d-model must be greater than zero.")
    if args.num_heads <= 0 or args.d_model % args.num_heads != 0:
        parser.error("--num-heads must be positive and divide --d-model.")
    if args.transformer_blocks <= 0 or args.ff_dim <= 0:
        parser.error(
            "--transformer-blocks and --ff-dim must be greater than zero."
        )
    if args.transformer_dense_units <= 0:
        parser.error("--transformer-dense-units must be greater than zero.")
    if not 0.0 <= args.transformer_dropout < 1.0:
        parser.error("--transformer-dropout must be in [0, 1).")
    if not 0.0 <= args.transformer_head_dropout < 1.0:
        parser.error("--transformer-head-dropout must be in [0, 1).")
    if args.expected_transformer_parameters <= 0:
        parser.error(
            "--expected-transformer-parameters must be greater than zero."
        )
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))

    synth_path = resolve_path(repo_root, args.synth_path)
    base_runs_path = resolve_path(repo_root, args.base_runs)
    if not args.dry_run:
        if not synth_path.exists():
            raise SystemExit(f"Cached CTGAN file not found: {synth_path}")
        if not base_runs_path.exists():
            raise SystemExit(
                f"Original per-seed comparison not found: {base_runs_path}"
            )

    settings_identity = (
        tuple(args.seeds),
        args.epochs,
        args.batch_size,
        args.val_split,
        args.focal_gamma,
        args.cb_beta,
        args.minority_per_batch,
        args.d_model,
        args.num_heads,
        args.transformer_blocks,
        args.ff_dim,
        args.transformer_dense_units,
        args.transformer_dropout,
        args.transformer_head_dropout,
        args.expected_transformer_parameters,
        fingerprint_files(
            [
                repo_root / "src" / "cnn_opt_1d_4gpu.py",
                repo_root / "src" / "cnn_opt.py",
                repo_root / "src" / "cnn_gan_foc.py",
                repo_root / "data" / "KDDTrain+.txt",
                repo_root / "data" / "KDDTest+.txt",
                synth_path,
                base_runs_path,
            ]
        ),
    )
    experiment_key = hashlib.sha256(
        repr(settings_identity).encode("utf-8")
    ).hexdigest()[:12]
    prefix = args.name_prefix.strip() or "model6_param_matched"

    plans: List[Dict[str, Any]] = []
    for seed in args.seeds:
        run_name = f"{prefix}_transformer_{experiment_key}_s{seed}"
        plans.append(
            {
                "model": "transformer",
                "seed": int(seed),
                "run_name": run_name,
                "result_path": (
                    results_dir / f"{run_name}_results.txt"
                ),
            }
        )

    print("Purpose: append Transformer to the original five-model table")
    print(f"Seeds: {args.seeds}")
    print(f"GPUs: {gpus}")
    print(f"Transformer runs: {len(plans)}")
    print(
        "Transformer: "
        f"d_model={args.d_model}, heads={args.num_heads}, "
        f"blocks={args.transformer_blocks}, ff_dim={args.ff_dim}, "
        f"dense={args.transformer_dense_units}, "
        f"parameters={args.expected_transformer_parameters}"
    )
    print("Decision policy: raw argmax")
    print(f"Experiment key: {experiment_key}")

    if args.dry_run:
        for plan in plans:
            command = build_transformer_command(
                repo_root,
                plan["run_name"],
                plan["seed"],
                synth_path,
                args,
            )
            print(
                f"[seed={plan['seed']}] {shlex.join(command)}"
            )
        print("Dry run complete; nothing was trained or written.")
        return

    (
        base_runs,
        expected_synth_counts,
        protocol_counts_by_seed,
    ) = validate_base_runs(
        base_runs_path,
        repo_root,
        results_dir,
        args.seeds,
        args,
    )
    observed_synth_counts = current_synth_counts(synth_path)
    if observed_synth_counts != expected_synth_counts:
        raise SystemExit(
            "The current cached CTGAN CSV does not match the synthetic "
            "class counts used by the original neural runs.\n"
            f"Original counts: {expected_synth_counts}\n"
            f"Current counts:  {observed_synth_counts}\n"
            "Use the exact original CTGAN CSV before adding Transformer."
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / f"{prefix}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plan_path = results_dir / f"{prefix}_transformer_plan.csv"
    pd.DataFrame(
        [
            {
                **plan,
                "experiment_key": experiment_key,
                "gpu": gpus[index % len(gpus)],
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "val_split": args.val_split,
                "focal_gamma": args.focal_gamma,
                "cb_beta": args.cb_beta,
                "minority_per_batch": args.minority_per_batch,
                "synth_path": str(synth_path),
                "synth_counts": expected_synth_counts,
                "d_model": args.d_model,
                "num_heads": args.num_heads,
                "transformer_blocks": args.transformer_blocks,
                "ff_dim": args.ff_dim,
                "dense_units": args.transformer_dense_units,
                "transformer_dropout": args.transformer_dropout,
                "head_dropout": args.transformer_head_dropout,
                "expected_parameters":
                    args.expected_transformer_parameters,
            }
            for index, plan in enumerate(plans)
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

    def gpu_worker(
        gpu: str,
        gpu_plans: List[Dict[str, Any]],
    ) -> None:
        for plan in gpu_plans:
            result_path = plan["result_path"]
            complete = transformer_result_is_complete(
                result_path,
                plan["run_name"],
                plan["seed"],
                expected_synth_counts,
                protocol_counts_by_seed[plan["seed"]],
                args,
            )
            if complete and not args.rerun:
                with print_lock:
                    print(
                        f"[GPU {gpu}] SKIP {plan['run_name']}",
                        flush=True,
                    )
                continue

            command = build_transformer_command(
                repo_root,
                plan["run_name"],
                plan["seed"],
                synth_path,
                args,
            )
            log_path = log_dir / f"{plan['run_name']}.log"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONHASHSEED"] = str(plan["seed"])
            environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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
            complete = transformer_result_is_complete(
                result_path,
                plan["run_name"],
                plan["seed"],
                expected_synth_counts,
                protocol_counts_by_seed[plan["seed"]],
                args,
            )

            with data_lock:
                runtimes[plan["run_name"]] = runtime
                if completed.returncode != 0 or not complete:
                    failures.append(
                        f"{plan['run_name']}: "
                        f"exit={completed.returncode}, log={log_path}"
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

    if failures:
        print(f"\nFailed Transformer runs: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(
            "Transformer comparison is incomplete. Fix the failed runs and "
            "rerun the same command; completed seeds will be skipped."
        )

    transformer_rows: List[Dict[str, Any]] = []
    for plan in plans:
        if not transformer_result_is_complete(
            plan["result_path"],
            plan["run_name"],
            plan["seed"],
            expected_synth_counts,
            protocol_counts_by_seed[plan["seed"]],
            args,
        ):
            raise SystemExit(
                f"Incomplete Transformer result: {plan['result_path']}"
            )
        transformer_rows.append(
            {
                "experiment_key": experiment_key,
                "model": "transformer",
                "seed": plan["seed"],
                "run_name": plan["run_name"],
                "dense_units": args.transformer_dense_units,
                "synth_class_counts": {
                    class_id: count
                    for class_id, count in enumerate(
                        expected_synth_counts
                    )
                    if count
                },
                "train_data": "KDDTrain+ + cached CTGAN CSV",
                "imbalance_method": (
                    "class-balanced focal loss + "
                    "minority-guaranteed batches"
                ),
                "decision_policy": "argmax",
                "test_data": "KDDTest+",
                **read_metrics(plan["result_path"]),
                "runtime_seconds": runtimes.get(plan["run_name"]),
                "result_path": str(plan["result_path"]),
                "paired_complete": True,
            }
        )

    combined = pd.concat(
        [base_runs, pd.DataFrame(transformer_rows)],
        ignore_index=True,
        sort=False,
    )
    model_order = {
        model_name: index
        for index, model_name in enumerate(ALL_MODELS)
    }
    combined["_model_order"] = combined["model"].map(model_order)
    combined = (
        combined.sort_values(["seed", "_model_order"])
        .drop(columns=["_model_order"])
        .reset_index(drop=True)
    )
    combined["paired_complete"] = True

    paired_runs_path = results_dir / f"{prefix}_paired_runs.csv"
    combined.to_csv(paired_runs_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for model_name in ALL_MODELS:
        model_runs = combined[combined["model"] == model_name]
        if len(model_runs) != len(args.seeds):
            raise ValueError(
                f"{model_name} has {len(model_runs)} runs; "
                f"expected {len(args.seeds)}."
            )
        summary: Dict[str, Any] = {
            "model": model_name,
            "paired_seeds": len(model_runs),
        }
        for metric in METRICS:
            values = pd.to_numeric(
                model_runs[metric],
                errors="raise",
            )
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        parameters = pd.to_numeric(
            model_runs.get("model_parameters"),
            errors="coerce",
        ).dropna()
        summary["model_parameters_mean"] = (
            float(parameters.mean()) if len(parameters) else None
        )
        summary_rows.append(summary)

    summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    formatted_rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        formatted_rows.append(
            {
                "Model": DISPLAY_NAMES[row["model"]],
                "Runs": int(row["paired_seeds"]),
                "Accuracy": format_metric(
                    row["accuracy_mean"],
                    row["accuracy_std"],
                ),
                "MCC": format_metric(
                    row["mcc_mean"],
                    row["mcc_std"],
                ),
                "Macro-F1": format_metric(
                    row["macro_f1_mean"],
                    row["macro_f1_std"],
                ),
                "Macro Recall": format_metric(
                    row["macro_recall_mean"],
                    row["macro_recall_std"],
                ),
                "R2L Recall": format_metric(
                    row["r2l_recall_mean"],
                    row["r2l_recall_std"],
                ),
                "U2R Recall": format_metric(
                    row["u2r_recall_mean"],
                    row["u2r_recall_std"],
                ),
            }
        )
    formatted = pd.DataFrame(formatted_rows)
    formatted_path = results_dir / f"{prefix}_summary_formatted.csv"
    formatted.to_csv(formatted_path, index=False)
    text_path = results_dir / f"{prefix}_summary.txt"
    text_path.write_text(
        (
            "Six-model parameter-matched KDDTest+ comparison\n"
            f"Seeds: {args.seeds}\n"
            "All predictions: raw argmax\n"
            "Neural validation: historical augmented 80/20 split\n"
            "Standard deviation: sample SD across paired seeds\n\n"
            f"{formatted.to_string(index=False)}\n"
        ),
        encoding="utf-8",
    )

    print("\n=== Six-model KDDTest+ comparison ===")
    print(formatted.to_string(index=False))
    print(f"\nTransformer plan: {plan_path}")
    print(f"All paired runs: {paired_runs_path}")
    print(f"Numeric summary: {summary_path}")
    print(f"Formatted summary: {formatted_path}")
    print(f"Readable table: {text_path}")


if __name__ == "__main__":
    main()
