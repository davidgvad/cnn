"""
Exhaustive Conv2D hyperparameter search using one independent job per GPU.

The expensive grid dimensions (R2L synthetic rows, beta, gamma) retrain the
CNN. Each trained model scores every R2L/U2R coefficient pair on its real
validation predictions without retraining. The same full hyperparameter
combination is then averaged across all requested seeds before ranking.

Typical run from the repository root:

    python -u src/grid_search_cnn_opt_4gpu.py \
      --gpus 0 1 2 3 \
      --seeds 0 1 2 \
      --prepare-synth
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from summarize_trials import parse_results_file


R2L_TRAIN_ATTACKS = {
    "warezclient",
    "guess_passwd",
    "warezmaster",
    "imap",
    "ftp_write",
    "multihop",
    "phf",
    "spy",
}

DEFAULT_R2L_SYNTH_COUNTS = "5000"
DEFAULT_CB_BETAS = (
    "0.9,0.95,0.975,0.99,0.995,0.999,0.9995,0.9999,0.99999"
)
DEFAULT_FOCAL_GAMMAS = (
    "0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,3.0"
)
DEFAULT_COEFFICIENT_VALUES = (
    "0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.25,"
    "0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.80,"
    "0.95,1.00,1.10,1.25,1.40,1.55,1.70,1.85,2.00"
)

VALIDATION_METRICS = [
    "selected_validation_minimum_minority_recall",
    "selected_validation_minority_recall",
    "selected_validation_rare_f1",
    "selected_validation_macro_f1",
    "selected_validation_macro_recall",
    "selected_validation_accuracy",
    "selected_validation_mcc",
]

REPORT_METRICS = [
    *VALIDATION_METRICS,
    "r2l_score_coefficient",
    "u2r_score_coefficient",
    "macro_f1",
    "macro_recall",
    "R2L_recall",
    "U2R_recall",
    "mcc",
    "accuracy",
]

CONFIG_FIELDS = [
    "config_id",
    "config_key",
    "r2l_synth_count",
    "target_r2l",
    "cb_beta",
    "focal_gamma",
]

COEFFICIENT_FIELDS = [
    "r2l_score_coefficient",
    "u2r_score_coefficient",
]

COEFFICIENT_GRID_METRICS = [
    "minimum_allowed_macro_f1",
    "meets_macro_f1_retention",
    "distance_from_argmax",
    "accuracy",
    "mcc",
    "macro_f1",
    "macro_recall",
    "r2l_recall",
    "u2r_recall",
    "minority_recall",
    "minimum_minority_recall",
    "minority_recall_gap",
    "r2l_f1",
    "u2r_f1",
    "rare_f1",
    "changed_predictions",
    "change_rate",
]

SHARED_RANK_COLUMNS = [
    "meets_macro_f1_retention_all_seeds",
    "minimum_minority_recall_mean",
    "minority_recall_mean",
    "rare_f1_mean",
    "minority_recall_gap_mean",
    "macro_f1_mean",
    "mcc_mean",
    "accuracy_mean",
    "distance_from_argmax_mean",
]

SHARED_RANK_ASCENDING = [
    False,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
]


def parse_int_csv(raw: str, option_name: str) -> List[int]:
    try:
        values = [
            int(value.strip())
            for value in raw.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must contain comma-separated integers."
        ) from exc
    if not values:
        raise ValueError(f"{option_name} cannot be empty.")
    if len(values) != len(set(values)):
        raise ValueError(f"{option_name} cannot contain duplicates.")
    return values


def parse_float_csv(raw: str, option_name: str) -> List[float]:
    try:
        values = [
            float(value.strip())
            for value in raw.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            f"{option_name} must contain comma-separated numbers."
        ) from exc
    if not values:
        raise ValueError(f"{option_name} cannot be empty.")
    if any(not np.isfinite(value) for value in values):
        raise ValueError(f"{option_name} values must be finite.")
    if len(values) != len(set(values)):
        raise ValueError(f"{option_name} cannot contain duplicates.")
    return values


def parse_gpus(values: Sequence[str]) -> List[str]:
    gpus: List[str] = []
    for value in values:
        gpus.extend(
            part.strip() for part in value.split(",") if part.strip()
        )
    if not gpus:
        raise ValueError("Provide at least one GPU ID.")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU IDs must be unique, got {gpus}.")
    return gpus


def slug(value: int | float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


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


def count_real_r2l(train_path: Path) -> int:
    raw_labels = pd.read_csv(
        train_path,
        header=None,
        usecols=[41],
    ).iloc[:, 0]
    normalized = raw_labels.astype(str).str.rstrip(".")
    return int(normalized.isin(R2L_TRAIN_ATTACKS).sum())


def count_pool_r2l(synth_path: Path) -> int:
    try:
        labels = pd.read_csv(synth_path, usecols=["class"])["class"]
    except ValueError as exc:
        raise ValueError(
            f"Synthetic pool has no 'class' column: {synth_path}"
        ) from exc
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.isna().any():
        raise ValueError("Synthetic pool contains non-numeric class labels.")
    return int((numeric == 2).sum())


def prepare_synth_pool(
    repo_root: Path,
    synth_path: Path,
    real_r2l_count: int,
    max_synth_count: int,
    gpu: str,
    args: argparse.Namespace,
) -> None:
    final_target = real_r2l_count + max_synth_count
    command = [
        sys.executable,
        "-u",
        str(repo_root / "src" / "cnn_opt.py"),
        "--ctgan-train",
        "--ctgan-regenerate",
        "--ctgan-only",
        "--synth-path",
        str(synth_path),
        "--target-r2l",
        str(final_target),
        "--target-u2r",
        "0",
        "--seed",
        str(args.ctgan_seed),
        "--ctgan-epochs",
        str(args.ctgan_epochs),
        "--ctgan-batch-size",
        str(args.ctgan_batch_size),
        "--ctgan-pac",
        str(args.ctgan_pac),
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    print("\nPreparing the shared maximum R2L synthetic pool:")
    print(shlex.join(command), flush=True)
    subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=True,
    )


def required_float(data: Dict[str, Any], key: str) -> float:
    value = data.get(key)
    if value is None:
        raise ValueError(f"Missing result field: {key}")
    return float(value)


def result_is_complete(
    path: Path,
    plan: Dict[str, Any],
) -> bool:
    if not path.exists():
        return False
    try:
        data = parse_results_file(path).data
        if data.get("run_name") != plan["run_name"]:
            return False
        checks = {
            "seed": plan["seed"],
            "focal_gamma": plan["focal_gamma"],
            "cb_beta": plan["cb_beta"],
            "target_r2l": plan["target_r2l"],
            "r2l_synthetic_rows_used": plan["r2l_synth_count"],
        }
        for key, expected in checks.items():
            if not np.isclose(required_float(data, key), float(expected)):
                return False
        for key in [
            "selected_validation_minimum_minority_recall",
            "selected_validation_rare_f1",
            "selected_validation_macro_f1",
            "r2l_score_coefficient",
            "u2r_score_coefficient",
            "Test Macro F1",
        ]:
            required_float(data, key)
        coefficient_pair_count = int(
            required_float(data, "coefficient_pairs_searched")
        )
        search_path = (
            path.parent
            / f"{plan['run_name']}_score_scaling_search.csv"
        )
        if not search_path.exists():
            return False
        coefficient_pairs = pd.read_csv(
            search_path,
            usecols=COEFFICIENT_FIELDS,
        )
        if (
            len(coefficient_pairs) != coefficient_pair_count
            or coefficient_pairs.duplicated().any()
        ):
            return False
        return True
    except (OSError, TypeError, ValueError, pd.errors.ParserError):
        return False


def build_command(
    repo_root: Path,
    plan: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    return [
        sys.executable,
        "-u",
        str(repo_root / "src" / "cnn_opt.py"),
        "--run-name",
        plan["run_name"],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(plan["seed"]),
        "--val-split",
        str(args.val_split),
        "--synth-path",
        str(args.synth_pool),
        "--target-r2l",
        str(plan["target_r2l"]),
        "--target-u2r",
        "0",
        "--focal-gamma",
        str(plan["focal_gamma"]),
        "--cb-beta",
        str(plan["cb_beta"]),
        "--groups",
        str(args.groups),
        "--base-filters",
        str(args.base_filters),
        "--dense-units",
        str(args.dense_units),
        "--dropout1",
        str(args.dropout1),
        "--dropout2",
        str(args.dropout2),
        "--minority-per-batch",
        str(args.minority_per_batch),
        "--feature-layout",
        "optimized",
        "--coefficient-values",
        args.coefficient_values,
        "--min-validation-macro-f1-retention",
        str(args.min_validation_macro_f1_retention),
    ]


def parse_completed_run(
    result_path: Path,
    plan: Dict[str, Any],
    runtime_seconds: float | None,
    gpu: str | None,
) -> Dict[str, Any]:
    data = parse_results_file(result_path).data
    row: Dict[str, Any] = {
        **plan,
        "gpu": gpu,
        "runtime_seconds": runtime_seconds,
        "result_path": str(result_path),
    }
    source_keys = {
        "selected_validation_minimum_minority_recall":
            "selected_validation_minimum_minority_recall",
        "selected_validation_minority_recall":
            "selected_validation_minority_recall",
        "selected_validation_rare_f1": "selected_validation_rare_f1",
        "selected_validation_macro_f1": "selected_validation_macro_f1",
        "selected_validation_macro_recall":
            "selected_validation_macro_recall",
        "selected_validation_accuracy": "selected_validation_accuracy",
        "selected_validation_mcc": "selected_validation_mcc",
        "r2l_score_coefficient": "r2l_score_coefficient",
        "u2r_score_coefficient": "u2r_score_coefficient",
        "macro_f1": "Test Macro F1",
        "macro_recall": "Test Macro Recall",
        "R2L_recall": "R2L Recall",
        "U2R_recall": "U2R Recall",
        "mcc": "MCC",
        "accuracy": "Test Accuracy (sklearn)",
    }
    for output_key, source_key in source_keys.items():
        row[output_key] = required_float(data, source_key)
    return row


def mean_and_std(values: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None, None
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    return mean, std


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key, group in raw.groupby(
        CONFIG_FIELDS,
        dropna=False,
        sort=False,
    ):
        row = dict(zip(CONFIG_FIELDS, key))
        row["completed_runs"] = len(group)
        for metric in REPORT_METRICS:
            mean, std = mean_and_std(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        by=[
            "selected_validation_minimum_minority_recall_mean",
            "selected_validation_minority_recall_mean",
            "selected_validation_rare_f1_mean",
            "selected_validation_macro_f1_mean",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def load_coefficient_grid(
    results_dir: Path,
    plan: Dict[str, Any],
    expected_coefficient_values: Sequence[float],
) -> pd.DataFrame:
    """Load and validate every coefficient pair from one seed."""
    search_path = (
        results_dir
        / f"{plan['run_name']}_score_scaling_search.csv"
    )
    search = pd.read_csv(search_path)
    required_columns = set(
        COEFFICIENT_FIELDS + COEFFICIENT_GRID_METRICS
    )
    missing = sorted(required_columns - set(search.columns))
    if missing:
        raise ValueError(
            f"{search_path} is missing columns: {missing}"
        )

    selected_columns = COEFFICIENT_FIELDS + COEFFICIENT_GRID_METRICS
    search = search[selected_columns].copy()
    for column in selected_columns:
        search[column] = pd.to_numeric(search[column], errors="raise")
        if not np.isfinite(search[column].to_numpy()).all():
            raise ValueError(
                f"{search_path} contains a non-finite {column} value."
            )

    expected_pairs = {
        (float(r2l), float(u2r))
        for r2l, u2r in itertools.product(
            expected_coefficient_values,
            repeat=2,
        )
    }
    observed_pairs = set(
        search[COEFFICIENT_FIELDS]
        .itertuples(index=False, name=None)
    )
    if len(search) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError(
            f"{search_path} does not contain the expected "
            f"{len(expected_pairs)} unique coefficient pairs."
        )

    for field in CONFIG_FIELDS:
        search[field] = plan[field]
    search["seed"] = int(plan["seed"])
    search["run_name"] = plan["run_name"]
    return search


def build_shared_coefficient_summary(
    coefficient_rows: pd.DataFrame,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    """
    Average each complete hyperparameter combination over the same seeds.

    A full combination includes R2L augmentation, beta, gamma, and both
    class-score coefficients. Incomplete combinations are not ranked.
    """
    group_fields = CONFIG_FIELDS + COEFFICIENT_FIELDS
    expected_seed_set = {int(seed) for seed in expected_seeds}
    rows: List[Dict[str, Any]] = []

    for key, group in coefficient_rows.groupby(
        group_fields,
        dropna=False,
        sort=False,
    ):
        observed_seeds = set(group["seed"].astype(int))
        if (
            observed_seeds != expected_seed_set
            or len(group) != len(expected_seed_set)
        ):
            continue

        row = dict(zip(group_fields, key))
        row["full_config_key"] = hashlib.sha256(
            (
                f"{row['config_key']}|"
                f"{float(row['r2l_score_coefficient']):.10g}|"
                f"{float(row['u2r_score_coefficient']):.10g}"
            ).encode("utf-8")
        ).hexdigest()[:12]
        row["completed_seeds"] = len(observed_seeds)
        row["seeds"] = ",".join(
            str(seed) for seed in sorted(observed_seeds)
        )
        for metric in COEFFICIENT_GRID_METRICS:
            mean, std = mean_and_std(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        row["meets_mean_macro_f1_retention"] = bool(
            row["macro_f1_mean"]
            >= row["minimum_allowed_macro_f1_mean"] - 1e-12
        )
        row["meets_macro_f1_retention_all_seeds"] = bool(
            np.all(
                np.isclose(
                    group["meets_macro_f1_retention"].to_numpy(),
                    1.0,
                )
            )
        )
        rows.append(row)

    if not rows:
        raise ValueError(
            "No hyperparameter combination has results for every "
            f"requested seed: {sorted(expected_seed_set)}"
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        by=SHARED_RANK_COLUMNS,
        ascending=SHARED_RANK_ASCENDING,
    ).reset_index(drop=True)


def select_best_pair_per_training_config(
    coefficient_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Keep the best shared coefficient pair for each trained configuration."""
    best = (
        coefficient_summary
        .groupby(CONFIG_FIELDS, dropna=False, sort=False)
        .head(1)
        .copy()
    )
    best = best.sort_values(
        by=SHARED_RANK_COLUMNS,
        ascending=SHARED_RANK_ASCENDING,
    ).reset_index(drop=True)
    best.insert(0, "validation_rank", np.arange(1, len(best) + 1))
    return best


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    parser = argparse.ArgumentParser(
        description=(
            "Exhaustive cnn_opt Conv2D search with one independent job "
            "per GPU."
        )
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--name-prefix", default="cnn2d_grid")
    parser.add_argument(
        "--r2l-synth-counts",
        default=DEFAULT_R2L_SYNTH_COUNTS,
        help="Numbers of R2L rows added, not final class totals.",
    )
    parser.add_argument("--cb-betas", default=DEFAULT_CB_BETAS)
    parser.add_argument("--focal-gammas", default=DEFAULT_FOCAL_GAMMAS)
    parser.add_argument(
        "--coefficient-values",
        default=DEFAULT_COEFFICIENT_VALUES,
    )
    parser.add_argument(
        "--min-validation-macro-f1-retention",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--synth-pool",
        default="data/synth_ctgan_r2l_generated5000.csv",
    )
    parser.add_argument("--prepare-synth", action="store_true")
    parser.add_argument("--regenerate-synth", action="store_true")
    parser.add_argument("--ctgan-seed", type=int, default=0)
    parser.add_argument("--ctgan-epochs", type=int, default=200)
    parser.add_argument("--ctgan-batch-size", type=int, default=4096)
    parser.add_argument("--ctgan-pac", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout1", type=float, default=0.25)
    parser.add_argument("--dropout2", type=float, default=0.30)
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-all-commands", action="store_true")
    args = parser.parse_args()

    try:
        gpus = parse_gpus(args.gpus)
        r2l_synth_counts = parse_int_csv(
            args.r2l_synth_counts,
            "--r2l-synth-counts",
        )
        cb_betas = parse_float_csv(args.cb_betas, "--cb-betas")
        focal_gammas = parse_float_csv(
            args.focal_gammas,
            "--focal-gammas",
        )
        coefficient_values = parse_float_csv(
            args.coefficient_values,
            "--coefficient-values",
        )
    except ValueError as error:
        parser.error(str(error))

    if any(count < 0 for count in r2l_synth_counts):
        parser.error("--r2l-synth-counts values must be 0 or greater.")
    if any(not 0.0 < beta < 1.0 for beta in cb_betas):
        parser.error("--cb-betas values must be between 0 and 1.")
    if any(gamma < 0.0 for gamma in focal_gammas):
        parser.error("--focal-gammas values must be 0 or greater.")
    if any(value <= 0.0 for value in coefficient_values):
        parser.error("--coefficient-values must be greater than zero.")
    if 1.0 not in coefficient_values:
        parser.error("--coefficient-values must include 1.0.")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if args.regenerate_synth and not args.prepare_synth:
        parser.error("--regenerate-synth requires --prepare-synth.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be greater than zero.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1.")
    if args.groups <= 0 or args.base_filters % args.groups != 0:
        parser.error("--groups must divide --base-filters.")
    if args.dense_units <= 0 or args.minority_per_batch < 0:
        parser.error(
            "--dense-units must be positive and "
            "--minority-per-batch cannot be negative."
        )
    if not (
        0.0 <= args.min_validation_macro_f1_retention <= 1.0
    ):
        parser.error(
            "--min-validation-macro-f1-retention must be between 0 and 1."
        )

    synth_path = Path(args.synth_pool).expanduser()
    if not synth_path.is_absolute():
        synth_path = repo_root / synth_path
    synth_path = synth_path.resolve()
    args.synth_pool = str(synth_path)

    train_path = repo_root / "data" / "KDDTrain+.txt"
    real_r2l_count = count_real_r2l(train_path)
    max_synth_count = max(r2l_synth_counts)

    should_prepare = args.prepare_synth and (
        not synth_path.exists() or args.regenerate_synth
    )
    if should_prepare and not args.dry_run:
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_synth_pool(
            repo_root,
            synth_path,
            real_r2l_count,
            max_synth_count,
            gpus[0],
            args,
        )

    if not args.dry_run:
        if not synth_path.exists():
            raise SystemExit(
                f"Synthetic pool not found: {synth_path}\n"
                "Run again with --prepare-synth."
            )
        available_r2l = count_pool_r2l(synth_path)
        if available_r2l < max_synth_count:
            raise SystemExit(
                f"Synthetic pool has {available_r2l} R2L rows, but this "
                f"grid needs {max_synth_count}.\n"
                "Run with --prepare-synth --regenerate-synth to rebuild it."
            )

    experiment_key = fingerprint_files(
        [
            repo_root / "src" / "cnn_opt.py",
            repo_root / "src" / "cnn_gan_foc.py",
            repo_root / "src" / "grid_search_cnn_opt_4gpu.py",
            train_path,
            repo_root / "data" / "KDDTest+.txt",
            synth_path,
        ]
    )

    combinations = list(
        itertools.product(
            r2l_synth_counts,
            cb_betas,
            focal_gammas,
        )
    )
    configs: List[Dict[str, Any]] = []
    for config_number, combination in enumerate(combinations, start=1):
        synth_count, beta, gamma = combination
        target_r2l = real_r2l_count + synth_count
        identity = (
            synth_count,
            target_r2l,
            beta,
            gamma,
            tuple(coefficient_values),
            args.min_validation_macro_f1_retention,
            args.epochs,
            args.batch_size,
            args.val_split,
            args.groups,
            args.base_filters,
            args.dense_units,
            args.dropout1,
            args.dropout2,
            args.minority_per_batch,
            experiment_key,
        )
        config_key = hashlib.sha256(
            repr(identity).encode("utf-8")
        ).hexdigest()[:10]
        configs.append(
            {
                "config_id": f"c{config_number:03d}",
                "config_key": config_key,
                "experiment_key": experiment_key,
                "r2l_synth_count": synth_count,
                "target_r2l": target_r2l,
                "cb_beta": beta,
                "focal_gamma": gamma,
            }
        )

    plans: List[Dict[str, Any]] = []
    for config in configs:
        for seed in args.seeds:
            run_name = (
                f"{args.name_prefix}_{config['config_id']}_"
                f"{config['config_key']}_r{config['r2l_synth_count']}"
                f"_b{slug(config['cb_beta'])}"
                f"_g{slug(config['focal_gamma'])}_s{seed}"
            )
            plans.append(
                {
                    **config,
                    "seed": seed,
                    "run_name": run_name,
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / f"{args.name_prefix}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plan_path = results_dir / f"{args.name_prefix}_plan.csv"
    pd.DataFrame(plans).to_csv(plan_path, index=False)

    coefficient_pair_count = len(coefficient_values) ** 2
    waves = int(np.ceil(len(plans) / len(gpus)))
    print(f"Real KDDTrain+ R2L rows: {real_r2l_count}")
    print(f"R2L synthetic amounts: {r2l_synth_counts}")
    print(f"Betas: {cb_betas}")
    print(f"Gammas: {focal_gammas}")
    print(
        f"Training configurations per seed: {len(configs)}; "
        f"total runs: {len(plans)}"
    )
    print(
        f"Score coefficients: {len(coefficient_values)} values, "
        f"{coefficient_pair_count} validation pairs per trained model"
    )
    print(
        "Full hyperparameter combinations after seed averaging: "
        f"{len(configs) * coefficient_pair_count}"
    )
    print(f"GPUs: {gpus}; approximately {waves} waves")
    print(f"Plan: {plan_path}")

    if args.dry_run:
        shown = plans if args.print_all_commands else plans[:8]
        for index, plan in enumerate(shown):
            gpu = gpus[index % len(gpus)]
            print(f"[GPU {gpu}] {shlex.join(build_command(repo_root, plan, args))}")
        if len(shown) < len(plans):
            print(
                f"... {len(plans) - len(shown)} more commands are stored "
                "in the plan."
            )
        print("Dry run complete; nothing was trained.")
        return

    task_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    for plan in plans:
        task_queue.put(plan)

    print_lock = threading.Lock()
    data_lock = threading.Lock()
    failures: List[str] = []
    statuses: Dict[str, str] = {}
    runtimes: Dict[str, float] = {}
    assigned_gpus: Dict[str, str] = {}

    def gpu_worker(gpu: str) -> None:
        while True:
            try:
                plan = task_queue.get_nowait()
            except queue.Empty:
                return

            run_name = plan["run_name"]
            result_path = results_dir / f"{run_name}_results.txt"
            if result_is_complete(result_path, plan):
                with data_lock:
                    statuses[run_name] = "skipped_complete"
                with print_lock:
                    print(f"[GPU {gpu}] SKIP {run_name}", flush=True)
                task_queue.task_done()
                continue

            command = build_command(repo_root, plan, args)
            log_path = log_dir / f"{run_name}.log"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONHASHSEED"] = str(plan["seed"])
            environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

            with print_lock:
                print(f"[GPU {gpu}] START {run_name}", flush=True)

            start = time.perf_counter()
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n\n=== attempt {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                )
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
            complete = result_is_complete(result_path, plan)
            status = (
                "completed"
                if completed.returncode == 0 and complete
                else "failed"
            )

            with data_lock:
                statuses[run_name] = status
                runtimes[run_name] = runtime
                assigned_gpus[run_name] = gpu
                if status == "failed":
                    failures.append(
                        f"{run_name}: exit={completed.returncode}, "
                        f"log={log_path}"
                    )

            with print_lock:
                print(
                    f"[GPU {gpu}] {status.upper()} {run_name} "
                    f"({runtime / 60.0:.1f} min)",
                    flush=True,
                )
            task_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(gpu_worker, gpu)
            for gpu in gpus
        ]
        for future in futures:
            future.result()

    plan_rows = []
    raw_rows: List[Dict[str, Any]] = []
    for plan in plans:
        run_name = plan["run_name"]
        result_path = results_dir / f"{run_name}_results.txt"
        plan_rows.append(
            {
                **plan,
                "status": statuses.get(run_name, "unknown"),
                "gpu": assigned_gpus.get(run_name),
                "runtime_seconds": runtimes.get(run_name),
                "result_path": str(result_path),
                "log_path": str(log_dir / f"{run_name}.log"),
            }
        )
        if result_is_complete(result_path, plan):
            raw_rows.append(
                parse_completed_run(
                    result_path,
                    plan,
                    runtimes.get(run_name),
                    assigned_gpus.get(run_name),
                )
            )
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)

    if not raw_rows:
        raise SystemExit("No completed runs were found. Check the log files.")

    raw = pd.DataFrame(raw_rows)
    raw_path = results_dir / f"{args.name_prefix}_all_runs.csv"
    raw.to_csv(raw_path, index=False)

    # This diagnostic file shows the coefficient pair independently selected
    # inside each seed. It is not the final multi-seed hyperparameter ranking.
    seedwise_summary = build_summary(raw)
    seedwise_summary_path = (
        results_dir
        / f"{args.name_prefix}_per_seed_selected_summary.csv"
    )
    seedwise_summary.to_csv(seedwise_summary_path, index=False)

    coefficient_frames: List[pd.DataFrame] = []
    coefficient_grid_errors: List[str] = []
    for plan in plans:
        result_path = results_dir / f"{plan['run_name']}_results.txt"
        if not result_is_complete(result_path, plan):
            continue
        try:
            coefficient_frames.append(
                load_coefficient_grid(
                    results_dir,
                    plan,
                    coefficient_values,
                )
            )
        except (OSError, TypeError, ValueError, pd.errors.ParserError) as exc:
            coefficient_grid_errors.append(
                f"{plan['run_name']}: {exc}"
            )

    if coefficient_grid_errors:
        failures.extend(coefficient_grid_errors)
    if not coefficient_frames:
        raise SystemExit(
            "No complete coefficient-search grids were found."
        )

    coefficient_rows = pd.concat(
        coefficient_frames,
        ignore_index=True,
    )
    coefficient_rows_path = (
        results_dir
        / f"{args.name_prefix}_coefficient_runs.csv"
    )
    coefficient_rows.to_csv(coefficient_rows_path, index=False)

    coefficient_summary = build_shared_coefficient_summary(
        coefficient_rows,
        args.seeds,
    )
    coefficient_summary_path = (
        results_dir
        / f"{args.name_prefix}_all_hyperparameter_combinations.csv"
    )
    coefficient_summary.to_csv(coefficient_summary_path, index=False)

    # One row per expensive CNN-training configuration, using the coefficient
    # pair whose validation metrics are best on average across all seeds.
    summary = select_best_pair_per_training_config(coefficient_summary)
    summary_path = results_dir / f"{args.name_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Multi-seed validation-ranked configurations ===")
    display_columns = [
        "validation_rank",
        "config_id",
        "r2l_synth_count",
        "cb_beta",
        "focal_gamma",
        "r2l_score_coefficient",
        "u2r_score_coefficient",
        "completed_seeds",
        "minimum_minority_recall_mean",
        "minimum_minority_recall_std",
        "rare_f1_mean",
        "macro_f1_mean",
    ]
    print(summary[display_columns].head(10).to_string(index=False))
    print(f"\nCompleted runs: {len(raw_rows)}/{len(plans)}")
    print(f"Plan/status: {plan_path}")
    print(f"All completed runs: {raw_path}")
    print(f"Per-seed selected diagnostic: {seedwise_summary_path}")
    print(f"All coefficient results by seed: {coefficient_rows_path}")
    print(
        "Every full hyperparameter combination averaged over seeds: "
        f"{coefficient_summary_path}"
    )
    print(f"Final validation-ranked summary: {summary_path}")
    print("KDDTest+ metrics were not used to create this ranking.")

    if failures:
        print(f"\nFailed runs: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
