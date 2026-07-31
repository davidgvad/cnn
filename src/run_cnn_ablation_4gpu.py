"""
Run the six-row CNN ablation table with paired seeds on four GPUs.

The comparison uses one shared cnn_opt preprocessing/backbone so each row
changes only the named training technique. Five models are trained per seed;
the sixth row (Integrated+Batch+Tuning) reuses the Integrated+Batch model and
changes only its fixed post-training decision rule.

Typical run from the repository root:

    python -u src/run_cnn_ablation_4gpu.py \
      --gpus 0 1 2 3 \
      --seeds 0 1 2 \
      --prepare-synth
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
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

from grid_search_cnn_opt_4gpu import (
    count_pool_r2l,
    count_real_r2l,
    fingerprint_files,
    parse_gpus,
    prepare_synth_pool,
)
from summarize_trials import parse_results_file


TABLE_ORDER = [
    "Baseline CNN",
    "CBF-CNN",
    "Naive CTGAN-CNN",
    "Integrated-CNN",
    "Integrated+Batch-CNN",
    "Integrated+Batch+Tuning",
]

METRICS = [
    "accuracy",
    "mcc",
    "macro_f1",
    "macro_recall",
    "r2l_recall",
    "u2r_recall",
]

STANDARD_RESULT_FIELDS = {
    "accuracy": "Test Accuracy (sklearn)",
    "mcc": "MCC",
    "macro_f1": "Test Macro F1",
    "macro_recall": "Test Macro Recall",
    "r2l_recall": "R2L Recall",
    "u2r_recall": "U2R Recall",
}

RAW_RESULT_FIELDS = {
    "accuracy": "Raw Argmax Test Accuracy",
    "mcc": "Raw Argmax MCC",
    "macro_f1": "Raw Argmax Test Macro F1",
    "macro_recall": "Raw Argmax Test Macro Recall",
    "r2l_recall": "Raw Argmax R2L Recall",
    "u2r_recall": "Raw Argmax U2R Recall",
}


def required_float(data: Dict[str, Any], key: str) -> float:
    value = data.get(key)
    if value is None:
        raise ValueError(f"Missing result field: {key}")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"Non-finite result field: {key}={value}")
    return number


def training_configurations(
    target_r2l: int,
    r2l_synth_count: int,
) -> List[Dict[str, Any]]:
    """Five trainings produce the six requested table rows."""
    return [
        {
            "training_id": "baseline",
            "display_name": "Baseline CNN",
            "loss_mode": "cross_entropy",
            "target_r2l": 0,
            "r2l_synthetic_rows": 0,
            "minority_per_batch": 0,
            "score_scaling": False,
        },
        {
            "training_id": "cbf",
            "display_name": "CBF-CNN",
            "loss_mode": "class_balanced_focal",
            "target_r2l": 0,
            "r2l_synthetic_rows": 0,
            "minority_per_batch": 0,
            "score_scaling": False,
        },
        {
            "training_id": "naive_ctgan",
            "display_name": "Naive CTGAN-CNN",
            "loss_mode": "cross_entropy",
            "target_r2l": target_r2l,
            "r2l_synthetic_rows": r2l_synth_count,
            "minority_per_batch": 0,
            "score_scaling": False,
        },
        {
            "training_id": "integrated",
            "display_name": "Integrated-CNN",
            "loss_mode": "class_balanced_focal",
            "target_r2l": target_r2l,
            "r2l_synthetic_rows": r2l_synth_count,
            "minority_per_batch": 0,
            "score_scaling": False,
        },
        {
            "training_id": "integrated_batch",
            "display_name": "Integrated+Batch-CNN",
            "loss_mode": "class_balanced_focal",
            "target_r2l": target_r2l,
            "r2l_synthetic_rows": r2l_synth_count,
            "minority_per_batch": 1,
            "score_scaling": True,
        },
    ]


def build_command(
    repo_root: Path,
    plan: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    command = [
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
        "--loss-mode",
        plan["loss_mode"],
        "--synth-path",
        str(args.synth_pool),
        "--target-r2l",
        str(plan["target_r2l"]),
        "--target-u2r",
        "0",
        "--focal-gamma",
        str(args.focal_gamma),
        "--cb-beta",
        str(args.cb_beta),
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
        str(plan["minority_per_batch"]),
        "--feature-layout",
        "optimized",
    ]
    if plan["score_scaling"]:
        command.extend(
            [
                "--r2l-score-coefficient",
                str(args.r2l_score_coefficient),
                "--u2r-score-coefficient",
                str(args.u2r_score_coefficient),
            ]
        )
    else:
        command.append("--no-score-scaling")
    return command


def result_is_complete(
    result_path: Path,
    plan: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not result_path.exists():
        return False
    try:
        data = parse_results_file(result_path).data
        if data.get("run_name") != plan["run_name"]:
            return False
        if data.get("loss_mode") != plan["loss_mode"]:
            return False

        numeric_checks = {
            "seed": plan["seed"],
            "target_r2l": plan["target_r2l"],
            "target_u2r": 0,
            "r2l_synthetic_rows_used": plan["r2l_synthetic_rows"],
            "minority_per_batch": plan["minority_per_batch"],
            "focal_gamma": args.focal_gamma,
            "cb_beta": args.cb_beta,
            "groups": args.groups,
            "base_filters": args.base_filters,
            "dense_units": args.dense_units,
            "dropout1": args.dropout1,
            "dropout2": args.dropout2,
        }
        for key, expected in numeric_checks.items():
            if not np.isclose(required_float(data, key), float(expected)):
                return False

        expected_selection = (
            "fixed_cli_coefficients"
            if plan["score_scaling"]
            else "disabled"
        )
        if data.get("score_scaling_selection") != expected_selection:
            return False
        if plan["score_scaling"]:
            if not np.isclose(
                required_float(data, "r2l_score_coefficient"),
                args.r2l_score_coefficient,
            ):
                return False
            if not np.isclose(
                required_float(data, "u2r_score_coefficient"),
                args.u2r_score_coefficient,
            ):
                return False

        for key in set(STANDARD_RESULT_FIELDS.values()) | set(
            RAW_RESULT_FIELDS.values()
        ):
            required_float(data, key)
        return True
    except (OSError, TypeError, ValueError):
        return False


def metric_row(
    data: Dict[str, Any],
    plan: Dict[str, Any],
    model_name: str,
    fields: Dict[str, str],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "model": model_name,
        "seed": int(plan["seed"]),
        "source_training_id": plan["training_id"],
        "source_run_name": plan["run_name"],
        "loss_mode": plan["loss_mode"],
        "ctgan_r2l_rows": int(plan["r2l_synthetic_rows"]),
        "minority_per_batch": int(plan["minority_per_batch"]),
        "score_scaling": (
            model_name == "Integrated+Batch+Tuning"
        ),
    }
    for metric, result_key in fields.items():
        row[metric] = required_float(data, result_key)
    return row


def expand_result_rows(
    result_path: Path,
    plan: Dict[str, Any],
) -> List[Dict[str, Any]]:
    data = parse_results_file(result_path).data
    if plan["training_id"] != "integrated_batch":
        return [
            metric_row(
                data,
                plan,
                plan["display_name"],
                STANDARD_RESULT_FIELDS,
            )
        ]

    # These two rows intentionally share one model and one probability array.
    # Only the post-training decision policy differs.
    return [
        metric_row(
            data,
            plan,
            "Integrated+Batch-CNN",
            RAW_RESULT_FIELDS,
        ),
        metric_row(
            data,
            plan,
            "Integrated+Batch+Tuning",
            STANDARD_RESULT_FIELDS,
        ),
    ]


def mean_and_std(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="raise")
    return (
        float(numeric.mean()),
        float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
    )


def build_summary(
    raw: pd.DataFrame,
    expected_seeds: Sequence[int],
) -> pd.DataFrame:
    expected_seed_set = {int(seed) for seed in expected_seeds}
    rows: List[Dict[str, Any]] = []
    for model_name in TABLE_ORDER:
        group = raw[raw["model"] == model_name]
        observed_seeds = set(group["seed"].astype(int))
        if (
            observed_seeds != expected_seed_set
            or len(group) != len(expected_seed_set)
        ):
            missing = sorted(expected_seed_set - observed_seeds)
            raise ValueError(
                f"{model_name} is missing complete seed results: {missing}"
            )

        row: Dict[str, Any] = {
            "model": model_name,
            "runs": len(group),
            "seeds": ",".join(
                str(seed) for seed in sorted(observed_seeds)
            ),
        }
        for metric in METRICS:
            mean, std = mean_and_std(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        rows.append(row)
    return pd.DataFrame(rows)


def format_percent(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}% ± {100.0 * std:.2f}%"


def formatted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    labels = {
        "accuracy": "Accuracy",
        "mcc": "MCC",
        "macro_f1": "Macro-F1",
        "macro_recall": "Macro Recall",
        "r2l_recall": "R2L Recall",
        "u2r_recall": "U2R Recall",
    }
    for row in summary.to_dict(orient="records"):
        formatted: Dict[str, Any] = {
            "Model": row["model"],
            "Runs": int(row["runs"]),
        }
        for metric, label in labels.items():
            formatted[label] = format_percent(
                float(row[f"{metric}_mean"]),
                float(row[f"{metric}_std"]),
            )
        rows.append(formatted)
    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    parser = argparse.ArgumentParser(
        description=(
            "Run the six controlled CNN ablations using one job per GPU."
        )
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--name-prefix", default="cnn_ablation_best")
    parser.add_argument(
        "--synth-pool",
        default="data/synth_ctgan_r2l_generated5000.csv",
    )
    parser.add_argument("--r2l-synth-count", type=int, default=5000)
    parser.add_argument("--prepare-synth", action="store_true")
    parser.add_argument("--regenerate-synth", action="store_true")
    parser.add_argument("--ctgan-seed", type=int, default=0)
    parser.add_argument("--ctgan-epochs", type=int, default=200)
    parser.add_argument("--ctgan-batch-size", type=int, default=4096)
    parser.add_argument("--ctgan-pac", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--cb-beta", type=float, default=0.99)
    parser.add_argument("--focal-gamma", type=float, default=0.5)
    parser.add_argument(
        "--r2l-score-coefficient",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--u2r-score-coefficient",
        type=float,
        default=0.10,
    )
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout1", type=float, default=0.25)
    parser.add_argument("--dropout2", type=float, default=0.30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-all-commands", action="store_true")
    args = parser.parse_args()

    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if args.r2l_synth_count < 0:
        parser.error("--r2l-synth-count must be 0 or greater.")
    if args.regenerate_synth and not args.prepare_synth:
        parser.error("--regenerate-synth requires --prepare-synth.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be greater than zero.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1.")
    if not 0.0 < args.cb_beta < 1.0:
        parser.error("--cb-beta must be between 0 and 1.")
    if args.focal_gamma < 0.0:
        parser.error("--focal-gamma must be 0 or greater.")
    if (
        args.r2l_score_coefficient <= 0.0
        or args.u2r_score_coefficient <= 0.0
    ):
        parser.error("Score coefficients must be greater than zero.")
    if args.groups <= 0 or args.base_filters % args.groups != 0:
        parser.error("--groups must divide --base-filters.")
    if args.dense_units <= 0:
        parser.error("--dense-units must be greater than zero.")
    if not 0.0 <= args.dropout1 < 1.0:
        parser.error("--dropout1 must be in [0, 1).")
    if not 0.0 <= args.dropout2 < 1.0:
        parser.error("--dropout2 must be in [0, 1).")

    synth_path = Path(args.synth_pool).expanduser()
    if not synth_path.is_absolute():
        synth_path = repo_root / synth_path
    synth_path = synth_path.resolve()
    args.synth_pool = str(synth_path)

    train_path = repo_root / "data" / "KDDTrain+.txt"
    test_path = repo_root / "data" / "KDDTest+.txt"
    real_r2l_count = count_real_r2l(train_path)
    final_r2l_target = real_r2l_count + args.r2l_synth_count

    should_prepare = args.prepare_synth and (
        not synth_path.exists() or args.regenerate_synth
    )
    if should_prepare and not args.dry_run:
        synth_path.parent.mkdir(parents=True, exist_ok=True)
        prepare_synth_pool(
            repo_root,
            synth_path,
            real_r2l_count,
            args.r2l_synth_count,
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
        if available_r2l < args.r2l_synth_count:
            raise SystemExit(
                f"Synthetic pool has {available_r2l} R2L rows, but this "
                f"experiment needs {args.r2l_synth_count}.\n"
                "Use --prepare-synth --regenerate-synth to rebuild it."
            )

    experiment_fingerprint = fingerprint_files(
        [
            repo_root / "src" / "cnn_opt.py",
            repo_root / "src" / "cnn_gan_foc.py",
            Path(__file__).resolve(),
            train_path,
            test_path,
            synth_path,
        ]
    )
    configs = training_configurations(
        final_r2l_target,
        args.r2l_synth_count,
    )
    plans: List[Dict[str, Any]] = []
    for config in configs:
        identity = (
            config,
            args.epochs,
            args.batch_size,
            args.val_split,
            args.cb_beta,
            args.focal_gamma,
            args.r2l_score_coefficient,
            args.u2r_score_coefficient,
            args.groups,
            args.base_filters,
            args.dense_units,
            args.dropout1,
            args.dropout2,
            experiment_fingerprint,
        )
        config_key = hashlib.sha256(
            repr(identity).encode("utf-8")
        ).hexdigest()[:10]
        for seed in args.seeds:
            plans.append(
                {
                    **config,
                    "config_key": config_key,
                    "experiment_fingerprint": experiment_fingerprint,
                    "seed": int(seed),
                    "run_name": (
                        f"{args.name_prefix}_{config['training_id']}_"
                        f"{config_key}_s{seed}"
                    ),
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results_dir / f"{args.name_prefix}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    plan_path = results_dir / f"{args.name_prefix}_plan.csv"
    protocol_path = results_dir / f"{args.name_prefix}_protocol.txt"
    pd.DataFrame(plans).to_csv(plan_path, index=False)
    protocol_path.write_text(
        "\n".join(
            [
                "Controlled six-row CNN ablation protocol",
                "",
                "All rows use the cnn_opt optimized feature layout, "
                "backbone, real-only stratified validation split, and "
                "val-macro-F1 checkpoint rule.",
                "Baseline CNN: cross-entropy; no synthetic rows; random "
                "batches; raw argmax.",
                "CBF-CNN: class-balanced loss; no synthetic rows; random "
                "batches; raw argmax.",
                "Naive CTGAN-CNN: cross-entropy; fixed R2L-only CTGAN pool; "
                "random batches; raw argmax.",
                "Integrated-CNN: class-balanced loss + fixed R2L-only CTGAN; "
                "random batches; raw argmax.",
                "Integrated+Batch-CNN: same as Integrated plus "
                "minority-guaranteed batches; raw argmax.",
                "Integrated+Batch+Tuning: the exact same trained model as "
                "Integrated+Batch, with one frozen score-coefficient pair.",
                "",
                f"Seeds: {args.seeds}",
                f"Generated R2L rows: {args.r2l_synth_count}",
                f"Beta: {args.cb_beta}",
                f"Gamma: {args.focal_gamma}",
                (
                    "Gamma=0 means class-balanced weighted cross-entropy; "
                    "the focal focusing term is inactive."
                    if np.isclose(args.focal_gamma, 0.0)
                    else (
                        f"Gamma={args.focal_gamma} keeps the focal focusing "
                        "term active."
                    )
                ),
                "The Naive CTGAN row is a controlled current R2L-only "
                "augmentation ablation, not an exact reproduction of the "
                "historical mixed-class CTGAN experiment.",
                f"R2L score coefficient: {args.r2l_score_coefficient}",
                f"U2R score coefficient: {args.u2r_score_coefficient}",
                "KDDTest+ is used only for final reporting, never for "
                "hyperparameter or coefficient selection.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Real KDDTrain+ R2L rows: {real_r2l_count}")
    print(f"Fixed generated R2L rows: {args.r2l_synth_count}")
    print(f"Class-balanced beta/gamma: {args.cb_beta}/{args.focal_gamma}")
    print(
        "Fixed R2L/U2R score coefficients: "
        f"{args.r2l_score_coefficient}/{args.u2r_score_coefficient}"
    )
    print(
        f"Five trainings per seed; {len(args.seeds)} seeds; "
        f"{len(plans)} total trainings"
    )
    print(
        f"GPUs: {gpus}; approximately "
        f"{int(np.ceil(len(plans) / len(gpus)))} waves"
    )
    print(f"Plan: {plan_path}")
    print(f"Protocol: {protocol_path}")

    if args.dry_run:
        shown = plans if args.print_all_commands else plans[:10]
        for index, plan in enumerate(shown):
            gpu = gpus[index % len(gpus)]
            print(
                f"[GPU {gpu}] "
                f"{shlex.join(build_command(repo_root, plan, args))}"
            )
        if len(shown) < len(plans):
            print(f"... {len(plans) - len(shown)} more commands")
        print("Dry run complete; nothing was trained.")
        return

    task_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    for plan in plans:
        task_queue.put(plan)

    print_lock = threading.Lock()
    data_lock = threading.Lock()
    statuses: Dict[str, str] = {}
    assigned_gpus: Dict[str, str] = {}
    runtimes: Dict[str, float] = {}
    failures: List[str] = []

    def gpu_worker(gpu: str) -> None:
        while True:
            try:
                plan = task_queue.get_nowait()
            except queue.Empty:
                return

            run_name = plan["run_name"]
            result_path = results_dir / f"{run_name}_results.txt"
            if result_is_complete(result_path, plan, args):
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
                    f"\n\n=== attempt "
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
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
            complete = result_is_complete(result_path, plan, args)
            status = (
                "completed"
                if completed.returncode == 0 and complete
                else "failed"
            )

            with data_lock:
                statuses[run_name] = status
                assigned_gpus[run_name] = gpu
                runtimes[run_name] = runtime
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

    plan_rows: List[Dict[str, Any]] = []
    metric_rows: List[Dict[str, Any]] = []
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
        if result_is_complete(result_path, plan, args):
            metric_rows.extend(expand_result_rows(result_path, plan))
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)

    if not metric_rows:
        raise SystemExit("No complete results were found. Check the logs.")

    raw = pd.DataFrame(metric_rows)
    raw_path = results_dir / f"{args.name_prefix}_all_runs.csv"
    raw.to_csv(raw_path, index=False)

    summary = build_summary(raw, args.seeds)
    summary_path = results_dir / f"{args.name_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    pretty = formatted_summary(summary)
    formatted_path = (
        results_dir / f"{args.name_prefix}_summary_formatted.csv"
    )
    pretty.to_csv(formatted_path, index=False)
    text_path = results_dir / f"{args.name_prefix}_summary.txt"
    text_path.write_text(
        pretty.to_string(index=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Paired-seed KDDTest+ comparison ===")
    print(pretty.to_string(index=False))
    print(f"\nCompleted trainings: {len(plans) - len(failures)}/{len(plans)}")
    print(f"Plan/status: {plan_path}")
    print(f"Protocol: {protocol_path}")
    print(f"Per-seed metrics: {raw_path}")
    print(f"Numeric mean/std: {summary_path}")
    print(f"Formatted table: {formatted_path}")
    print(f"Readable table: {text_path}")

    if failures:
        print(f"\nFailed runs: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
