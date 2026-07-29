"""
Compare Conv1D and Conv2D with shared hyperparameters and paired seeds.

Example:
    python -u src/compare_cnn_1d_2d.py \
      --configs 10 \
      --seeds 0 1 2 3 4 \
      --gpus 0 1 2 3 \
      --epochs 25 \
      --batch-size 256

The script randomly chooses shared configurations. For each configuration and
seed, it runs both models on the same GPU. Existing completed runs are skipped,
so rerunning the same command resumes an interrupted comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import random
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from summarize_trials import parse_results_file


CONFIG_FIELDS = [
    "focal_gamma",
    "cb_beta",
    "groups",
    "base_filters",
    "dense_units",
    "dropout1",
    "dropout2",
    "minority_per_batch",
]

SEARCH_VALUES = {
    "focal_gamma": [1.0, 1.5, 2.0, 2.5],
    "cb_beta": [0.99, 0.999, 0.9995, 0.9999],
    "groups": [1, 2, 4, 8],
    "base_filters": [32, 64, 128],
    "dense_units": [128, 256, 512],
    "dropout1": [0.10, 0.25, 0.40],
    "dropout2": [0.10, 0.30, 0.50],
    "minority_per_batch": [1, 2, 4],
}

METRICS = [
    "val_macro_f1",
    "test_macro_f1",
    "R2L_precision",
    "R2L_recall",
    "R2L_f1",
    "U2R_precision",
    "U2R_recall",
    "U2R_f1",
    "mcc",
    "accuracy",
    "model_parameters",
]


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
    """Identify the exact model code and data used by this experiment."""
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


def result_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        macro_f1 = parse_results_file(path).data.get("macro_f1")
        return isinstance(macro_f1, (int, float))
    except Exception:
        return False


def build_command(
    repo_root: Path,
    model_kind: str,
    run_name: str,
    plan: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    script_name = "cnn_opt_1d_4gpu.py" if model_kind == "1d" else "cnn_opt.py"
    command = [
        sys.executable,
        "-u",
        str(repo_root / "src" / script_name),
        "--run-name",
        run_name,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--seed",
        str(plan["seed"]),
        "--val-split",
        str(args.val_split),
        "--synth-path",
        args.synth_path,
        "--feature-layout",
        "optimized",
    ]
    argument_names = {
        "focal_gamma": "--focal-gamma",
        "cb_beta": "--cb-beta",
        "groups": "--groups",
        "base_filters": "--base-filters",
        "dense_units": "--dense-units",
        "dropout1": "--dropout1",
        "dropout2": "--dropout2",
        "minority_per_batch": "--minority-per-batch",
    }
    for field in CONFIG_FIELDS:
        command.extend([argument_names[field], str(plan[field])])

    # Each child process sees exactly one GPU.
    if model_kind == "1d":
        command.extend(["--num-gpus", "1"])
    # Keep the architecture comparison on raw argmax predictions.
    command.append("--no-thresholds")
    return command


def parse_run(
    result_path: Path,
    model_kind: str,
    plan: Dict[str, Any],
    runtime: float | None,
) -> Dict[str, Any]:
    parsed = parse_results_file(result_path).data
    return {
        "config_id": plan["config_id"],
        "config_key": plan["config_key"],
        "experiment_key": plan["experiment_key"],
        "model": model_kind,
        "seed": plan["seed"],
        **{field: plan[field] for field in CONFIG_FIELDS},
        "val_macro_f1": parsed.get("val_macro_f1"),
        "test_macro_f1": parsed.get("macro_f1"),
        "R2L_precision": parsed.get("R2L_precision"),
        "R2L_recall": parsed.get("R2L_recall"),
        "R2L_f1": parsed.get("R2L_f1"),
        "U2R_precision": parsed.get("U2R_precision"),
        "U2R_recall": parsed.get("U2R_recall"),
        "U2R_f1": parsed.get("U2R_f1"),
        "mcc": parsed.get("mcc"),
        "accuracy": parsed.get("accuracy"),
        "model_parameters": parsed.get("Model Parameters"),
        "runtime_seconds": runtime,
        "result_path": str(result_path),
    }


def mean_and_std(values: pd.Series) -> tuple[float | None, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None, None
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0
    return mean, std


def build_summary(raw: pd.DataFrame, configs: List[Dict[str, Any]]) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    config_lookup = {config["config_id"]: config for config in configs}

    for config_id, config in config_lookup.items():
        config_runs = raw[raw["config_id"] == config_id]
        summary: Dict[str, Any] = {
            "config_id": config_id,
            "config_key": config["config_key"],
            "experiment_key": config["experiment_key"],
            **{field: config[field] for field in CONFIG_FIELDS},
        }

        for model_kind in ("1d", "2d"):
            model_runs = config_runs[config_runs["model"] == model_kind]
            summary[f"runs_{model_kind}"] = len(model_runs)
            for metric in METRICS:
                mean, std = mean_and_std(model_runs[metric])
                summary[f"{metric}_{model_kind}_mean"] = mean
                summary[f"{metric}_{model_kind}_std"] = std

        paired = config_runs.pivot(
            index="seed",
            columns="model",
            values="test_macro_f1",
        ).reindex(columns=["1d", "2d"]).dropna()
        differences = paired["2d"] - paired["1d"]
        summary["paired_seeds"] = len(paired)
        summary["test_macro_f1_delta_2d_minus_1d_mean"] = (
            float(differences.mean()) if len(differences) else None
        )
        summary["test_macro_f1_delta_2d_minus_1d_std"] = (
            float(differences.std(ddof=1))
            if len(differences) > 1
            else (0.0 if len(differences) == 1 else None)
        )
        summary["wins_1d"] = int((differences < 0).sum())
        summary["wins_2d"] = int((differences > 0).sum())
        summary["ties"] = int((differences == 0).sum())
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    log_dir = results_dir / "conv_comparison_logs"

    parser = argparse.ArgumentParser(
        description="Paired multi-config, multi-seed Conv1D versus Conv2D comparison."
    )
    parser.add_argument("--configs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--search-seed", type=int, default=123)
    parser.add_argument("--name-prefix", type=str, default="conv_compare")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument(
        "--synth-path",
        type=str,
        default="data/synth_ctgan_5class.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.configs <= 0:
        parser.error("--configs must be greater than 0.")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique seed values.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be greater than 0.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1.")
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))

    synth_path = Path(args.synth_path)
    if not synth_path.is_absolute():
        synth_path = repo_root / synth_path
    if not args.dry_run and not synth_path.exists():
        raise SystemExit(
            f"Cached CTGAN file not found: {synth_path}\n"
            "Generate it once before starting the comparison."
        )

    experiment_key = fingerprint_files(
        [
            repo_root / "src" / "cnn_opt.py",
            repo_root / "src" / "cnn_opt_1d_4gpu.py",
            repo_root / "src" / "cnn_gan_foc.py",
            repo_root / "data" / "KDDTrain+.txt",
            repo_root / "data" / "KDDTest+.txt",
            synth_path,
        ]
    )

    combinations = [
        combination
        for combination in itertools.product(
            *(SEARCH_VALUES[field] for field in CONFIG_FIELDS)
        )
        if combination[3] % combination[2] == 0
    ]
    if args.configs > len(combinations):
        parser.error(f"--configs cannot exceed {len(combinations)}.")
    sampled = random.Random(args.search_seed).sample(combinations, args.configs)

    configs: List[Dict[str, Any]] = []
    for number, combination in enumerate(sampled, start=1):
        values = dict(zip(CONFIG_FIELDS, combination))
        identity = (
            combination,
            args.epochs,
            args.batch_size,
            args.val_split,
            args.synth_path,
            experiment_key,
        )
        configs.append(
            {
                "config_id": f"c{number:03d}",
                "config_key": hashlib.sha256(
                    repr(identity).encode("utf-8")
                ).hexdigest()[:8],
                "experiment_key": experiment_key,
                **values,
            }
        )

    plans: List[Dict[str, Any]] = []
    for config in configs:
        for seed in args.seeds:
            base_name = (
                f"{args.name_prefix}_{config['config_id']}_"
                f"{config['config_key']}_s{seed}"
            )
            plans.append(
                {
                    **config,
                    "seed": seed,
                    "run_name_1d": f"{base_name}_1d",
                    "run_name_2d": f"{base_name}_2d",
                }
            )

    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plan_path = results_dir / f"{args.name_prefix}_plan.csv"
    pd.DataFrame(plans).to_csv(plan_path, index=False)

    print(f"Configurations: {len(configs)}")
    print(f"Seeds per configuration: {len(args.seeds)}")
    print(f"Total runs: {len(plans) * 2}")
    print(f"GPUs: {gpus}")
    print(f"Plan: {plan_path}")

    if args.dry_run:
        for plan in plans:
            for model_kind in ("1d", "2d"):
                run_name = plan[f"run_name_{model_kind}"]
                command = build_command(
                    repo_root,
                    model_kind,
                    run_name,
                    plan,
                    args,
                )
                print(f"[{model_kind}] {shlex.join(command)}")
        print("Dry run complete; nothing was trained.")
        return

    assignments = {gpu: [] for gpu in gpus}
    for index, plan in enumerate(plans):
        assignments[gpus[index % len(gpus)]].append(plan)

    print_lock = threading.Lock()
    data_lock = threading.Lock()
    failures: List[str] = []
    runtimes: Dict[str, float] = {}

    def gpu_worker(gpu: str, gpu_plans: List[Dict[str, Any]]) -> None:
        for plan in gpu_plans:
            # Alternate order so neither architecture always runs first.
            config_number = int(plan["config_id"][1:])
            order = (
                ("1d", "2d")
                if (config_number + int(plan["seed"])) % 2 == 0
                else ("2d", "1d")
            )
            for model_kind in order:
                run_name = plan[f"run_name_{model_kind}"]
                result_path = results_dir / f"{run_name}_results.txt"
                if result_is_complete(result_path):
                    with print_lock:
                        print(f"[GPU {gpu}] SKIP {run_name}", flush=True)
                    continue

                command = build_command(
                    repo_root,
                    model_kind,
                    run_name,
                    plan,
                    args,
                )
                log_path = log_dir / f"{run_name}.log"
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                environment["PYTHONHASHSEED"] = str(plan["seed"])

                with print_lock:
                    print(f"[GPU {gpu}] START {run_name}", flush=True)

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
                    runtimes[run_name] = runtime
                    if completed.returncode != 0 or not complete:
                        failures.append(
                            f"{run_name}: exit={completed.returncode}, log={log_path}"
                        )

                with print_lock:
                    status = "DONE" if completed.returncode == 0 and complete else "FAILED"
                    print(
                        f"[GPU {gpu}] {status} {run_name} "
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

    raw_rows: List[Dict[str, Any]] = []
    for plan in plans:
        for model_kind in ("1d", "2d"):
            run_name = plan[f"run_name_{model_kind}"]
            result_path = results_dir / f"{run_name}_results.txt"
            if result_is_complete(result_path):
                raw_rows.append(
                    parse_run(
                        result_path,
                        model_kind,
                        plan,
                        runtimes.get(run_name),
                    )
                )

    if not raw_rows:
        raise SystemExit("No completed runs were found. Check the log files.")

    raw = pd.DataFrame(raw_rows)
    raw_path = results_dir / f"{args.name_prefix}_all_runs.csv"
    raw.to_csv(raw_path, index=False)

    summary = build_summary(raw, configs)
    summary_path = results_dir / f"{args.name_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    paired = raw.pivot(
        index=["config_id", "seed"],
        columns="model",
        values="test_macro_f1",
    ).reindex(columns=["1d", "2d"]).dropna()
    differences = paired["2d"] - paired["1d"]
    overall = pd.DataFrame(
        [
            {
                "completed_pairs": len(paired),
                "test_macro_f1_1d_mean": float(paired["1d"].mean()),
                "test_macro_f1_2d_mean": float(paired["2d"].mean()),
                "delta_2d_minus_1d_mean": float(differences.mean()),
                "delta_2d_minus_1d_std": (
                    float(differences.std(ddof=1))
                    if len(differences) > 1
                    else 0.0
                ),
                "wins_1d": int((differences < 0).sum()),
                "wins_2d": int((differences > 0).sum()),
                "ties": int((differences == 0).sum()),
            }
        ]
    )
    overall_path = results_dir / f"{args.name_prefix}_overall.csv"
    overall.to_csv(overall_path, index=False)

    print("\n=== Comparison complete ===")
    print(f"Completed pairs: {len(paired)}/{len(plans)}")
    print(
        "Mean Test Macro-F1: "
        f"1D={overall.iloc[0]['test_macro_f1_1d_mean']:.4f}, "
        f"2D={overall.iloc[0]['test_macro_f1_2d_mean']:.4f}"
    )
    print(
        "Mean difference (2D - 1D): "
        f"{overall.iloc[0]['delta_2d_minus_1d_mean']:.4f}"
    )
    print(
        f"Wins: 1D={overall.iloc[0]['wins_1d']}, "
        f"2D={overall.iloc[0]['wins_2d']}, "
        f"ties={overall.iloc[0]['ties']}"
    )
    print(f"All runs: {raw_path}")
    print(f"Per-configuration summary: {summary_path}")
    print(f"Overall summary: {overall_path}")

    if failures:
        print(f"\nFailed runs: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
