"""
Train cnn_opt and cost-sensitive XGBoost, then compare both on KDDTest+.

Run from the repository root:

    CUDA_VISIBLE_DEVICES=0 python -u src/compare_cnn_opt_xgboost.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict


RESULT_KEYS = {
    "Test Accuracy (sklearn)": "accuracy",
    "MCC": "mcc",
    "Test Macro F1": "macro_f1",
    "Test Macro Recall": "macro_recall",
    "R2L Recall": "r2l_recall",
    "U2R Recall": "u2r_recall",
}


def read_metrics(path: Path) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        output_name = RESULT_KEYS.get(key.strip())
        if output_name is not None:
            values[output_name] = float(raw_value.strip())

    missing = [name for name in RESULT_KEYS.values() if name not in values]
    if missing:
        raise ValueError(f"Missing metrics in {path}: {missing}")
    return values


def run_command(command: list[str], repo_root: Path, env: Dict[str, str]) -> None:
    print("\nRunning:")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Compare cnn_opt with cost-sensitive XGBoost on KDDTest+."
    )
    parser.add_argument("--run-name", default="cnn_vs_xgboost")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpu",
        default="0",
        help="Physical GPU made visible to both models, for example 0.",
    )
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--cnn-batch-size", type=int, default=256)
    parser.add_argument(
        "--synth-path",
        default="data/synth_ctgan_5class.csv",
        help="Cached CTGAN CSV used by cnn_opt.",
    )
    parser.add_argument(
        "--cnn-no-thresholds",
        action="store_true",
        help="Compare raw CNN argmax predictions instead of its default thresholds.",
    )
    parser.add_argument(
        "--xgb-device",
        choices=["cpu", "cuda"],
        default="cuda",
    )
    parser.add_argument("--xgb-n-estimators", type=int, default=1000)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=50)
    args = parser.parse_args()

    prefix = args.run_name.strip() or "cnn_vs_xgboost"
    cnn_run = f"{prefix}_cnn_s{args.seed}"
    xgb_run = f"{prefix}_xgb_s{args.seed}"
    results_dir = repo_root / "results"
    cnn_results = results_dir / f"{cnn_run}_results.txt"
    xgb_results = results_dir / f"{xgb_run}_results.txt"

    synth_path = Path(args.synth_path).expanduser()
    if not synth_path.is_absolute():
        synth_path = repo_root / synth_path
    if not synth_path.exists():
        raise SystemExit(
            f"cnn_opt needs its cached synthetic CSV, but this file was not found:\n"
            f"  {synth_path}\n"
            "Create it first with cnn_opt --ctgan-train --ctgan-only, or pass "
            "--synth-path to the existing file."
        )
    if importlib.util.find_spec("xgboost") is None:
        raise SystemExit(
            "XGBoost is not installed. Install it with:\n"
            "  python -m pip install xgboost==2.1.3"
        )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cnn_command = [
        sys.executable,
        "-u",
        str(repo_root / "src" / "cnn_opt.py"),
        "--run-name",
        cnn_run,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.cnn_epochs),
        "--batch-size",
        str(args.cnn_batch_size),
        "--synth-path",
        str(synth_path),
    ]
    if args.cnn_no_thresholds:
        cnn_command.append("--no-thresholds")

    xgb_command = [
        sys.executable,
        "-u",
        str(repo_root / "src" / "cost_sensitive_xgboost.py"),
        "--run-name",
        xgb_run,
        "--seed",
        str(args.seed),
        "--device",
        args.xgb_device,
        "--n-estimators",
        str(args.xgb_n_estimators),
        "--early-stopping-rounds",
        str(args.xgb_early_stopping_rounds),
    ]

    print(
        "CNN: complete cnn_opt pipeline "
        "(CTGAN + focal loss + minority batches + "
        f"{'argmax' if args.cnn_no_thresholds else 'fixed thresholds'})."
    )
    print("XGBoost: real KDDTrain+ with balanced per-class sample weights.")
    print("Both are evaluated on the same untouched KDDTest+.")

    # XGBoost is faster, so run it first and catch GPU/package problems before
    # starting the more expensive CNN training.
    run_command(xgb_command, repo_root, env)
    run_command(cnn_command, repo_root, env)

    cnn_metrics = read_metrics(cnn_results)
    xgb_metrics = read_metrics(xgb_results)
    fieldnames = [
        "model",
        "seed",
        "train_data",
        "decision_policy",
        "test_data",
        "accuracy",
        "mcc",
        "macro_f1",
        "macro_recall",
        "r2l_recall",
        "u2r_recall",
    ]
    rows = [
        {
            "model": "cnn_opt",
            "seed": args.seed,
            "train_data": "KDDTrain+ + cached R2L CTGAN",
            "decision_policy": (
                "argmax"
                if args.cnn_no_thresholds
                else "R2L=0.55/U2R=0.40 rejection thresholds"
            ),
            "test_data": "KDDTest+",
            **cnn_metrics,
        },
        {
            "model": "cost_sensitive_xgboost",
            "seed": args.seed,
            "train_data": "real KDDTrain+",
            "decision_policy": "argmax",
            "test_data": "KDDTest+",
            **xgb_metrics,
        },
    ]

    output_path = results_dir / f"{prefix}_comparison.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== CNN vs cost-sensitive XGBoost: KDDTest+ ===")
    print(
        f"{'Model':<25} {'Accuracy':>10} {'MCC':>10} {'Macro-F1':>10} "
        f"{'Macro Rec':>10} {'R2L Rec':>10} {'U2R Rec':>10}"
    )
    for row in rows:
        print(
            f"{row['model']:<25} "
            f"{row['accuracy']:>10.4f} "
            f"{row['mcc']:>10.4f} "
            f"{row['macro_f1']:>10.4f} "
            f"{row['macro_recall']:>10.4f} "
            f"{row['r2l_recall']:>10.4f} "
            f"{row['u2r_recall']:>10.4f}"
        )
    print(f"\nSaved comparison: {output_path}")


if __name__ == "__main__":
    main()
