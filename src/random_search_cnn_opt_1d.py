"""
Random hyperparameter search for cnn_opt_1d_4gpu.py.

Each training run uses all requested GPUs. Configurations run one after another
so they do not compete for the same GPUs.

Example:
    python -u src/random_search_cnn_opt_1d.py \
      --trials 12 \
      --seeds 0 1 2 \
      --auto-summarize
"""

from __future__ import annotations

import argparse
import csv
import itertools
import random
import subprocess
import sys
from pathlib import Path
from typing import List


def parse_float_list(raw: str) -> List[float]:
    return [float(value.strip()) for value in raw.split(",") if value.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Random search for the four-GPU cnn_opt Conv1D model."
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--search-seed", type=int, default=123)
    parser.add_argument("--name-prefix", type=str, default="random1d")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--synth-path", type=str, default="data/synth_ctgan_5class.csv"
    )

    # Values that may be randomly selected.
    parser.add_argument(
        "--focal-gammas", type=str, default="1.0,1.5,2.0,2.5"
    )
    parser.add_argument(
        "--cb-betas", type=str, default="0.99,0.999,0.9995,0.9999"
    )
    parser.add_argument("--groups", type=str, default="1,2,4,8")
    parser.add_argument("--base-filters", type=str, default="32,64,128")
    parser.add_argument("--dense-units", type=str, default="128,256,512")
    parser.add_argument("--dropout1-values", type=str, default="0.10,0.25,0.40")
    parser.add_argument("--dropout2-values", type=str, default="0.10,0.30,0.50")
    parser.add_argument(
        "--minority-per-batch-values", type=str, default="1,2,4"
    )

    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--auto-summarize", action="store_true")
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be greater than 0.")
    if not args.seeds:
        parser.error("Provide at least one training seed.")

    focal_gammas = parse_float_list(args.focal_gammas)
    cb_betas = parse_float_list(args.cb_betas)
    groups = parse_int_list(args.groups)
    base_filters = parse_int_list(args.base_filters)
    dense_units = parse_int_list(args.dense_units)
    dropout1_values = parse_float_list(args.dropout1_values)
    dropout2_values = parse_float_list(args.dropout2_values)
    minority_values = parse_int_list(args.minority_per_batch_values)

    search_space = [
        configuration
        for configuration in itertools.product(
            focal_gammas,
            cb_betas,
            groups,
            base_filters,
            dense_units,
            dropout1_values,
            dropout2_values,
            minority_values,
        )
        if configuration[3] % configuration[2] == 0
    ]

    if args.trials > len(search_space):
        parser.error(
            f"Requested {args.trials} trials, but the search space has "
            f"only {len(search_space)} valid configurations."
        )

    rng = random.Random(args.search_seed)
    configurations = rng.sample(search_space, args.trials)
    training_script = repo_root / "src" / "cnn_opt_1d_4gpu.py"
    result_dir = repo_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    commands = []

    for trial_number, configuration in enumerate(configurations, start=1):
        gamma, beta, group, filters, dense, dropout1, dropout2, minority = configuration

        for seed in args.seeds:
            run_name = (
                f"{args.name_prefix}_t{trial_number:03d}"
                f"_fg{slug(gamma)}_b{slug(beta)}"
                f"_g{group}_f{filters}_d{dense}"
                f"_do{slug(dropout1)}-{slug(dropout2)}"
                f"_m{minority}_s{seed}"
            )

            command = [
                sys.executable,
                "-u",
                str(training_script),
                "--run-name",
                run_name,
                "--num-gpus",
                str(args.num_gpus),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--seed",
                str(seed),
                "--synth-path",
                args.synth_path,
                "--focal-gamma",
                str(gamma),
                "--cb-beta",
                str(beta),
                "--groups",
                str(group),
                "--base-filters",
                str(filters),
                "--dense-units",
                str(dense),
                "--dropout1",
                str(dropout1),
                "--dropout2",
                str(dropout2),
                "--minority-per-batch",
                str(minority),
                "--no-thresholds",
            ]
            commands.append((run_name, command))
            plan_rows.append(
                {
                    "trial": trial_number,
                    "run_name": run_name,
                    "seed": seed,
                    "focal_gamma": gamma,
                    "cb_beta": beta,
                    "groups": group,
                    "base_filters": filters,
                    "dense_units": dense,
                    "dropout1": dropout1,
                    "dropout2": dropout2,
                    "minority_per_batch": minority,
                }
            )

    plan_path = result_dir / f"{args.name_prefix}_search_plan.csv"
    with plan_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(plan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plan_rows)

    print(f"Random configurations: {len(configurations)}")
    print(f"Seeds per configuration: {len(args.seeds)}")
    print(f"Total training runs: {len(commands)}")
    print(f"Search plan: {plan_path}")

    failures = 0
    for run_number, (run_name, command) in enumerate(commands, start=1):
        print(
            f"\n=== Run {run_number}/{len(commands)}: {run_name} ===",
            flush=True,
        )
        print(" ".join(command), flush=True)

        if args.dry_run:
            continue

        result = subprocess.run(command, cwd=repo_root, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"FAILED: {run_name} (exit={result.returncode})")
            if not args.continue_on_error:
                raise SystemExit(result.returncode)

    print(f"\nSearch finished with {failures} failed run(s).")

    if args.auto_summarize and not args.dry_run:
        summarize_command = [
            sys.executable,
            "-u",
            str(repo_root / "src" / "summarize_trials.py"),
            "--glob",
            f"{args.name_prefix}_*_results.txt",
            "--group-by",
            "focal_gamma",
            "cb_beta",
            "groups",
            "base_filters",
            "dense_units",
            "dropout1",
            "dropout2",
            "minority_per_batch",
            "--metric",
            "val_macro_f1",
            "--out-raw",
            str(result_dir / f"{args.name_prefix}_raw.csv"),
            "--out-summary",
            str(result_dir / f"{args.name_prefix}_summary.csv"),
        ]
        subprocess.run(summarize_command, cwd=repo_root, check=True)
    else:
        print(
            "Run summarize_trials.py afterward, or use --auto-summarize.",
            flush=True,
        )


if __name__ == "__main__":
    main()
