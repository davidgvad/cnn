"""
Evaluate one validation-selected cnn2d_grid winner on KDDTest+.

The script does not train or tune anything. It reads one shared R2L/U2R
coefficient pair from the multi-seed validation summary, loads every saved
seed model for that configuration, and compares raw argmax with the same
frozen score-scaling rule.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

from cnn_opt import apply_class_score_scaling, score_scaling_metrics
from tune_cnn_opt_thresholds import (
    load_validation_and_test,
    read_metadata,
    read_saved_counts,
)


POLICY_ORDER = [
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
    "minimum_minority_recall",
    "minority_recall",
    "rare_f1",
    "r2l_f1",
    "u2r_f1",
]


def resolve_path(repo_root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def mean_and_std(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="raise")
    return (
        float(numeric.mean()),
        float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
    )


def build_summary(
    rows: pd.DataFrame,
    expected_seeds: List[int],
) -> pd.DataFrame:
    expected_seed_set = set(expected_seeds)
    output: List[Dict[str, Any]] = []
    for policy in POLICY_ORDER:
        group = rows[rows["policy"] == policy]
        observed_seeds = set(group["seed"].astype(int))
        if (
            observed_seeds != expected_seed_set
            or len(group) != len(expected_seed_set)
        ):
            raise ValueError(
                f"{policy} does not contain exactly the expected seeds "
                f"{sorted(expected_seed_set)}."
            )
        row: Dict[str, Any] = {
            "policy": policy,
            "runs": len(group),
            "seeds": ",".join(str(seed) for seed in sorted(observed_seeds)),
        }
        for metric in METRICS:
            mean, std = mean_and_std(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        output.append(row)
    return pd.DataFrame(output)


def format_percent(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}% ± {100.0 * std:.2f}%"


def formatted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    labels = {
        "accuracy": "Accuracy",
        "mcc": "MCC",
        "macro_f1": "Macro-F1",
        "macro_recall": "Macro Recall",
        "r2l_recall": "R2L Recall",
        "u2r_recall": "U2R Recall",
    }
    output: List[Dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        formatted: Dict[str, Any] = {
            "Policy": row["policy"],
            "Runs": int(row["runs"]),
        }
        for metric, label in labels.items():
            formatted[label] = format_percent(
                float(row[f"{metric}_mean"]),
                float(row[f"{metric}_std"]),
            )
        output.append(formatted)
    return pd.DataFrame(output)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a validation-ranked cnn2d_grid configuration using "
            "one frozen coefficient pair across all saved seed models."
        )
    )
    parser.add_argument(
        "--summary",
        default="results/cnn2d_grid_summary.csv",
    )
    parser.add_argument(
        "--plan",
        default="results/cnn2d_grid_plan.csv",
    )
    parser.add_argument(
        "--validation-rank",
        type=int,
        default=2,
        help=(
            "Validation rank to evaluate. Rank 2 is the selected active-focal "
            "configuration (gamma=0.5)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    if args.validation_rank <= 0:
        parser.error("--validation-rank must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    summary_path = resolve_path(repo_root, args.summary)
    plan_path = resolve_path(repo_root, args.plan)
    validation_summary = pd.read_csv(
        summary_path,
        dtype={"config_id": str, "config_key": str},
    )
    plan = pd.read_csv(
        plan_path,
        dtype={"config_id": str, "config_key": str},
    )

    required_summary_columns = {
        "validation_rank",
        "config_id",
        "config_key",
        "r2l_score_coefficient",
        "u2r_score_coefficient",
        "completed_seeds",
    }
    missing_summary = sorted(
        required_summary_columns - set(validation_summary.columns)
    )
    if missing_summary:
        raise ValueError(
            f"{summary_path} is missing columns: {missing_summary}"
        )
    winner_rows = validation_summary[
        validation_summary["validation_rank"] == args.validation_rank
    ]
    if len(winner_rows) != 1:
        raise ValueError(
            f"Expected one row with validation_rank={args.validation_rank}, "
            f"found {len(winner_rows)}."
        )
    winner = winner_rows.iloc[0]
    config_id = str(winner["config_id"])
    config_key = str(winner["config_key"])
    r2l_coefficient = float(winner["r2l_score_coefficient"])
    u2r_coefficient = float(winner["u2r_score_coefficient"])
    for name, coefficient in [
        ("R2L", r2l_coefficient),
        ("U2R", u2r_coefficient),
    ]:
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(
                f"{name} coefficient must be finite and greater than zero."
            )

    required_plan_columns = {
        "config_id",
        "config_key",
        "seed",
        "run_name",
    }
    missing_plan = sorted(required_plan_columns - set(plan.columns))
    if missing_plan:
        raise ValueError(f"{plan_path} is missing columns: {missing_plan}")
    selected_plans = plan[
        (plan["config_id"].astype(str) == config_id)
        & (plan["config_key"].astype(str) == config_key)
    ].copy()
    selected_plans["seed"] = selected_plans["seed"].astype(int)
    selected_plans = selected_plans.sort_values("seed")
    expected_count = int(winner["completed_seeds"])
    if len(selected_plans) != expected_count:
        raise ValueError(
            f"Winner expects {expected_count} seeds, but {len(selected_plans)} "
            "matching plan rows were found."
        )
    seeds = selected_plans["seed"].tolist()
    if len(seeds) != len(set(seeds)):
        raise ValueError("The selected plan contains duplicate seeds.")

    print(
        f"Validation rank {args.validation_rank}: "
        f"{config_id}/{config_key}"
    )
    print(
        "Frozen score coefficients: "
        f"R2L={r2l_coefficient}, U2R={u2r_coefficient}"
    )
    print("Seeds:", seeds)
    print("No training or coefficient search will be performed.")

    result_rows: List[Dict[str, Any]] = []
    for plan_row in selected_plans.to_dict(orient="records"):
        run_name = str(plan_row["run_name"])
        seed = int(plan_row["seed"])
        model_path = repo_root / "model" / f"{run_name}_best.keras"
        results_path = repo_root / "results" / f"{run_name}_results.txt"
        if not model_path.exists():
            raise FileNotFoundError(f"Saved model not found: {model_path}")
        if not results_path.exists():
            raise FileNotFoundError(
                f"Saved run metadata not found: {results_path}"
            )

        metadata = read_metadata(results_path)
        if int(float(metadata.get("seed", -1))) != seed:
            raise ValueError(f"Seed metadata mismatch for {run_name}.")
        feature_layout = metadata.get("feature_layout")
        if feature_layout not in {"optimized", "legacy"}:
            raise ValueError(
                f"Invalid saved feature layout for {run_name}: "
                f"{feature_layout}"
            )
        val_split = float(metadata["val_split"])
        synth_path = resolve_path(repo_root, metadata["synth_path"])

        _, _, X_test, y_test = load_validation_and_test(
            seed=seed,
            val_split=val_split,
            synth_path=synth_path,
            feature_layout=feature_layout,
            expected_real_counts=read_saved_counts(
                metadata,
                "real_counts",
            ),
            expected_synth_counts=read_saved_counts(
                metadata,
                "synth_counts",
            ),
            expected_aug_counts=read_saved_counts(
                metadata,
                "aug_counts",
            ),
        )
        print(f"Seed {seed}: loading {model_path.name}")
        model = tf.keras.models.load_model(model_path, compile=False)
        if tuple(model.input_shape[1:]) != (11, 11, 1):
            raise ValueError(
                f"Unexpected model input shape for {run_name}: "
                f"{model.input_shape}"
            )
        probabilities = model.predict(
            X_test,
            batch_size=args.batch_size,
            verbose=0,
        )
        raw_predictions = np.argmax(probabilities, axis=1)
        tuned_predictions = apply_class_score_scaling(
            probabilities,
            {
                2: r2l_coefficient,
                3: u2r_coefficient,
            },
        )
        raw_metrics = score_scaling_metrics(
            y_test,
            raw_predictions,
            raw_predictions,
        )
        tuned_metrics = score_scaling_metrics(
            y_test,
            tuned_predictions,
            raw_predictions,
        )
        for policy, metrics in [
            ("Integrated+Batch-CNN", raw_metrics),
            ("Integrated+Batch+Tuning", tuned_metrics),
        ]:
            result_rows.append(
                {
                    "policy": policy,
                    "seed": seed,
                    "source_run_name": run_name,
                    "r2l_score_coefficient": (
                        1.0
                        if policy == "Integrated+Batch-CNN"
                        else r2l_coefficient
                    ),
                    "u2r_score_coefficient": (
                        1.0
                        if policy == "Integrated+Batch-CNN"
                        else u2r_coefficient
                    ),
                    **metrics,
                }
            )
        tf.keras.backend.clear_session()

    results_dir = repo_root / "results"
    output_prefix = (
        args.output_prefix.strip()
        if args.output_prefix
        else f"cnn2d_grid_rank{args.validation_rank:03d}_fixed_test"
    )
    all_runs_path = results_dir / f"{output_prefix}_all_runs.csv"
    numeric_summary_path = results_dir / f"{output_prefix}_summary.csv"
    formatted_path = results_dir / f"{output_prefix}_summary_formatted.csv"
    text_path = results_dir / f"{output_prefix}_summary.txt"

    all_runs = pd.DataFrame(result_rows)
    all_runs.to_csv(all_runs_path, index=False)
    numeric_summary = build_summary(all_runs, seeds)
    numeric_summary.to_csv(numeric_summary_path, index=False)
    pretty = formatted_summary(numeric_summary)
    pretty.to_csv(formatted_path, index=False)
    text_path.write_text(
        pretty.to_string(index=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Frozen multi-seed KDDTest+ evaluation ===")
    print(pretty.to_string(index=False))
    print(f"\nPer-seed results: {all_runs_path}")
    print(f"Numeric mean/std: {numeric_summary_path}")
    print(f"Formatted table: {formatted_path}")
    print(f"Readable table: {text_path}")


if __name__ == "__main__":
    main()
