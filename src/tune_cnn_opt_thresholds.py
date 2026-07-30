"""
Tune R2L/U2R score coefficients for an already-trained cnn_opt Conv2D model.

Coefficients are selected only on the reconstructed real-record subset of the
model's validation split. The objective first maximizes the weaker of the R2L
and U2R recalls, then their mean and rare-class F1. KDDTest+ is predicted only
after the pair is fixed.

Decision rule:
    adjusted_score[R2L] = probability[R2L] / r2l_coefficient
    adjusted_score[U2R] = probability[U2R] / u2r_coefficient

The other three class scores are unchanged. Therefore (1.0, 1.0) is ordinary
argmax, values below 1 promote a rare class, and values above 1 suppress it.
These are operating-point parameters, not calibrated probability cutoffs.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
)
from sklearn.model_selection import train_test_split

from cnn_gan_foc import (  # type: ignore
    CLASS_NAMES,
    _repo_paths,
    apply_one_hot,
    apply_scaler,
    collapse_attack_labels,
    fit_one_hot_encoders,
    fit_scaler,
    load_nsl_kdd_txt,
    prepare_xy_from_processed,
    save_confusion_matrices,
)
from cnn_opt import (  # type: ignore
    apply_class_score_scaling,
    build_optimized_feature_order,
    score_coefficient_values,
    score_scaling_metrics,
    search_score_coefficients,
    select_synth_for_targets,
)


CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]

COLUMNS_TO_SCALE = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "count",
    "srv_count",
    "dst_host_count",
    "dst_host_srv_count",
]

def read_metadata(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    metadata: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def read_saved_counts(metadata: Dict[str, str], key: str) -> np.ndarray | None:
    value = metadata.get(key)
    if value is None:
        return None
    try:
        counts = np.asarray(ast.literal_eval(value), dtype=np.int64)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {key} from the results file.") from exc
    if counts.shape != (5,):
        raise ValueError(f"Expected five values for {key}, got {counts.tolist()}.")
    return counts


def verify_counts(
    name: str,
    actual: np.ndarray,
    expected: np.ndarray | None,
) -> None:
    if expected is not None and not np.array_equal(actual, expected):
        raise ValueError(
            f"{name} class counts differ from the saved training run. "
            f"Expected {expected.tolist()}, got {actual.tolist()}. "
            "Use the same data and synthetic CSV that trained the model."
        )


def resolve_model_and_metadata(
    repo_root: Path,
    run_name: str | None,
    model_path_arg: str | None,
) -> Tuple[str, Path, Path, Dict[str, str]]:
    if run_name:
        resolved_run_name = run_name.strip()
        model_path = repo_root / "model" / f"{resolved_run_name}_best.keras"
    else:
        model_path = Path(str(model_path_arg)).expanduser()
        if not model_path.is_absolute():
            model_path = repo_root / model_path
        model_path = model_path.resolve()
        stem = model_path.stem
        resolved_run_name = stem[:-5] if stem.endswith("_best") else stem

    results_path = (
        repo_root / "results" / f"{resolved_run_name}_results.txt"
    )
    return (
        resolved_run_name,
        model_path.resolve(),
        results_path,
        read_metadata(results_path),
    )


def load_validation_and_test(
    seed: int,
    val_split: float,
    synth_path: Path,
    feature_layout: str,
    expected_real_counts: np.ndarray | None,
    expected_synth_counts: np.ndarray | None,
    expected_aug_counts: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce cnn_opt's real-only validation split and KDDTest+ input."""
    paths = _repo_paths()
    train_df = load_nsl_kdd_txt(paths.data_dir / "KDDTrain+.txt").drop(
        columns=["num_outbound_cmds"]
    )
    test_df = load_nsl_kdd_txt(paths.data_dir / "KDDTest+.txt").drop(
        columns=["num_outbound_cmds"]
    )
    train_df = collapse_attack_labels(train_df, is_train=True)
    test_df = collapse_attack_labels(test_df, is_train=False)

    if not synth_path.exists():
        raise FileNotFoundError(f"Cached CTGAN CSV not found: {synth_path}")
    synth_pool = pd.read_csv(synth_path)
    real_counts = np.bincount(
        train_df["class"].to_numpy(dtype=np.int64),
        minlength=5,
    )
    if expected_synth_counts is not None:
        targets = {
            class_id: int(
                real_counts[class_id] + expected_synth_counts[class_id]
            )
            for class_id in range(5)
        }
        synth_df, _, _ = select_synth_for_targets(
            synth_pool,
            train_df["class"].to_numpy(dtype=np.int64),
            targets,
        )
    else:
        # Compatibility for older result files that did not save selected
        # synthetic counts.
        synth_df = synth_pool
    synth_counts = np.bincount(
        synth_df["class"].to_numpy(dtype=np.int64),
        minlength=5,
    )
    verify_counts("Real training", real_counts, expected_real_counts)
    verify_counts("Synthetic", synth_counts, expected_synth_counts)
    verify_counts(
        "Augmented training",
        real_counts + synth_counts,
        expected_aug_counts,
    )

    encoders, feature_names = fit_one_hot_encoders(
        train_df,
        CATEGORICAL_COLUMNS,
    )
    train_real_ohe = apply_one_hot(
        train_df,
        encoders,
        feature_names,
        CATEGORICAL_COLUMNS,
    )
    scaler = fit_scaler(train_real_ohe, COLUMNS_TO_SCALE)

    train_proc = apply_scaler(
        apply_one_hot(
            train_df,
            encoders,
            feature_names,
            CATEGORICAL_COLUMNS,
        ),
        scaler,
        COLUMNS_TO_SCALE,
    )
    test_proc = apply_scaler(
        apply_one_hot(
            test_df,
            encoders,
            feature_names,
            CATEGORICAL_COLUMNS,
        ),
        scaler,
        COLUMNS_TO_SCALE,
    )
    test_proc = test_proc[train_proc.columns]

    feature_columns = [c for c in train_proc.columns if c != "class"]
    if feature_layout == "optimized":
        ordered_features = build_optimized_feature_order(feature_columns)
    else:
        ordered_features = feature_columns
    ordered_columns = ordered_features + ["class"]
    train_proc = train_proc[ordered_columns]
    test_proc = test_proc[ordered_columns]

    X_all, y_all, X_test, y_test = prepare_xy_from_processed(
        train_proc,
        test_proc,
    )
    _, X_val, _, y_val = train_test_split(
        X_all,
        y_all,
        test_size=val_split,
        random_state=seed,
        stratify=y_all,
    )
    if len(y_val) == 0:
        raise ValueError("The reconstructed validation split has no real rows.")
    return X_val, y_val, X_test, y_test


def threshold_predictions(
    probabilities: np.ndarray,
    r2l_threshold: float,
    u2r_threshold: float,
) -> np.ndarray:
    """Compatibility name for class-specific score scaling."""
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError(
            f"Expected probability shape (n, 5), got {probabilities.shape}."
        )
    return apply_class_score_scaling(
        probabilities,
        {
            2: float(r2l_threshold),
            3: float(u2r_threshold),
        },
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_predictions: np.ndarray,
) -> Dict[str, float]:
    return score_scaling_metrics(y_true, y_pred, raw_predictions)


def threshold_values(
    minimum: float,
    maximum: float,
    step: float,
) -> List[float]:
    return score_coefficient_values(minimum, maximum, step)


def search_thresholds(
    y_val: np.ndarray,
    val_probabilities: np.ndarray,
    candidates: List[float],
    macro_f1_retention: float = 0.90,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    _, rows = search_score_coefficients(
        y_val,
        val_probabilities,
        candidates,
        macro_f1_retention=macro_f1_retention,
    )
    search_frame = pd.DataFrame(rows)
    # Legacy columns kept for existing analysis commands.
    search_frame["r2l_threshold"] = search_frame[
        "r2l_score_coefficient"
    ]
    search_frame["u2r_threshold"] = search_frame[
        "u2r_score_coefficient"
    ]
    search_frame.insert(0, "rank", np.arange(1, len(search_frame) + 1))
    best_row = search_frame.iloc[0].to_dict()
    return best_row, search_frame


def format_metric_lines(
    label: str,
    metrics: Dict[str, float],
) -> List[str]:
    return [
        f"{label} Accuracy: {metrics['accuracy']}",
        f"{label} MCC: {metrics['mcc']}",
        f"{label} Macro F1: {metrics['macro_f1']}",
        f"{label} Macro Recall: {metrics['macro_recall']}",
        f"{label} R2L Recall: {metrics['r2l_recall']}",
        f"{label} U2R Recall: {metrics['u2r_recall']}",
        f"{label} Minority Recall Mean: {metrics['minority_recall']}",
        f"{label} R2L F1: {metrics['r2l_f1']}",
        f"{label} U2R F1: {metrics['u2r_f1']}",
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Tune cnn_opt score coefficients for balanced R2L/U2R "
            "validation recall."
        )
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--run-name",
        help="Existing cnn_opt run prefix from results/ and model/.",
    )
    model_group.add_argument(
        "--model-path",
        help="Path to an existing cnn_opt .keras model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Defaults to the seed saved in the run's results file.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Defaults to the value saved in the run's results file.",
    )
    parser.add_argument(
        "--synth-path",
        default=None,
        help="Defaults to the path saved in the run's results file.",
    )
    parser.add_argument(
        "--feature-layout",
        choices=["optimized", "legacy"],
        default=None,
        help="Defaults to the layout saved in the run's results file.",
    )
    parser.add_argument(
        "--coefficient-min",
        "--threshold-min",
        dest="threshold_min",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--coefficient-max",
        "--threshold-max",
        dest="threshold_max",
        type=float,
        default=2.00,
    )
    parser.add_argument(
        "--coefficient-step",
        "--threshold-step",
        dest="threshold_step",
        type=float,
        default=0.15,
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--min-validation-macro-f1-retention",
        type=float,
        default=0.90,
    )
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    run_name, model_path, source_results_path, metadata = (
        resolve_model_and_metadata(
            repo_root,
            args.run_name,
            args.model_path,
        )
    )
    if not model_path.exists():
        raise SystemExit(f"Saved model not found: {model_path}")

    if args.seed is not None:
        seed = args.seed
    elif "seed" in metadata:
        seed = int(float(metadata["seed"]))
    else:
        parser.error("--seed is required when it cannot be read from results.")

    if args.val_split is not None:
        val_split = args.val_split
    elif "val_split" in metadata:
        val_split = float(metadata["val_split"])
    else:
        parser.error(
            "--val-split is required when it cannot be read from results."
        )

    if args.feature_layout is not None:
        feature_layout = args.feature_layout
    elif "feature_layout" in metadata:
        feature_layout = metadata["feature_layout"]
    else:
        parser.error(
            "--feature-layout is required when it cannot be read from results."
        )

    if args.synth_path is not None:
        synth_value = args.synth_path
    elif "synth_path" in metadata:
        synth_value = metadata["synth_path"]
    else:
        parser.error(
            "--synth-path is required when it cannot be read from results."
        )
    synth_path = Path(synth_value).expanduser()
    if not synth_path.is_absolute():
        synth_path = repo_root / synth_path
    synth_path = synth_path.resolve()

    if not 0.0 < val_split < 1.0:
        parser.error("--val-split must be between 0 and 1.")
    if feature_layout not in {"optimized", "legacy"}:
        parser.error(
            "Saved feature_layout must be either 'optimized' or 'legacy'."
        )
    if not np.isfinite(args.threshold_min) or args.threshold_min <= 0.0:
        parser.error("--coefficient-min must be finite and greater than zero.")
    if (
        not np.isfinite(args.threshold_max)
        or args.threshold_max < args.threshold_min
    ):
        parser.error(
            "--coefficient-max must be finite and at least --coefficient-min."
        )
    if not args.threshold_min <= 1.0 <= args.threshold_max:
        parser.error(
            "The coefficient range must include 1.0 "
            "(the raw-argmax baseline)."
        )
    if not np.isfinite(args.threshold_step) or args.threshold_step <= 0.0:
        parser.error(
            "--coefficient-step must be finite and greater than zero."
        )
    estimated_candidates = (
        (args.threshold_max - args.threshold_min)
        / args.threshold_step
        + 2.0
    )
    if (
        not np.isfinite(estimated_candidates)
        or estimated_candidates > np.sqrt(100_000)
    ):
        parser.error(
            "The coefficient grid is too large. Increase "
            "--coefficient-step or narrow the search range."
        )
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")
    if not (
        np.isfinite(args.min_validation_macro_f1_retention)
        and 0.0 <= args.min_validation_macro_f1_retention <= 1.0
    ):
        parser.error(
            "--min-validation-macro-f1-retention must be between 0 and 1."
        )

    print("Loading model:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (11, 11, 1):
        raise SystemExit(
            "This tuner expects the Conv2D input shape (11, 11, 1), "
            f"but the saved model uses {model.input_shape}."
        )
    X_val, y_val, X_test, y_test = load_validation_and_test(
        seed=seed,
        val_split=val_split,
        synth_path=synth_path,
        feature_layout=feature_layout,
        expected_real_counts=read_saved_counts(metadata, "real_counts"),
        expected_synth_counts=read_saved_counts(metadata, "synth_counts"),
        expected_aug_counts=read_saved_counts(metadata, "aug_counts"),
    )
    print(
        "Note: the original run did not save validation row IDs; this split is "
        "reconstructed from its seed and current data files."
    )
    validation_counts = np.bincount(y_val, minlength=5)
    print("Real validation class counts:", validation_counts)
    if validation_counts[2] == 0 or validation_counts[3] == 0:
        raise SystemExit(
            "The real validation subset must contain both R2L and U2R "
            "records to tune their score coefficients."
        )
    print("Predicting real validation rows once...")
    val_probabilities = model.predict(
        X_val,
        batch_size=args.batch_size,
        verbose=0,
    )

    candidates = threshold_values(
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )
    print(
        f"Searching {len(candidates) ** 2} score-coefficient pairs "
        "using balanced validation R2L/U2R recall..."
    )
    best, search_frame = search_thresholds(
        y_val,
        val_probabilities,
        candidates,
        macro_f1_retention=args.min_validation_macro_f1_retention,
    )
    r2l_threshold = float(best["r2l_threshold"])
    u2r_threshold = float(best["u2r_threshold"])

    raw_val_predictions = np.argmax(val_probabilities, axis=1)
    tuned_val_predictions = threshold_predictions(
        val_probabilities,
        r2l_threshold,
        u2r_threshold,
    )
    raw_val_metrics = calculate_metrics(
        y_val,
        raw_val_predictions,
        raw_val_predictions,
    )
    tuned_val_metrics = calculate_metrics(
        y_val,
        tuned_val_predictions,
        raw_val_predictions,
    )

    # KDDTest+ is predicted only after the coefficient pair is selected.
    print("Score coefficients selected. Predicting KDDTest+ once...")
    test_probabilities = model.predict(
        X_test,
        batch_size=args.batch_size,
        verbose=0,
    )
    raw_test_predictions = np.argmax(test_probabilities, axis=1)
    tuned_test_predictions = threshold_predictions(
        test_probabilities,
        r2l_threshold,
        u2r_threshold,
    )
    raw_test_metrics = calculate_metrics(
        y_test,
        raw_test_predictions,
        raw_test_predictions,
    )
    tuned_test_metrics = calculate_metrics(
        y_test,
        tuned_test_predictions,
        raw_test_predictions,
    )

    boundary_hits = []
    if np.isclose(r2l_threshold, args.threshold_min):
        boundary_hits.append("R2L minimum")
    if np.isclose(r2l_threshold, args.threshold_max):
        boundary_hits.append("R2L maximum")
    if np.isclose(u2r_threshold, args.threshold_min):
        boundary_hits.append("U2R minimum")
    if np.isclose(u2r_threshold, args.threshold_max):
        boundary_hits.append("U2R maximum")
    if boundary_hits:
        print(
            "WARNING: a selected coefficient reached the search boundary "
            f"({', '.join(boundary_hits)})."
        )

    output_prefix = (
        args.output_prefix.strip()
        if args.output_prefix
        else f"{run_name}_threshold_tuned"
    )
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    search_path = results_dir / f"{output_prefix}_search.csv"
    comparison_path = results_dir / f"{output_prefix}_comparison.csv"
    results_path = results_dir / f"{output_prefix}_results.txt"
    search_frame.to_csv(search_path, index=False)

    comparison_rows = [
        {
            "model": "cnn_opt_raw_argmax",
            "seed": seed,
            "r2l_score_coefficient": 1.0,
            "u2r_score_coefficient": 1.0,
            "r2l_threshold": 1.0,
            "u2r_threshold": 1.0,
            **raw_test_metrics,
        },
        {
            "model": "cnn_opt_validation_tuned",
            "seed": seed,
            "r2l_score_coefficient": r2l_threshold,
            "u2r_score_coefficient": u2r_threshold,
            "r2l_threshold": r2l_threshold,
            "u2r_threshold": u2r_threshold,
            **tuned_test_metrics,
        },
    ]
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

    raw_report = classification_report(
        y_test,
        raw_test_predictions,
        labels=np.arange(5),
        target_names=CLASS_NAMES,
        digits=8,
        zero_division=0,
    )
    tuned_report = classification_report(
        y_test,
        tuned_test_predictions,
        labels=np.arange(5),
        target_names=CLASS_NAMES,
        digits=8,
        zero_division=0,
    )
    save_confusion_matrices(
        y_test,
        raw_test_predictions,
        results_dir,
        f"{output_prefix}_raw_argmax",
    )
    save_confusion_matrices(
        y_test,
        tuned_test_predictions,
        results_dir,
        output_prefix,
    )

    with results_path.open("w", encoding="utf-8") as output_file:
        output_file.write(
            "CNN_OPT validation-only class-specific score scaling\n\n"
        )
        output_file.write(f"source_run_name: {run_name}\n")
        output_file.write(f"source_model: {model_path}\n")
        output_file.write(f"source_results: {source_results_path}\n")
        output_file.write(f"seed: {seed}\n")
        output_file.write(f"val_split: {val_split}\n")
        output_file.write(f"feature_layout: {feature_layout}\n")
        output_file.write(f"synth_path: {synth_path}\n")
        output_file.write(
            "selection_data: reconstructed real subset of original validation\n"
        )
        output_file.write(
            "reconstruction_note: original validation row IDs were not saved; "
            "the split was recreated from the saved seed and current data\n"
        )
        output_file.write(
            "selection_objective: retain validation macro-F1, then maximize "
            "the lower R2L/U2R recall, their mean, and rare-class F1\n"
        )
        output_file.write(
            "min_validation_macro_f1_retention: "
            f"{args.min_validation_macro_f1_retention}\n"
        )
        output_file.write(
            "score_scaling_rule: divide R2L/U2R scores by their "
            "coefficients before argmax\n"
        )
        output_file.write(f"coefficient_min: {args.threshold_min}\n")
        output_file.write(f"coefficient_max: {args.threshold_max}\n")
        output_file.write(f"coefficient_step: {args.threshold_step}\n")
        output_file.write(
            f"coefficient_pairs_searched: {len(search_frame)}\n"
        )
        # Legacy keys kept so older result parsers continue to work.
        output_file.write(f"threshold_min: {args.threshold_min}\n")
        output_file.write(f"threshold_max: {args.threshold_max}\n")
        output_file.write(f"threshold_step: {args.threshold_step}\n")
        output_file.write(f"threshold_pairs_searched: {len(search_frame)}\n")
        output_file.write(
            f"r2l_score_coefficient: {r2l_threshold}\n"
        )
        output_file.write(
            f"u2r_score_coefficient: {u2r_threshold}\n"
        )
        output_file.write(f"r2l_threshold: {r2l_threshold}\n")
        output_file.write(f"u2r_threshold: {u2r_threshold}\n\n")
        for line in format_metric_lines("Raw Validation", raw_val_metrics):
            output_file.write(line + "\n")
        output_file.write("\n")
        for line in format_metric_lines("Tuned Validation", tuned_val_metrics):
            output_file.write(line + "\n")
        output_file.write("\n")
        for line in format_metric_lines("Raw Test", raw_test_metrics):
            output_file.write(line + "\n")
        output_file.write("\n")
        for line in format_metric_lines("Tuned Test", tuned_test_metrics):
            output_file.write(line + "\n")
        output_file.write("\n")
        output_file.write(
            f"Test Accuracy (sklearn): {tuned_test_metrics['accuracy']}\n"
        )
        output_file.write(f"MCC: {tuned_test_metrics['mcc']}\n")
        output_file.write(
            f"Test Macro F1: {tuned_test_metrics['macro_f1']}\n"
        )
        output_file.write(
            f"Test Macro Recall: {tuned_test_metrics['macro_recall']}\n"
        )
        output_file.write(
            f"R2L Recall: {tuned_test_metrics['r2l_recall']}\n"
        )
        output_file.write(
            f"U2R Recall: {tuned_test_metrics['u2r_recall']}\n\n"
        )
        output_file.write("Raw argmax test report:\n")
        output_file.write(raw_report)
        output_file.write("\n\nTuned test report:\n")
        output_file.write(tuned_report)
        output_file.write("\n")

    print("\n=== Selected score coefficients ===")
    print(f"R2L coefficient: {r2l_threshold:.4f}")
    print(f"U2R coefficient: {u2r_threshold:.4f}")
    print(
        "Validation weaker-minority recall: "
        f"raw={raw_val_metrics['minimum_minority_recall']:.4f}, "
        f"tuned={tuned_val_metrics['minimum_minority_recall']:.4f}"
    )
    print(
        "Validation minority recall mean: "
        f"raw={raw_val_metrics['minority_recall']:.4f}, "
        f"tuned={tuned_val_metrics['minority_recall']:.4f}"
    )
    print("\n=== KDDTest+ ===")
    print(
        f"{'Policy':<22} {'Accuracy':>9} {'MCC':>9} {'Macro-F1':>10} "
        f"{'Macro Rec':>10} {'R2L Rec':>9} {'U2R Rec':>9}"
    )
    for label, metrics in [
        ("Raw argmax", raw_test_metrics),
        ("Validation-tuned", tuned_test_metrics),
    ]:
        print(
            f"{label:<22} "
            f"{metrics['accuracy']:>9.4f} "
            f"{metrics['mcc']:>9.4f} "
            f"{metrics['macro_f1']:>10.4f} "
            f"{metrics['macro_recall']:>10.4f} "
            f"{metrics['r2l_recall']:>9.4f} "
            f"{metrics['u2r_recall']:>9.4f}"
        )
    print(f"\nSearch grid: {search_path}")
    print(f"Test comparison: {comparison_path}")
    print(f"Detailed results: {results_path}")


if __name__ == "__main__":
    main()
