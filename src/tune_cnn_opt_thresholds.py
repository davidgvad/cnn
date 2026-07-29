"""
Tune R2L/U2R decision thresholds for an already-trained cnn_opt Conv2D model.

Thresholds are selected only on the reconstructed real-record subset of the
model's validation split. The objective is the mean R2L/U2R recall. KDDTest+
is predicted only after the pair is fixed.

Decision rule:
    adjusted_score[R2L] = probability[R2L] / r2l_threshold
    adjusted_score[U2R] = probability[U2R] / u2r_threshold

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
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    recall_score,
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
from cnn_opt import build_optimized_feature_order  # type: ignore


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
    """
    Reproduce cnn_opt's augmented split, then retain only real validation rows.

    The source marker follows the exact same shuffle and stratified split as
    cnn_opt but is removed before preprocessing/model input.
    """
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
    synth_df = pd.read_csv(synth_path)
    real_counts = np.bincount(
        train_df["class"].to_numpy(dtype=np.int64),
        minlength=5,
    )
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

    source_column = "__real_source__"
    if source_column in train_df.columns or source_column in synth_df.columns:
        raise ValueError(f"Unexpected reserved column: {source_column}")
    real_tagged = train_df.copy()
    synth_tagged = synth_df.copy()
    real_tagged[source_column] = True
    synth_tagged[source_column] = False
    augmented_tagged = (
        pd.concat([real_tagged, synth_tagged], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    source_is_real = augmented_tagged.pop(source_column).to_numpy(dtype=bool)

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
            augmented_tagged,
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
    (
        _,
        X_val,
        _,
        y_val,
        _,
        val_is_real,
    ) = train_test_split(
        X_all,
        y_all,
        source_is_real,
        test_size=val_split,
        random_state=seed,
        stratify=y_all,
    )
    real_mask = np.asarray(val_is_real, dtype=bool)
    X_val_real = X_val[real_mask]
    y_val_real = y_val[real_mask]
    if len(y_val_real) == 0:
        raise ValueError("The reconstructed validation split has no real rows.")
    return X_val_real, y_val_real, X_test, y_test


def threshold_predictions(
    probabilities: np.ndarray,
    r2l_threshold: float,
    u2r_threshold: float,
) -> np.ndarray:
    if r2l_threshold <= 0.0 or u2r_threshold <= 0.0:
        raise ValueError("Decision thresholds must be greater than zero.")
    scores = np.asarray(probabilities, dtype=np.float64).copy()
    if scores.ndim != 2 or scores.shape[1] != 5:
        raise ValueError(f"Expected probability shape (n, 5), got {scores.shape}.")
    scores[:, 2] /= float(r2l_threshold)
    scores[:, 3] /= float(u2r_threshold)
    return np.argmax(scores, axis=1).astype(np.int64)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_predictions: np.ndarray,
) -> Dict[str, float]:
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true,
        y_pred,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    changed = int(np.count_nonzero(y_pred != raw_predictions))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(np.mean(per_class_recall)),
        "r2l_recall": float(per_class_recall[2]),
        "u2r_recall": float(per_class_recall[3]),
        "minority_recall": float(np.mean(per_class_recall[[2, 3]])),
        "minimum_minority_recall": float(
            np.min(per_class_recall[[2, 3]])
        ),
        "r2l_f1": float(per_class_f1[2]),
        "u2r_f1": float(per_class_f1[3]),
        "rare_f1": float(np.mean(per_class_f1[[2, 3]])),
        "changed_predictions": float(changed),
        "change_rate": float(changed / len(y_true)),
    }


def threshold_values(
    minimum: float,
    maximum: float,
    step: float,
) -> List[float]:
    count = int(np.floor((maximum - minimum) / step + 1e-9))
    values = [minimum + index * step for index in range(count + 1)]
    if not np.isclose(values[-1], maximum):
        values.append(maximum)
    values.append(1.0)
    return sorted(
        {
            round(float(value), 10)
            for value in values
            if minimum <= value <= maximum
        }
    )


def search_thresholds(
    y_val: np.ndarray,
    val_probabilities: np.ndarray,
    candidates: List[float],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    raw_predictions = np.argmax(val_probabilities, axis=1)
    rows: List[Dict[str, Any]] = []

    for r2l_threshold in candidates:
        for u2r_threshold in candidates:
            predictions = threshold_predictions(
                val_probabilities,
                r2l_threshold,
                u2r_threshold,
            )
            metrics = calculate_metrics(
                y_val,
                predictions,
                raw_predictions,
            )
            distance_from_argmax = (
                abs(np.log(r2l_threshold))
                + abs(np.log(u2r_threshold))
            )
            row: Dict[str, Any] = {
                "r2l_threshold": r2l_threshold,
                "u2r_threshold": u2r_threshold,
                "distance_from_argmax": float(distance_from_argmax),
                **metrics,
            }
            rows.append(row)

    if not rows:
        raise ValueError("Threshold search produced no candidates.")
    search_frame = pd.DataFrame(rows).sort_values(
        by=[
            "minority_recall",
            "minimum_minority_recall",
            "rare_f1",
            "macro_f1",
            "mcc",
            "accuracy",
            "change_rate",
            "distance_from_argmax",
            "r2l_threshold",
            "u2r_threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)
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
            "Tune cnn_opt thresholds for highest mean R2L/U2R "
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
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=2.00)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024)
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
    if args.threshold_min <= 0.0:
        parser.error("--threshold-min must be greater than zero.")
    if args.threshold_max < args.threshold_min:
        parser.error("--threshold-max must be at least --threshold-min.")
    if not args.threshold_min <= 1.0 <= args.threshold_max:
        parser.error(
            "The threshold range must include 1.0 (the raw-argmax baseline)."
        )
    if args.threshold_step <= 0.0:
        parser.error("--threshold-step must be greater than zero.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

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
    print(
        "Real validation class counts:",
        np.bincount(y_val, minlength=5),
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
        f"Searching {len(candidates) ** 2} threshold pairs "
        "using mean validation R2L/U2R recall..."
    )
    best, search_frame = search_thresholds(
        y_val,
        val_probabilities,
        candidates,
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

    # KDDTest+ predictions are made only after the threshold pair is selected.
    print("Thresholds selected. Predicting KDDTest+ once...")
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
            "WARNING: selected threshold reached the search boundary "
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
            "r2l_threshold": 1.0,
            "u2r_threshold": 1.0,
            **raw_test_metrics,
        },
        {
            "model": "cnn_opt_validation_tuned",
            "seed": seed,
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
        output_file.write("CNN_OPT validation-only threshold tuning\n\n")
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
            "selection_objective: mean validation R2L/U2R recall\n"
        )
        output_file.write(
            "threshold_rule: divide R2L/U2R scores by their thresholds\n"
        )
        output_file.write(f"threshold_min: {args.threshold_min}\n")
        output_file.write(f"threshold_max: {args.threshold_max}\n")
        output_file.write(f"threshold_step: {args.threshold_step}\n")
        output_file.write(f"threshold_pairs_searched: {len(search_frame)}\n")
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

    print("\n=== Selected decision thresholds ===")
    print(f"R2L threshold: {r2l_threshold:.4f}")
    print(f"U2R threshold: {u2r_threshold:.4f}")
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
