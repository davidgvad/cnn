"""
Standard or cost-sensitive XGBoost baseline for NSL-KDD five-class.

Cost-sensitive mode gives each training row a balanced class weight:

    weight(class c) = number_of_training_rows / (5 * rows_in_class_c)

This gives rare R2L/U2R rows more influence without changing KDDTest+.
"""

from __future__ import annotations

import argparse
import csv
from typing import Dict, Tuple

import numpy as np
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
    save_confusion_matrices,
    save_multiclass_roc,
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


def load_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the same label mapping, one-hot encoding, and scaling as cnn_opt."""
    paths = _repo_paths()
    train_df = load_nsl_kdd_txt(paths.data_dir / "KDDTrain+.txt").drop(
        columns=["num_outbound_cmds"]
    )
    test_df = load_nsl_kdd_txt(paths.data_dir / "KDDTest+.txt").drop(
        columns=["num_outbound_cmds"]
    )
    train_df = collapse_attack_labels(train_df, is_train=True)
    test_df = collapse_attack_labels(test_df, is_train=False)

    encoders, feature_names = fit_one_hot_encoders(
        train_df,
        CATEGORICAL_COLUMNS,
    )
    train_ohe = apply_one_hot(
        train_df,
        encoders,
        feature_names,
        CATEGORICAL_COLUMNS,
    )
    test_ohe = apply_one_hot(
        test_df,
        encoders,
        feature_names,
        CATEGORICAL_COLUMNS,
    )

    scaler = fit_scaler(train_ohe, COLUMNS_TO_SCALE)
    train_proc = apply_scaler(train_ohe, scaler, COLUMNS_TO_SCALE)
    test_proc = apply_scaler(test_ohe, scaler, COLUMNS_TO_SCALE)
    test_proc = test_proc[train_proc.columns]

    y_train = train_proc["class"].to_numpy(dtype=np.int64)
    y_test = test_proc["class"].to_numpy(dtype=np.int64)
    X_train = train_proc.drop(columns=["class"]).to_numpy(dtype=np.float32)
    X_test = test_proc.drop(columns=["class"]).to_numpy(dtype=np.float32)

    if X_train.shape[1] != 121 or X_test.shape[1] != 121:
        raise ValueError(
            "Expected 121 processed features. "
            f"Got train={X_train.shape[1]}, test={X_test.shape[1]}."
        )
    return X_train, y_train, X_test, y_test


def balanced_class_weights(y: np.ndarray, num_classes: int = 5) -> np.ndarray:
    """Return sklearn-style balanced weights, one number per class."""
    counts = np.bincount(y, minlength=num_classes)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Cannot compute class weights; missing classes: {missing}")
    return len(y) / (float(num_classes) * counts.astype(np.float64))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    per_class_recall = recall_score(
        y_true,
        y_pred,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(np.mean(per_class_recall)),
        "r2l_recall": float(per_class_recall[2]),
        "u2r_recall": float(per_class_recall[3]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cost-sensitive XGBoost baseline on NSL-KDD."
    )
    parser.add_argument("--run-name", default="cost_sensitive_xgboost")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument(
        "--class-weighting",
        choices=["none", "balanced"],
        default="balanced",
        help="none is standard XGBoost; balanced is cost-sensitive XGBoost.",
    )
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.80)
    parser.add_argument("--colsample-bytree", type=float, default=0.80)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Use cuda for one visible GPU, or cpu.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--verbose-eval",
        type=int,
        default=50,
        help="Print validation loss every N boosting rounds; 0 disables it.",
    )
    args = parser.parse_args()

    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between 0 and 1.")
    if args.n_estimators <= 0:
        parser.error("--n-estimators must be greater than 0.")
    if args.early_stopping_rounds < 0:
        parser.error("--early-stopping-rounds must be 0 or greater.")

    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "XGBoost is not installed. Install it with: "
            "python -m pip install xgboost==2.1.3"
        ) from exc

    np.random.seed(args.seed)
    paths = _repo_paths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, X_test, y_test = load_processed_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y_all,
    )

    if args.class_weighting == "balanced":
        class_weights = balanced_class_weights(y_train)
        train_sample_weights = class_weights[y_train]
    else:
        class_weights = None
        train_sample_weights = None

    model_args = {
        "objective": "multi:softprob",
        "num_class": 5,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "gamma": args.gamma,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": args.device,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }
    if args.early_stopping_rounds > 0:
        model_args["early_stopping_rounds"] = args.early_stopping_rounds

    model = XGBClassifier(**model_args)
    fit_args = {
        "eval_set": [(X_val, y_val)],
        "verbose": args.verbose_eval if args.verbose_eval > 0 else False,
    }
    if train_sample_weights is not None:
        fit_args["sample_weight"] = train_sample_weights
    model.fit(X_train, y_train, **fit_args)

    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test).astype(np.int64)
    metrics = calculate_metrics(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(5),
        target_names=CLASS_NAMES,
        digits=8,
        zero_division=0,
    )

    prefix = args.run_name.strip() or "cost_sensitive_xgboost"
    model_path = paths.model_dir / f"{prefix}.json"
    results_path = paths.results_dir / f"{prefix}_results.txt"
    metrics_path = paths.results_dir / f"{prefix}_metrics.csv"
    model.save_model(str(model_path))
    save_confusion_matrices(y_test, y_pred, paths.results_dir, prefix)
    save_multiclass_roc(y_test, y_proba, paths.results_dir, prefix)

    train_counts = np.bincount(y_train, minlength=5)
    val_counts = np.bincount(y_val, minlength=5)
    test_counts = np.bincount(y_test, minlength=5)
    best_iteration = getattr(model, "best_iteration", None)

    model_label = (
        "xgboost_cost_sensitive"
        if args.class_weighting == "balanced"
        else "xgboost_standard"
    )
    with results_path.open("w", encoding="utf-8") as f:
        if args.class_weighting == "balanced":
            f.write(
                "Cost-sensitive XGBoost "
                "(balanced per-class sample weights)\n\n"
            )
        else:
            f.write("Standard XGBoost (no class/sample weights)\n\n")
        f.write(f"run_name: {prefix}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"class_weighting: {args.class_weighting}\n")
        f.write("validation_weighting: none\n")
        f.write("train_data: real KDDTrain+ only\n")
        f.write("test_data: KDDTest+\n")
        f.write("decision_policy: argmax\n")
        f.write(f"xgboost_version: {xgb.__version__}\n")
        f.write(f"device: {args.device}\n")
        f.write(f"val_split: {args.val_split}\n")
        f.write(f"class_order: {CLASS_NAMES}\n")
        f.write(f"train_counts: {train_counts.tolist()}\n")
        f.write(f"validation_counts: {val_counts.tolist()}\n")
        f.write(f"test_counts: {test_counts.tolist()}\n")
        f.write(
            "class_weights: "
            f"{class_weights.tolist() if class_weights is not None else 'not used'}\n\n"
        )
        f.write(f"n_estimators: {args.n_estimators}\n")
        f.write(f"early_stopping_rounds: {args.early_stopping_rounds}\n")
        f.write(f"best_iteration: {best_iteration}\n")
        f.write(f"max_depth: {args.max_depth}\n")
        f.write(f"learning_rate: {args.learning_rate}\n")
        f.write(f"subsample: {args.subsample}\n")
        f.write(f"colsample_bytree: {args.colsample_bytree}\n")
        f.write(f"min_child_weight: {args.min_child_weight}\n")
        f.write(f"reg_lambda: {args.reg_lambda}\n")
        f.write(f"gamma: {args.gamma}\n\n")
        f.write(f"Test Accuracy (sklearn): {metrics['accuracy']}\n")
        f.write(f"MCC: {metrics['mcc']}\n")
        f.write(f"Test Macro F1: {metrics['macro_f1']}\n")
        f.write(f"Test Macro Recall: {metrics['macro_recall']}\n")
        f.write(f"R2L Recall: {metrics['r2l_recall']}\n")
        f.write(f"U2R Recall: {metrics['u2r_recall']}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "seed",
                "train_data",
                "decision_policy",
                "accuracy",
                "mcc",
                "macro_f1",
                "macro_recall",
                "r2l_recall",
                "u2r_recall",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "model": model_label,
                "seed": args.seed,
                "train_data": "real KDDTrain+",
                "decision_policy": "argmax",
                **metrics,
            }
        )

    print(f"\n=== {model_label}: KDDTest+ ===")
    print("Class order:", CLASS_NAMES)
    print("Training class counts:", train_counts)
    print("Class weighting:", args.class_weighting)
    if class_weights is not None:
        print("Per-class weights:", class_weights)
    print(f"Accuracy:     {metrics['accuracy']:.6f}")
    print(f"MCC:          {metrics['mcc']:.6f}")
    print(f"Macro-F1:     {metrics['macro_f1']:.6f}")
    print(f"Macro recall: {metrics['macro_recall']:.6f}")
    print(f"R2L recall:   {metrics['r2l_recall']:.6f}")
    print(f"U2R recall:   {metrics['u2r_recall']:.6f}")
    print(report)
    print(f"Saved model: {model_path}")
    print(f"Saved results: {results_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
