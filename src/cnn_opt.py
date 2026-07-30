"""
cnn_opt.py — Optimized feature layout + CTGAN + Focal (NSL-KDD 5-class)
----------------------------------------------------------------------

What this script changes vs `cnn_fin.py`
---------------------------------------
- Uses an "optimized" feature-to-grid layout instead of the naive column order.
  The idea: keep semantically related numeric features adjacent, and keep the
  one-hot blocks (protocol/flag/service) contiguous, to make 2D conv locality
  less arbitrary.

What stays ON by default
------------------------
- CTGAN augmentation (expects a cached CSV, or can generate it)
- Class-balanced focal loss with best sweep defaults:
    cb_beta = 0.9999
    focal_gamma = 1.5
- Each training batch includes R2L and U2R by default
- groups configurable (groups=1 => standard conv; groups>1 => grouped conv)
- Validation-tuned class-specific score scaling:
    search R2L/U2R coefficients on real validation records
    freeze the selected pair before evaluating KDDTest+

Recommended workflow
--------------------
1) Precompute CTGAN once (R2L-only recommended):
   python -u src/cnn_opt.py --ctgan-train --ctgan-regenerate --ctgan-only --target-r2l 5000 --target-u2r 0

2) Train/eval (uses cached synth by default):
   python -u src/cnn_opt.py --epochs 25 --run-name opt_best
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

# Reuse the proven preprocessing + CTGAN utilities from `cnn_gan_foc.py`
from cnn_gan_foc import (  # type: ignore
    CLASS_NAMES,
    ClassBalancedFocalLoss,
    _generate_synth_ctgan,
    _maybe_load_synth,
    _repo_paths,
    apply_one_hot,
    apply_scaler,
    collapse_attack_labels,
    compute_cb_alpha_effective_number,
    fit_one_hot_encoders,
    fit_scaler,
    load_nsl_kdd_txt,
    prepare_xy_from_processed,
    save_confusion_matrices,
    save_multiclass_roc,
)


# --- Feature layout helpers ----------------------------------------------------

NUM_BASIC = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
]

NUM_CONTENT = [
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "is_host_login",
    "is_guest_login",
]

NUM_TRAFFIC = [
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
]

NUM_HOST = [
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


def build_optimized_feature_order(feature_cols: List[str]) -> List[str]:
    """
    Return a reordered list of the 121 feature columns (no 'class') to form an 11x11 grid
    with better locality than a naive reshape.
    """
    feature_set = set(feature_cols)
    if len(feature_cols) != 121:
        raise ValueError(f"Expected 121 feature columns, got {len(feature_cols)}")

    numeric_order = NUM_BASIC + NUM_CONTENT + NUM_TRAFFIC + NUM_HOST
    missing_numeric = [c for c in numeric_order if c not in feature_set]
    if missing_numeric:
        raise ValueError(f"Missing expected numeric columns in processed data: {missing_numeric}")

    protocol_cols = sorted([c for c in feature_cols if c.startswith("protocol_type_")])
    flag_cols = sorted([c for c in feature_cols if c.startswith("flag_")])
    service_cols = sorted([c for c in feature_cols if c.startswith("service_")])

    used = set(numeric_order) | set(protocol_cols) | set(flag_cols) | set(service_cols)
    extras = sorted([c for c in feature_cols if c not in used])

    ordered = numeric_order + protocol_cols + flag_cols + service_cols + extras

    if len(ordered) != 121 or len(set(ordered)) != 121:
        raise ValueError("Optimized feature layout produced an invalid feature ordering (dup/missing).")

    # Final safety: same set
    if set(ordered) != feature_set:
        raise ValueError("Optimized feature layout does not match the original feature set.")

    return ordered


def save_feature_grid(feature_order: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid = np.array(feature_order, dtype=object).reshape(11, 11)
    with out_path.open("w", encoding="utf-8") as f:
        for r in range(11):
            f.write("\t".join(str(x) for x in grid[r]) + "\n")


# --- Training helpers ----------------------------------------------------------


class ValF1Callback(tf.keras.callbacks.Callback):
    """Logs val_macro_f1 + val_rare_f1 so we can checkpoint on imbalance-aware metrics."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = int(batch_size)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        logs = logs or {}
        y_proba = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        macro_f1 = float(f1_score(self.y_val, y_pred, average="macro"))
        r2l_f1 = float(f1_score(self.y_val == 2, y_pred == 2))
        u2r_f1 = float(f1_score(self.y_val == 3, y_pred == 3))
        rare_f1 = float((r2l_f1 + u2r_f1) / 2.0)

        logs["val_macro_f1"] = macro_f1
        logs["val_rare_f1"] = rare_f1

        print(f" — val_macro_f1: {macro_f1:.4f} — val_rare_f1: {rare_f1:.4f}", flush=True)


def apply_class_score_scaling(
    y_proba: np.ndarray,
    class_coefficients: Dict[int, float],
) -> np.ndarray:
    """
    Divide selected class scores by positive coefficients, then take argmax.

    A coefficient below 1 promotes a class, a coefficient above 1 suppresses
    it, and 1 leaves it unchanged. The adjusted values are decision scores,
    not probabilities.
    """
    probabilities = np.asarray(y_proba, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected a 2D probability array, got shape {probabilities.shape}."
        )
    if probabilities.shape[1] == 0:
        raise ValueError("The probability array must contain at least one class.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("The probability array contains a non-finite value.")

    adjusted_scores = probabilities.copy()
    for class_id, coefficient_raw in class_coefficients.items():
        if class_id < 0 or class_id >= probabilities.shape[1]:
            raise ValueError(
                f"Score-scaling class ID is out of range: {class_id}"
            )
        coefficient = float(coefficient_raw)
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(
                f"Score coefficient for class {class_id} must be finite "
                "and greater than zero."
            )
        adjusted_scores[:, class_id] /= coefficient

    return np.argmax(adjusted_scores, axis=1).astype(np.int64)


def apply_prediction_thresholds(
    y_proba: np.ndarray,
    class_thresholds: Dict[int, float],
) -> np.ndarray:
    """Apply the legacy rare-class rejection rule used by older scripts."""
    probabilities = np.asarray(y_proba)
    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected a 2D probability array, got shape {probabilities.shape}."
        )

    for class_id, threshold in class_thresholds.items():
        if class_id < 0 or class_id >= probabilities.shape[1]:
            raise ValueError(f"Threshold class ID is out of range: {class_id}")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(
                f"Threshold for class {class_id} must be between 0 and 1."
            )

    ranked_classes = np.argsort(probabilities, axis=1)[:, ::-1]
    predictions = np.empty(len(probabilities), dtype=np.int64)
    for row_index, ranking in enumerate(ranked_classes):
        selected_class = int(ranking[0])
        for class_id_raw in ranking:
            class_id = int(class_id_raw)
            threshold = class_thresholds.get(class_id)
            if (
                threshold is None
                or probabilities[row_index, class_id] >= threshold
            ):
                selected_class = class_id
                break
        predictions[row_index] = selected_class

    return predictions


def score_coefficient_values(
    minimum: float,
    maximum: float,
    step: float,
) -> List[float]:
    """Build an inclusive coefficient grid without floating-point drift."""
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or not np.isfinite(step)
        or minimum <= 0.0
        or maximum < minimum
        or step <= 0.0
    ):
        raise ValueError(
            "Coefficient grid requires finite values, minimum > 0, "
            "maximum >= minimum, and step > 0."
        )
    values = [
        minimum + index * step
        for index in range(
            int(np.floor((maximum - minimum) / step + 1e-9)) + 1
        )
    ]
    if not np.isclose(values[-1], maximum):
        values.append(maximum)
    if minimum <= 1.0 <= maximum:
        values.append(1.0)
    return sorted({round(float(value), 10) for value in values})


def parse_score_coefficient_values(raw: str) -> List[float]:
    """Parse a comma-separated, positive score-coefficient list."""
    try:
        values = [
            float(value.strip())
            for value in raw.split(",")
            if value.strip()
        ]
    except ValueError as exc:
        raise ValueError(
            "Score coefficients must be comma-separated numbers."
        ) from exc
    if not values:
        raise ValueError("Provide at least one score coefficient.")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            "Every score coefficient must be finite and greater than zero."
        )
    return sorted({round(value, 10) for value in values})


def score_scaling_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    raw_y_pred: np.ndarray,
) -> Dict[str, float]:
    """Metrics used to rank validation-set score coefficients."""
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
    changed = int(np.count_nonzero(y_pred != raw_y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=np.arange(5),
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(np.mean(per_class_recall)),
        "r2l_recall": float(per_class_recall[2]),
        "u2r_recall": float(per_class_recall[3]),
        "minority_recall": float(np.mean(per_class_recall[[2, 3]])),
        "minimum_minority_recall": float(
            np.min(per_class_recall[[2, 3]])
        ),
        "minority_recall_gap": float(
            abs(per_class_recall[2] - per_class_recall[3])
        ),
        "r2l_f1": float(per_class_f1[2]),
        "u2r_f1": float(per_class_f1[3]),
        "rare_f1": float(np.mean(per_class_f1[[2, 3]])),
        "changed_predictions": float(changed),
        "change_rate": float(changed / len(y_true)),
    }


def search_score_coefficients(
    y_val: np.ndarray,
    val_probabilities: np.ndarray,
    candidates: List[float],
    macro_f1_retention: float = 0.90,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    Select coefficients using validation data only.

    Candidates must first retain the requested fraction of raw validation
    macro-F1. Among eligible pairs, ranking maximizes the weaker of R2L/U2R
    recall, then their mean. Remaining metrics favor precise, balanced, and
    less-disruptive predictions.
    """
    y_val = np.asarray(y_val)
    probabilities = np.asarray(val_probabilities)
    if probabilities.ndim != 2 or probabilities.shape[1] != 5:
        raise ValueError(
            f"Expected validation probability shape (n, 5), "
            f"got {probabilities.shape}."
        )
    if len(y_val) == 0 or len(y_val) != len(probabilities):
        raise ValueError(
            "Validation labels and probabilities must have the same "
            "non-zero length."
        )
    if not candidates:
        raise ValueError("The score-coefficient grid cannot be empty.")
    if not 0.0 <= macro_f1_retention <= 1.0:
        raise ValueError("Macro-F1 retention must be between 0 and 1.")

    raw_predictions = np.argmax(probabilities, axis=1)
    raw_metrics = score_scaling_metrics(
        y_val,
        raw_predictions,
        raw_predictions,
    )
    minimum_macro_f1 = (
        raw_metrics["macro_f1"] * float(macro_f1_retention)
    )
    rows: List[Dict[str, float]] = []
    for r2l_coefficient in candidates:
        for u2r_coefficient in candidates:
            predictions = apply_class_score_scaling(
                probabilities,
                {
                    2: r2l_coefficient,
                    3: u2r_coefficient,
                },
            )
            metrics = score_scaling_metrics(
                y_val,
                predictions,
                raw_predictions,
            )
            rows.append(
                {
                    "r2l_score_coefficient": float(r2l_coefficient),
                    "u2r_score_coefficient": float(u2r_coefficient),
                    "distance_from_argmax": float(
                        abs(np.log(r2l_coefficient))
                        + abs(np.log(u2r_coefficient))
                    ),
                    "minimum_allowed_macro_f1": minimum_macro_f1,
                    "meets_macro_f1_retention": float(
                        metrics["macro_f1"] >= minimum_macro_f1
                    ),
                    **metrics,
                }
            )

    rows.sort(
        key=lambda row: (
            -row["meets_macro_f1_retention"],
            -row["minimum_minority_recall"],
            -row["minority_recall"],
            -row["rare_f1"],
            row["minority_recall_gap"],
            -row["macro_f1"],
            -row["mcc"],
            -row["accuracy"],
            row["change_rate"],
            row["distance_from_argmax"],
            row["r2l_score_coefficient"],
            row["u2r_score_coefficient"],
        )
    )
    return rows[0], rows


def select_synth_for_targets(
    synth_pool: pd.DataFrame,
    real_labels: np.ndarray,
    targets: Dict[int, int],
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Take deterministic class-specific subsets from one cached synthetic pool.

    Each target is the desired real+synthetic class total. For example, if
    real R2L has 995 records, target 5000 selects 4005 cached R2L records.
    """
    if "class" not in synth_pool.columns:
        raise ValueError("The synthetic pool must contain a 'class' column.")

    real_labels = np.asarray(real_labels, dtype=np.int64)
    if np.any((real_labels < 0) | (real_labels >= 5)):
        raise ValueError("Real training labels must be class IDs from 0 to 4.")
    real_counts = np.bincount(real_labels, minlength=5)

    numeric_labels = pd.to_numeric(synth_pool["class"], errors="coerce")
    if numeric_labels.isna().any():
        raise ValueError("The synthetic pool contains a non-numeric class.")
    pool_labels = numeric_labels.to_numpy(dtype=np.float64)
    if np.any(pool_labels != np.floor(pool_labels)):
        raise ValueError("Synthetic class IDs must be integers.")
    pool_labels = pool_labels.astype(np.int64)
    if np.any((pool_labels < 0) | (pool_labels >= 5)):
        raise ValueError("Synthetic class IDs must be between 0 and 4.")
    pool_counts = np.bincount(pool_labels, minlength=5)

    selected_indices: List[int] = []
    for class_id in range(5):
        target = int(targets.get(class_id, 0))
        if target < 0:
            raise ValueError("Synthetic class targets cannot be negative.")
        needed = max(0, target - int(real_counts[class_id]))
        available_indices = np.flatnonzero(pool_labels == class_id)
        if needed > len(available_indices):
            class_name = CLASS_NAMES[class_id]
            raise ValueError(
                f"Target {target} for {class_name} needs {needed} synthetic "
                f"rows, but the cached pool contains only "
                f"{len(available_indices)}. Generate a larger synthetic pool."
            )
        selected_indices.extend(
            int(index) for index in available_indices[:needed]
        )

    selected = synth_pool.iloc[selected_indices].copy().reset_index(drop=True)
    selected["class"] = selected["class"].astype(np.int64)
    return selected, real_counts, pool_counts


class BalancedBatchSequence(tf.keras.utils.Sequence):
    """Create batches that always include R2L and U2R samples when available."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        minority_per_batch: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.X = X
        self.y = np.asarray(y)
        self.batch_size = int(batch_size)
        self.minority_per_batch = int(minority_per_batch)
        self.seed = int(seed)
        self.epoch = 0

        if len(self.X) != len(self.y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(self.y) == 0:
            raise ValueError("Training data cannot be empty.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if self.minority_per_batch <= 0:
            raise ValueError("minority_per_batch must be greater than 0.")

        # NSL-KDD labels: 2 = R2L and 3 = U2R.
        self.minority_indices = {
            class_id: np.flatnonzero(self.y == class_id) for class_id in (2, 3)
        }
        self.available_minority_classes = [
            class_id for class_id, indices in self.minority_indices.items() if len(indices) > 0
        ]

        guaranteed_count = self.minority_per_batch * len(self.available_minority_classes)
        if guaranteed_count > self.batch_size:
            raise ValueError(
                "batch_size is too small for the requested minority samples per batch."
            )

        self.all_indices = np.arange(len(self.y))
        self.steps_per_epoch = int(np.ceil(len(self.y) / self.batch_size))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, batch_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("Batch index is out of range.")

        # A repeatable but different random selection for every batch and epoch.
        rng = np.random.default_rng(
            self.seed + self.epoch * self.steps_per_epoch + int(batch_index)
        )

        selected: List[int] = []
        for class_id in self.available_minority_classes:
            class_indices = self.minority_indices[class_id]
            chosen = rng.choice(
                class_indices,
                size=self.minority_per_batch,
                replace=len(class_indices) < self.minority_per_batch,
            )
            selected.extend(chosen.tolist())

        # Fill the rest of the batch from the complete training set.
        remaining = self.batch_size - len(selected)
        if remaining > 0:
            selected.extend(
                rng.choice(self.all_indices, size=remaining, replace=True).tolist()
            )

        batch_indices = np.asarray(selected, dtype=np.int64)
        rng.shuffle(batch_indices)
        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self) -> None:
        self.epoch += 1


def build_opt_cnn(
    loss: tf.keras.losses.Loss,
    groups: int = 1,
    base_filters: int = 64,
    dense_units: int = 256,
    dropout1: float = 0.25,
    dropout2: float = 0.30,
    use_batch_norm: bool = True,
    use_residual: bool = True,
) -> tf.keras.Model:
    """
    CNN for 11x11 grayscale:
    - stem conv
    - optional grouped conv block (groups param)
    - 1x1 channel mixing + optional residual
    - dense head
    """
    groups = int(groups)
    base_filters = int(base_filters)
    if base_filters % groups != 0:
        raise ValueError(f"--groups must divide --base-filters. Got groups={groups}, base_filters={base_filters}")

    inputs = tf.keras.Input(shape=(11, 11, 1))

    # Stem (must be groups=1 because in_channels=1)
    x = tf.keras.layers.Conv2D(base_filters, 3, padding="same", use_bias=not use_batch_norm)(inputs)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    shortcut = x
    x = tf.keras.layers.Conv2D(
        base_filters,
        3,
        padding="same",
        groups=groups,
        use_bias=not use_batch_norm,
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Channel mixing
    x = tf.keras.layers.Conv2D(base_filters, 1, padding="same", use_bias=not use_batch_norm)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    if use_residual:
        x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(int(dense_units), activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_opt")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def save_training_plot(history: tf.keras.callbacks.History, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(history.history.get("accuracy", []), label="Train Acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Acc")
    if "val_macro_f1" in history.history:
        plt.plot(history.history["val_macro_f1"], label="Val Macro-F1")
    if "val_rare_f1" in history.history:
        plt.plot(history.history["val_rare_f1"], label="Val Rare-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_training_plot.png", dpi=400)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NSL-KDD 5-class: optimized feature layout + CTGAN + focal.")

    parser.add_argument("--run-name", type=str, default="opt", help="Prefix for outputs in results/ and model/.")

    # Training
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument(
        "--minority-per-batch",
        type=int,
        default=1,
        help="Guaranteed R2L and U2R samples per training batch. Use 0 for ordinary batching.",
    )

    # Focal (best sweep defaults)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--cb-beta", type=float, default=0.9999)

    # CTGAN (ON by default: if synth file missing, you must --ctgan-train)
    parser.add_argument("--synth-path", type=str, default="data/synth_ctgan_5class.csv")
    parser.add_argument("--ctgan-train", action="store_true")
    parser.add_argument("--ctgan-regenerate", action="store_true")
    parser.add_argument("--ctgan-only", action="store_true")
    parser.add_argument("--ctgan-epochs", type=int, default=200)
    parser.add_argument("--ctgan-batch-size", type=int, default=4096)
    parser.add_argument("--ctgan-pac", type=int, default=10)

    # CTGAN targets (defaults: R2L-only to 5000, U2R disabled)
    parser.add_argument("--target-dos", type=int, default=0)
    parser.add_argument("--target-probe", type=int, default=0)
    parser.add_argument(
        "--target-r2l",
        type=int,
        default=5000,
        help=(
            "Desired real+synthetic R2L total before the real validation "
            "holdout; this is not the number of rows generated."
        ),
    )
    parser.add_argument("--target-u2r", type=int, default=0)
    parser.add_argument("--target-normal", type=int, default=0)

    # Feature layout
    parser.add_argument(
        "--feature-layout",
        choices=["optimized", "legacy"],
        default="optimized",
        help="optimized: semantic grouping + one-hot blocks; legacy: original column order.",
    )

    # CNN architecture
    parser.add_argument("--groups", type=int, default=1, help="groups=1 is standard conv; >1 enables grouped conv.")
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout1", type=float, default=0.25)
    parser.add_argument("--dropout2", type=float, default=0.30)
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--no-residual", action="store_true")

    # Post-training score-scaling policy.
    parser.add_argument(
        "--r2l-score-coefficient",
        "--r2l-threshold",
        dest="r2l_score_coefficient",
        type=float,
        default=None,
        help=(
            "Use this fixed R2L coefficient. If both class coefficients are "
            "omitted, they are selected on validation data."
        ),
    )
    parser.add_argument(
        "--u2r-score-coefficient",
        "--u2r-threshold",
        dest="u2r_score_coefficient",
        type=float,
        default=None,
        help=(
            "Use this fixed U2R coefficient. If both class coefficients are "
            "omitted, they are selected on validation data."
        ),
    )
    parser.add_argument("--coefficient-min", type=float, default=0.05)
    parser.add_argument("--coefficient-max", type=float, default=2.00)
    parser.add_argument("--coefficient-step", type=float, default=0.15)
    parser.add_argument(
        "--coefficient-values",
        default=None,
        help=(
            "Optional comma-separated coefficient grid. When provided, "
            "it replaces coefficient-min/max/step."
        ),
    )
    parser.add_argument(
        "--min-validation-macro-f1-retention",
        type=float,
        default=0.90,
        help=(
            "Coefficient pairs must retain at least this fraction of raw "
            "validation macro-F1 before minority recall is optimized."
        ),
    )
    parser.add_argument(
        "--no-score-scaling",
        "--no-thresholds",
        dest="no_score_scaling",
        action="store_true",
        help="Use ordinary argmax instead of class-specific score scaling.",
    )

    args = parser.parse_args()
    if args.minority_per_batch < 0:
        parser.error("--minority-per-batch must be 0 or greater.")
    for option_name, target in [
        ("--target-dos", args.target_dos),
        ("--target-probe", args.target_probe),
        ("--target-r2l", args.target_r2l),
        ("--target-u2r", args.target_u2r),
        ("--target-normal", args.target_normal),
    ]:
        if target < 0:
            parser.error(f"{option_name} must be 0 or greater.")
    fixed_coefficients_given = (
        args.r2l_score_coefficient is not None
        or args.u2r_score_coefficient is not None
    )
    if fixed_coefficients_given and (
        args.r2l_score_coefficient is None
        or args.u2r_score_coefficient is None
    ):
        parser.error(
            "Provide both --r2l-score-coefficient and "
            "--u2r-score-coefficient, or omit both to tune them."
        )
    for option_name, coefficient in [
        ("--r2l-score-coefficient", args.r2l_score_coefficient),
        ("--u2r-score-coefficient", args.u2r_score_coefficient),
    ]:
        if coefficient is not None and (
            not np.isfinite(coefficient) or coefficient <= 0.0
        ):
            parser.error(
                f"{option_name} must be finite and greater than zero."
            )
    if args.no_score_scaling and fixed_coefficients_given:
        parser.error(
            "--no-score-scaling cannot be combined with fixed score "
            "coefficients."
        )
    if not np.isfinite(args.coefficient_min) or args.coefficient_min <= 0.0:
        parser.error("--coefficient-min must be finite and greater than zero.")
    if (
        not np.isfinite(args.coefficient_max)
        or args.coefficient_max < args.coefficient_min
    ):
        parser.error(
            "--coefficient-max must be finite and at least --coefficient-min."
        )
    if not np.isfinite(args.coefficient_step) or args.coefficient_step <= 0.0:
        parser.error("--coefficient-step must be finite and greater than zero.")
    if not (
        np.isfinite(args.min_validation_macro_f1_retention)
        and 0.0 <= args.min_validation_macro_f1_retention <= 1.0
    ):
        parser.error(
            "--min-validation-macro-f1-retention must be between 0 and 1."
        )
    auto_tune_score_scaling = (
        not args.no_score_scaling and not fixed_coefficients_given
    )
    explicit_coefficient_candidates: List[float] | None = None
    if args.coefficient_values and not auto_tune_score_scaling:
        parser.error(
            "--coefficient-values is only used when coefficients are "
            "selected by validation search."
        )
    if auto_tune_score_scaling:
        if args.coefficient_values:
            try:
                explicit_coefficient_candidates = (
                    parse_score_coefficient_values(args.coefficient_values)
                )
            except ValueError as error:
                parser.error(str(error))
            if 1.0 not in explicit_coefficient_candidates:
                parser.error(
                    "--coefficient-values must include 1.0, which is the "
                    "raw-argmax operating point."
                )
            estimated_candidates = len(explicit_coefficient_candidates)
        else:
            if not args.coefficient_min <= 1.0 <= args.coefficient_max:
                parser.error(
                    "The coefficient search range must include 1.0, which "
                    "is the raw-argmax operating point."
                )
            estimated_candidates = (
                (args.coefficient_max - args.coefficient_min)
                / args.coefficient_step
                + 2.0
            )
        if not np.isfinite(estimated_candidates) or (
            estimated_candidates > np.sqrt(100_000)
        ):
            parser.error(
                "The coefficient grid is too large. Increase "
                "--coefficient-step or narrow the search range."
            )

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    paths = _repo_paths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    # Load and label-map
    train_df = load_nsl_kdd_txt(paths.data_dir / "KDDTrain+.txt").drop(columns=["num_outbound_cmds"])
    test_df = load_nsl_kdd_txt(paths.data_dir / "KDDTest+.txt").drop(columns=["num_outbound_cmds"])
    train_df = collapse_attack_labels(train_df, is_train=True)
    test_df = collapse_attack_labels(test_df, is_train=False)

    # CTGAN: load or generate synth
    synth_path = (paths.repo_root / args.synth_path).resolve()
    synth_df = None if args.ctgan_regenerate else _maybe_load_synth(synth_path)
    targets = {
        0: args.target_dos,
        1: args.target_probe,
        2: args.target_r2l,
        3: args.target_u2r,
        4: args.target_normal,
    }

    if synth_df is None:
        if args.ctgan_train:
            synth_df = _generate_synth_ctgan(
                train_df_raw=train_df,
                seed=args.seed,
                epochs=args.ctgan_epochs,
                batch_size=args.ctgan_batch_size,
                pac=args.ctgan_pac,
                targets=targets,
            )
            synth_path.parent.mkdir(parents=True, exist_ok=True)
            synth_df.to_csv(synth_path, index=False)
        else:
            raise ValueError(
                f"CTGAN synth file not found at {synth_path}. "
                "Precompute it with --ctgan-train (or point --synth-path to an existing CSV)."
            )

    synth_pool_df = synth_df
    missing_synth_columns = sorted(
        set(train_df.columns) - set(synth_pool_df.columns)
    )
    extra_synth_columns = sorted(
        set(synth_pool_df.columns) - set(train_df.columns)
    )
    if missing_synth_columns or extra_synth_columns:
        raise ValueError(
            "Synthetic pool columns do not match KDDTrain+. "
            f"Missing={missing_synth_columns}, extra={extra_synth_columns}"
        )
    synth_pool_df = synth_pool_df[train_df.columns]
    synth_df, real_counts, synth_pool_counts = select_synth_for_targets(
        synth_pool=synth_pool_df,
        real_labels=train_df["class"].to_numpy(dtype=np.int64),
        targets=targets,
    )

    if args.ctgan_only:
        synth_counts = (
            np.bincount(synth_df["class"].to_numpy(dtype=np.int64), minlength=5)
            if len(synth_df)
            else np.zeros(5, dtype=np.int64)
        )
        print("\n=== CTGAN precompute complete (ctgan-only) ===")
        print("Synth path:", synth_path)
        print("Targets:", targets)
        print("Cached pool class counts:", synth_pool_counts)
        print("Selected synth class counts:", synth_counts)
        return

    # Combine train + synth
    source_column = "__is_real_training_record__"
    if source_column in train_df.columns or source_column in synth_df.columns:
        raise ValueError(f"Unexpected reserved column: {source_column}")
    real_tagged = train_df.copy()
    synth_tagged = synth_df.copy()
    real_tagged[source_column] = True
    synth_tagged[source_column] = False
    train_aug_tagged = (
        pd.concat([real_tagged, synth_tagged], ignore_index=True)
        .reset_index(drop=True)
    )
    source_is_real = train_aug_tagged.pop(source_column).to_numpy(dtype=bool)
    train_aug_df = train_aug_tagged

    # Fit OHE + scaler on REAL train only (stable 121 layout)
    categorical_columns = ["protocol_type", "service", "flag"]
    columns_to_scale = [
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

    encoders, feature_names = fit_one_hot_encoders(train_df, categorical_columns)
    train_real_ohe = apply_one_hot(train_df, encoders, feature_names, categorical_columns)
    scaler = fit_scaler(train_real_ohe, columns_to_scale)

    train_proc = apply_scaler(apply_one_hot(train_aug_df, encoders, feature_names, categorical_columns), scaler, columns_to_scale)
    test_proc = apply_scaler(apply_one_hot(test_df, encoders, feature_names, categorical_columns), scaler, columns_to_scale)

    # Align test columns to train columns
    test_proc = test_proc[train_proc.columns]

    # Apply feature layout reorder (affects the 11x11 mapping only)
    feature_cols = [c for c in train_proc.columns if c != "class"]
    if args.feature_layout == "optimized":
        ordered_features = build_optimized_feature_order(feature_cols)
    else:
        ordered_features = feature_cols

    ordered_cols = ordered_features + ["class"]
    train_proc = train_proc[ordered_cols]
    test_proc = test_proc[ordered_cols]

    # Save the grid map for reference
    grid_path = paths.results_dir / f"{args.run_name}_feature_grid.tsv"
    save_feature_grid(ordered_features, grid_path)

    X_all, y_all, X_test, y_test = prepare_xy_from_processed(train_proc, test_proc)

    # Split only REAL records so every augmentation target uses the same
    # validation fold. Synthetic records are added only to model training.
    X_real = X_all[source_is_real]
    y_real = y_all[source_is_real]
    X_synth = X_all[~source_is_real]
    y_synth = y_all[~source_is_real]
    (
        X_real_tr,
        X_val,
        y_real_tr,
        y_val,
    ) = train_test_split(
        X_real,
        y_real,
        test_size=float(args.val_split),
        random_state=args.seed,
        stratify=y_real,
    )
    X_tr = np.concatenate([X_real_tr, X_synth], axis=0)
    y_tr = np.concatenate([y_real_tr, y_synth], axis=0)
    train_order = np.random.default_rng(args.seed).permutation(len(y_tr))
    X_tr = X_tr[train_order]
    y_tr = y_tr[train_order]
    real_validation_counts = np.bincount(y_val, minlength=5)

    # Focal alpha on training split
    alpha, alpha_counts = compute_cb_alpha_effective_number(y_tr, beta=args.cb_beta, num_classes=5)
    loss_obj = ClassBalancedFocalLoss(alpha=alpha, gamma=args.focal_gamma)

    model = build_opt_cnn(
        loss=loss_obj,
        groups=args.groups,
        base_filters=args.base_filters,
        dense_units=args.dense_units,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        use_batch_norm=not args.no_bn,
        use_residual=not args.no_residual,
    )
    model_parameters = int(model.count_params())

    prefix = args.run_name.strip() or "opt"
    weights_path = paths.model_dir / f"{prefix}_best.weights.h5"
    model_path = paths.model_dir / f"{prefix}_best.keras"

    callbacks: List[tf.keras.callbacks.Callback] = [
        ValF1Callback(X_val=X_val, y_val=y_val, batch_size=args.batch_size),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=8,
            restore_best_weights=False,
        ),
    ]

    if args.minority_per_batch > 0:
        train_batches = BalancedBatchSequence(
            X=X_tr,
            y=y_tr,
            batch_size=args.batch_size,
            minority_per_batch=args.minority_per_batch,
            seed=args.seed,
        )
        print(
            f"Balanced batches enabled: at least {args.minority_per_batch} R2L "
            f"and {args.minority_per_batch} U2R sample(s) per batch."
        )
        history = model.fit(
            train_batches,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            callbacks=callbacks,
        )

    save_training_plot(history, paths.results_dir, prefix)

    # Load best weights and save the model
    if weights_path.exists():
        model.load_weights(weights_path)
    model.save(model_path)
    best_val_macro_f1 = max(history.history.get("val_macro_f1", [0.0]))

    # Select the deployment policy without looking at KDDTest+.
    coefficient_search_path: Path | None = None
    coefficient_pair_count = 0
    raw_validation_metrics: Dict[str, float] | None = None
    selected_validation_metrics: Dict[str, float] | None = None
    if args.no_score_scaling:
        score_scaling_selection = "disabled"
        class_score_coefficients = {2: 1.0, 3: 1.0}
    elif fixed_coefficients_given:
        score_scaling_selection = "fixed_cli_coefficients"
        class_score_coefficients = {
            2: float(args.r2l_score_coefficient),
            3: float(args.u2r_score_coefficient),
        }
    else:
        score_scaling_selection = "validation_grid_search"
        if (
            real_validation_counts[2] == 0
            or real_validation_counts[3] == 0
        ):
            raise ValueError(
                "Cannot tune score coefficients because the real validation "
                "subset does not contain both R2L and U2R records."
            )
        candidates = (
            explicit_coefficient_candidates
            if explicit_coefficient_candidates is not None
            else score_coefficient_values(
                args.coefficient_min,
                args.coefficient_max,
                args.coefficient_step,
            )
        )
        coefficient_pair_count = len(candidates) ** 2
        if coefficient_pair_count > 100_000:
            raise ValueError(
                "The coefficient grid contains "
                f"{coefficient_pair_count} pairs. "
                "Increase --coefficient-step or narrow the search range."
            )
        print(
            f"Searching {coefficient_pair_count} R2L/U2R "
            "score-coefficient pairs "
            "on real validation records..."
        )
        print("Real validation class counts:", real_validation_counts)
        val_probabilities = model.predict(
            X_val,
            batch_size=args.batch_size,
            verbose=0,
        )
        raw_val_predictions = np.argmax(val_probabilities, axis=1)
        raw_validation_metrics = score_scaling_metrics(
            y_val,
            raw_val_predictions,
            raw_val_predictions,
        )
        (
            selected_validation_metrics,
            search_rows,
        ) = search_score_coefficients(
            y_val,
            val_probabilities,
            candidates,
            macro_f1_retention=(
                args.min_validation_macro_f1_retention
            ),
        )
        class_score_coefficients = {
            2: selected_validation_metrics["r2l_score_coefficient"],
            3: selected_validation_metrics["u2r_score_coefficient"],
        }
        coefficient_search_path = (
            paths.results_dir / f"{prefix}_score_scaling_search.csv"
        )
        search_frame = pd.DataFrame(search_rows)
        search_frame.insert(
            0,
            "rank",
            np.arange(1, len(search_frame) + 1),
        )
        search_frame.to_csv(coefficient_search_path, index=False)
        print(
            "Selected validation coefficients: "
            f"R2L={class_score_coefficients[2]:.4f}, "
            f"U2R={class_score_coefficients[3]:.4f}"
        )
        boundary_hits = []
        if np.isclose(
            class_score_coefficients[2],
            args.coefficient_min,
        ):
            boundary_hits.append("R2L minimum")
        if np.isclose(
            class_score_coefficients[2],
            args.coefficient_max,
        ):
            boundary_hits.append("R2L maximum")
        if np.isclose(
            class_score_coefficients[3],
            args.coefficient_min,
        ):
            boundary_hits.append("U2R minimum")
        if np.isclose(
            class_score_coefficients[3],
            args.coefficient_max,
        ):
            boundary_hits.append("U2R maximum")
        if boundary_hits:
            print(
                "WARNING: selected coefficient reached the search boundary "
                f"({', '.join(boundary_hits)})."
            )

    score_scaling_applied = not args.no_score_scaling

    # The coefficient pair is now frozen; evaluate KDDTest+ exactly once.
    loss, keras_argmax_acc = model.evaluate(X_test, y_test, verbose=0)
    y_proba = model.predict(X_test, verbose=0)
    raw_y_pred = np.argmax(y_proba, axis=1)
    y_pred = (
        apply_class_score_scaling(y_proba, class_score_coefficients)
        if score_scaling_applied
        else raw_y_pred.copy()
    )
    changed_predictions = int(np.count_nonzero(y_pred != raw_y_pred))
    r2l_losses = int(np.count_nonzero((raw_y_pred == 2) & (y_pred != 2)))
    u2r_losses = int(np.count_nonzero((raw_y_pred == 3) & (y_pred != 3)))
    r2l_gains = int(np.count_nonzero((raw_y_pred != 2) & (y_pred == 2)))
    u2r_gains = int(np.count_nonzero((raw_y_pred != 3) & (y_pred == 3)))

    raw_report = classification_report(
        y_test,
        raw_y_pred,
        digits=8,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        digits=8,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    raw_test_macro_f1 = float(f1_score(y_test, raw_y_pred, average="macro"))
    raw_per_class_recall = recall_score(
        y_test,
        raw_y_pred,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    raw_macro_recall = float(np.mean(raw_per_class_recall))
    raw_mcc = matthews_corrcoef(y_test, raw_y_pred)
    raw_accuracy = accuracy_score(y_test, raw_y_pred)
    test_macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    per_class_recall = recall_score(
        y_test,
        y_pred,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    macro_recall = float(np.mean(per_class_recall))
    mcc = matthews_corrcoef(y_test, y_pred)
    acc_sklearn = accuracy_score(y_test, y_pred)

    if score_scaling_applied:
        save_confusion_matrices(
            y_test,
            raw_y_pred,
            paths.results_dir,
            f"{prefix}_raw_argmax",
        )
    save_confusion_matrices(y_test, y_pred, paths.results_dir, prefix)
    save_multiclass_roc(y_test, y_proba, paths.results_dir, prefix)

    # Count summary
    synth_counts = np.bincount(synth_df["class"].to_numpy(dtype=np.int64), minlength=5) if len(synth_df) else np.zeros(5, dtype=np.int64)
    aug_counts = np.bincount(y_all, minlength=5)
    training_counts = np.bincount(y_tr, minlength=5)

    out_txt = paths.results_dir / f"{prefix}_results.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("CNN_OPT (Optimized feature layout + CTGAN + Focal)\n\n")
        f.write(f"run_name: {prefix}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"epochs: {args.epochs}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"val_split: {args.val_split}\n")
        f.write(f"feature_layout: {args.feature_layout}\n")
        f.write(f"feature_grid_tsv: {grid_path}\n\n")
        f.write(f"synth_path: {synth_path}\n")
        f.write(f"targets: {targets}\n")
        f.write(f"target_r2l: {args.target_r2l}\n")
        f.write(f"target_u2r: {args.target_u2r}\n")
        f.write(f"real_counts: {real_counts.tolist()}\n")
        f.write(f"synth_pool_counts: {synth_pool_counts.tolist()}\n")
        f.write(f"synth_counts: {synth_counts.tolist()}\n")
        f.write(f"aug_counts: {aug_counts.tolist()}\n\n")
        f.write(f"training_counts: {training_counts.tolist()}\n")
        f.write(f"r2l_synthetic_rows_used: {int(synth_counts[2])}\n\n")
        f.write(f"focal_gamma: {args.focal_gamma}\n")
        f.write(f"cb_beta: {args.cb_beta}\n")
        f.write(f"minority_per_batch: {args.minority_per_batch}\n")
        f.write(f"alpha_counts(train_split): {alpha_counts.tolist()}\n")
        f.write(f"alpha: {[float(x) for x in alpha.tolist()]}\n\n")
        f.write(f"groups: {args.groups}\n")
        f.write(f"base_filters: {args.base_filters}\n")
        f.write(f"dense_units: {args.dense_units}\n")
        f.write(f"dropout1: {args.dropout1}\n")
        f.write(f"dropout2: {args.dropout2}\n")
        f.write(f"use_batch_norm: {not args.no_bn}\n")
        f.write(f"use_residual: {not args.no_residual}\n\n")
        f.write(f"Model Parameters: {model_parameters}\n")
        f.write(f"Best Validation Macro F1: {best_val_macro_f1}\n\n")
        decision_policy = (
            "class_specific_score_scaling"
            if score_scaling_applied
            else "raw_argmax"
        )
        f.write(f"decision_policy: {decision_policy}\n")
        f.write(f"score_scaling_applied: {score_scaling_applied}\n")
        f.write(f"score_scaling_selection: {score_scaling_selection}\n")
        if score_scaling_selection == "validation_grid_search":
            f.write(
                "score_scaling_selection_data: "
                "real_validation_records_only\n"
            )
            f.write(
                "score_scaling_selection_objective: retain validation "
                "macro-F1, then maximize the lower R2L/U2R recall, their "
                "mean, and rare-class F1\n"
            )
        elif score_scaling_selection == "fixed_cli_coefficients":
            f.write("score_scaling_selection_data: command_line\n")
        else:
            f.write("score_scaling_selection_data: not_applicable\n")
        f.write(
            f"real_validation_counts: "
            f"{real_validation_counts.tolist()}\n"
        )
        f.write(
            "r2l_score_coefficient: "
            f"{class_score_coefficients[2]}\n"
        )
        f.write(
            "u2r_score_coefficient: "
            f"{class_score_coefficients[3]}\n"
        )
        f.write(
            "r2l_score_multiplier: "
            f"{1.0 / class_score_coefficients[2]}\n"
        )
        f.write(
            "u2r_score_multiplier: "
            f"{1.0 / class_score_coefficients[3]}\n"
        )
        f.write(f"coefficient_min: {args.coefficient_min}\n")
        f.write(f"coefficient_max: {args.coefficient_max}\n")
        f.write(f"coefficient_step: {args.coefficient_step}\n")
        f.write(
            "coefficient_values: "
            f"{args.coefficient_values or 'uniform_min_max_step'}\n"
        )
        f.write(
            "min_validation_macro_f1_retention: "
            f"{args.min_validation_macro_f1_retention}\n"
        )
        f.write(f"coefficient_pairs_searched: {coefficient_pair_count}\n")
        f.write(f"coefficient_search_csv: {coefficient_search_path}\n")
        if (
            raw_validation_metrics is not None
            and selected_validation_metrics is not None
        ):
            for metric_name in [
                "accuracy",
                "mcc",
                "macro_f1",
                "macro_recall",
                "r2l_recall",
                "u2r_recall",
                "minimum_minority_recall",
                "minority_recall",
                "rare_f1",
            ]:
                f.write(
                    f"raw_validation_{metric_name}: "
                    f"{raw_validation_metrics[metric_name]}\n"
                )
                f.write(
                    f"selected_validation_{metric_name}: "
                    f"{selected_validation_metrics[metric_name]}\n"
                )
        f.write(
            f"score_scaling_changed_predictions: {changed_predictions}\n\n"
        )
        f.write(f"r2l_predictions_lost_after_scaling: {r2l_losses}\n")
        f.write(f"u2r_predictions_lost_after_scaling: {u2r_losses}\n")
        f.write(f"r2l_predictions_gained_after_scaling: {r2l_gains}\n")
        f.write(f"u2r_predictions_gained_after_scaling: {u2r_gains}\n\n")
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy (keras): {keras_argmax_acc}\n")
        f.write(f"Raw Argmax Test Accuracy: {raw_accuracy}\n")
        f.write(f"Raw Argmax Test Macro F1: {raw_test_macro_f1}\n")
        f.write(f"Raw Argmax Test Macro Recall: {raw_macro_recall}\n")
        f.write(f"Raw Argmax R2L Recall: {float(raw_per_class_recall[2])}\n")
        f.write(f"Raw Argmax U2R Recall: {float(raw_per_class_recall[3])}\n")
        f.write(f"Raw Argmax MCC: {raw_mcc}\n")
        f.write(f"Test Accuracy (sklearn): {acc_sklearn}\n")
        f.write(f"Test Macro F1: {test_macro_f1}\n")
        f.write(f"Test Macro Recall: {macro_recall}\n")
        f.write(f"R2L Recall: {float(per_class_recall[2])}\n")
        f.write(f"U2R Recall: {float(per_class_recall[3])}\n")
        f.write(f"MCC: {mcc}\n\n")
        f.write("Raw argmax report:\n")
        f.write(raw_report)
        f.write("\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    print("\n=== CNN_OPT Test Results ===")
    print("Feature layout:", args.feature_layout)
    print("Feature grid:", grid_path)
    print("Real train counts:", real_counts)
    print("Cached synth pool counts:", synth_pool_counts)
    print("Selected synth counts:", synth_counts)
    print("Aug counts:", aug_counts)
    print("Actual training counts:", training_counts)
    print("Alpha:", alpha)
    print("Score scaling applied:", score_scaling_applied)
    print("Score coefficient selection:", score_scaling_selection)
    print("Class score coefficients:", class_score_coefficients)
    print("Changed predictions:", changed_predictions)
    print("R2L predictions lost / gained:", r2l_losses, "/", r2l_gains)
    print("U2R predictions lost / gained:", u2r_losses, "/", u2r_gains)
    print(f"Test loss: {loss:.6f}")
    print(f"Raw argmax accuracy: {raw_accuracy:.6f}")
    print(f"Raw argmax macro-F1: {raw_test_macro_f1:.6f}")
    print(f"Raw argmax macro recall: {raw_macro_recall:.6f}")
    print(f"Raw argmax R2L recall: {raw_per_class_recall[2]:.6f}")
    print(f"Raw argmax U2R recall: {raw_per_class_recall[3]:.6f}")
    print(f"Raw argmax MCC: {raw_mcc:.6f}")
    print(f"Final accuracy: {acc_sklearn:.6f}")
    print(f"Final macro-F1: {test_macro_f1:.6f}")
    print(f"Final macro recall: {macro_recall:.6f}")
    print(f"Final R2L recall: {per_class_recall[2]:.6f}")
    print(f"Final U2R recall: {per_class_recall[3]:.6f}")
    print(f"Final MCC: {mcc:.6f}")
    print(f"Model parameters: {model_parameters}")
    print(f"Best validation macro-F1: {best_val_macro_f1:.6f}")
    print(report)
    print(f"\nSaved: {out_txt}")
    print(f"Saved model: {model_path}")
    if coefficient_search_path is not None:
        print(f"Saved coefficient search: {coefficient_search_path}")


if __name__ == "__main__":
    main()
