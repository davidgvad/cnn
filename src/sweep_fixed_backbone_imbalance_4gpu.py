"""
Seeded imbalance-control sweep for five fixed NSL-KDD model backbones.

This script deliberately does not tune architecture hyperparameters and never
loads KDDTest+.  For each neural backbone it trains every declared (beta,
focal-gamma) pair on the same three seed-specific, stratified KDDTrain+ folds.
The two fixed XGBoost variants are trained on those folds as well.  Validation
probabilities are retained, then every R2L/U2R score-scaling pair is evaluated
with the same pair applied to all requested seeds.

Default fixed families:
  - parameter-matched Conv1D
  - compact Conv2D
  - compact Transformer
  - standard XGBoost
  - cost-sensitive XGBoost

Default search size:
  3 neural families * 8 betas * 8 focal gammas * 3 seeds = 576 fits
  2 XGBoost families * 3 seeds                              =   6 fits
  total                                                     = 582 fits

The 40 x 40 score grid does not retrain a model.  It is evaluated from saved
validation probabilities with chunked NumPy confusion matrices.  Rankings are
computed separately for every family and report mean +/- sample standard
deviation, including R2L/U2R precision, recall, and F1.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

import run_no_ctgan_model_ablation_4gpu as core


DEFAULT_BETAS = [0.90, 0.95, 0.975, 0.99, 0.995, 0.999, 0.9995, 0.9999]
DEFAULT_FOCAL_GAMMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
NEURAL_FAMILIES = ("conv1d", "conv2d", "transformer")
XGBOOST_FAMILIES = ("xgboost_standard", "xgboost_cost_sensitive")
ALL_FAMILIES = (*NEURAL_FAMILIES, *XGBOOST_FAMILIES)

FAMILY_DISPLAY_NAMES = {
    "conv1d": "Conv1D",
    "conv2d": "Conv2D",
    "transformer": "Transformer",
    "xgboost_standard": "XGBoost standard",
    "xgboost_cost_sensitive": "XGBoost cost-sensitive",
}

FIXED_BACKBONES: Dict[str, Dict[str, Any]] = {
    "conv1d": {
        "groups": 1,
        "base_filters": 64,
        "dense_units": 48,
        "dropout1": 0.25,
        "dropout2": 0.30,
        "batch_norm": True,
        "residual": True,
        "expected_parameters": 109_797,
    },
    "conv2d": {
        "groups": 1,
        "base_filters": 64,
        "dense_units": 256,
        "dropout1": 0.25,
        "dropout2": 0.30,
        "batch_norm": True,
        "residual": True,
        "expected_parameters": 109_381,
    },
    "transformer": {
        "d_model": 64,
        "num_heads": 4,
        "blocks": 2,
        "ff_dim": 128,
        "dense_units": 512,
        "dropout": 0.10,
        "head_dropout": 0.30,
        "expected_parameters": 110_661,
    },
    "xgboost": {
        "objective": "multi:softprob",
        "num_class": 5,
        "n_estimators": 1_000,
        "early_stopping_rounds": 50,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "min_child_weight": 1.0,
        "reg_lambda": 1.0,
        "split_gamma": 0.0,
        "tree_method": "hist",
    },
}

TABLE_METRICS = [
    "accuracy",
    "mcc",
    "macro_f1",
    "macro_recall",
    "minimum_minority_recall",
    "rare_f1",
    "r2l_precision",
    "r2l_recall",
    "r2l_f1",
    "u2r_precision",
    "u2r_recall",
    "u2r_f1",
]


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def build_training_configurations(
    betas: Sequence[float],
    focal_gammas: Sequence[float],
) -> List[Dict[str, Any]]:
    configurations: List[Dict[str, Any]] = []
    for family in NEURAL_FAMILIES:
        for beta in betas:
            for focal_gamma in focal_gammas:
                payload = {
                    "family": family,
                    "beta": float(beta),
                    "focal_gamma": float(focal_gamma),
                    "class_weighting": "effective_number",
                    "minority_batches": True,
                    "backbone": FIXED_BACKBONES[family],
                }
                key = stable_hash(payload, 10)
                configurations.append(
                    {
                        **payload,
                        "training_id": f"{family}_{key}",
                        "configuration_key": key,
                        "loss_description": (
                            "class_balanced_cross_entropy"
                            if np.isclose(focal_gamma, 0.0)
                            else "class_balanced_focal"
                        ),
                    }
                )

    for family, weighting in (
        ("xgboost_standard", "none"),
        ("xgboost_cost_sensitive", "balanced"),
    ):
        payload = {
            "family": family,
            "beta": None,
            "focal_gamma": None,
            "class_weighting": weighting,
            "minority_batches": False,
            "backbone": FIXED_BACKBONES["xgboost"],
        }
        key = stable_hash(payload, 10)
        configurations.append(
            {
                **payload,
                "training_id": f"{family}_{key}",
                "configuration_key": key,
                "loss_description": "xgboost_multiclass_logloss",
            }
        )
    return configurations


def atomic_csv(
    path: Path,
    frame: pd.DataFrame,
    *,
    compression: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression=compression)
    os.replace(temporary, path)


def metrics_from_confusions(confusions: np.ndarray) -> pd.DataFrame:
    """Calculate all reported metrics from P x 5 x 5 confusion matrices."""
    matrices = np.asarray(confusions, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (5, 5):
        raise ValueError(f"Expected confusion shape (P, 5, 5), got {matrices.shape}.")

    true_totals = matrices.sum(axis=2)
    predicted_totals = matrices.sum(axis=1)
    true_positives = np.diagonal(matrices, axis1=1, axis2=2)
    totals = matrices.sum(axis=(1, 2))
    correct = true_positives.sum(axis=1)

    precision = np.divide(
        true_positives,
        predicted_totals,
        out=np.zeros_like(true_positives),
        where=predicted_totals != 0,
    )
    recall = np.divide(
        true_positives,
        true_totals,
        out=np.zeros_like(true_positives),
        where=true_totals != 0,
    )
    f1 = np.divide(
        2.0 * true_positives,
        true_totals + predicted_totals,
        out=np.zeros_like(true_positives),
        where=(true_totals + predicted_totals) != 0,
    )
    accuracy = np.divide(
        correct,
        totals,
        out=np.zeros_like(correct),
        where=totals != 0,
    )

    mcc_numerator = correct * totals - np.sum(true_totals * predicted_totals, axis=1)
    mcc_denominator = np.sqrt(
        (totals**2 - np.sum(predicted_totals**2, axis=1))
        * (totals**2 - np.sum(true_totals**2, axis=1))
    )
    mcc = np.divide(
        mcc_numerator,
        mcc_denominator,
        out=np.zeros_like(mcc_numerator),
        where=mcc_denominator != 0,
    )

    return pd.DataFrame(
        {
            "accuracy": accuracy,
            "mcc": mcc,
            "macro_f1": f1.mean(axis=1),
            "macro_recall": recall.mean(axis=1),
            "rare_f1": f1[:, [2, 3]].mean(axis=1),
            "minimum_minority_recall": recall[:, [2, 3]].min(axis=1),
            "r2l_precision": precision[:, 2],
            "r2l_recall": recall[:, 2],
            "r2l_f1": f1[:, 2],
            "u2r_precision": precision[:, 3],
            "u2r_recall": recall[:, 3],
            "u2r_f1": f1[:, 3],
        }
    )


def score_pair_confusions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    pair_r2l: np.ndarray,
    pair_u2r: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    """Evaluate all score pairs in sample chunks without an sklearn inner loop."""
    y_true = np.asarray(labels, dtype=np.int64)
    y_proba = np.asarray(probabilities, dtype=np.float64)
    r2l_coefficients = np.asarray(pair_r2l, dtype=np.float64)
    u2r_coefficients = np.asarray(pair_u2r, dtype=np.float64)
    if y_proba.shape != (len(y_true), 5):
        raise ValueError(
            f"Probability/label mismatch: {y_proba.shape} versus {len(y_true)}."
        )
    if not np.isfinite(y_proba).all():
        raise ValueError("Validation probabilities contain a non-finite value.")
    if (
        r2l_coefficients.shape != u2r_coefficients.shape
        or r2l_coefficients.ndim != 1
        or np.any(r2l_coefficients <= 0.0)
        or np.any(u2r_coefficients <= 0.0)
    ):
        raise ValueError("Score-pair arrays must be aligned and strictly positive.")

    pair_count = len(r2l_coefficients)
    pair_offsets = (25 * np.arange(pair_count, dtype=np.int64))[None, :]
    confusion_counts = np.zeros((pair_count, 25), dtype=np.int64)
    majority_classes = np.asarray([0, 1, 4], dtype=np.int8)

    for start in range(0, len(y_true), chunk_size):
        stop = min(len(y_true), start + chunk_size)
        chunk_probabilities = y_proba[start:stop]
        chunk_labels = y_true[start:stop]
        majority_probabilities = chunk_probabilities[:, [0, 1, 4]]
        majority_choice = np.argmax(majority_probabilities, axis=1)
        base_predictions = majority_classes[majority_choice]
        base_scores = majority_probabilities[
            np.arange(len(chunk_probabilities)), majority_choice
        ]

        predictions = np.broadcast_to(
            base_predictions[:, None],
            (len(chunk_probabilities), pair_count),
        ).copy()
        best_scores = np.broadcast_to(
            base_scores[:, None],
            (len(chunk_probabilities), pair_count),
        ).copy()

        r2l_scores = chunk_probabilities[:, 2, None] / r2l_coefficients[None, :]
        r2l_wins = (r2l_scores > best_scores) | (
            (r2l_scores == best_scores) & (predictions > 2)
        )
        np.copyto(best_scores, r2l_scores, where=r2l_wins)
        predictions[r2l_wins] = 2

        u2r_scores = chunk_probabilities[:, 3, None] / u2r_coefficients[None, :]
        u2r_wins = (u2r_scores > best_scores) | (
            (u2r_scores == best_scores) & (predictions > 3)
        )
        predictions[u2r_wins] = 3

        codes = chunk_labels[:, None] * 5 + predictions.astype(np.int64)
        linear_codes = codes + pair_offsets
        confusion_counts += np.bincount(
            linear_codes.ravel(),
            minlength=pair_count * 25,
        ).reshape(pair_count, 25)

    return confusion_counts.reshape(pair_count, 5, 5)


def train_neural_configuration(
    configuration: Dict[str, Any],
    fold: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, Dict[str, Any]]:
    import tensorflow as tf

    from cnn_gan_foc import ClassBalancedFocalLoss  # type: ignore
    from cnn_opt import BalancedBatchSequence, ValF1Callback, build_opt_cnn  # type: ignore
    from cnn_opt_1d_4gpu import (  # type: ignore
        build_opt_cnn_1d,
        build_vanilla_transformer,
    )

    visible_gpus = tf.config.list_physical_devices("GPU")
    if not args.allow_cpu and len(visible_gpus) != 1:
        raise RuntimeError(
            "Neural worker expected exactly one visible GPU, but TensorFlow "
            f"sees {len(visible_gpus)}: {visible_gpus}."
        )
    for device in visible_gpus:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass

    deterministic_ops_enabled = False
    if args.deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
            deterministic_ops_enabled = True
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "Deterministic TensorFlow operations were requested but could "
                "not be enabled. Rerun without --deterministic-ops."
            ) from error

    seed = int(args.worker_seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    family = configuration["family"]
    y_train = np.asarray(fold["y_train"], dtype=np.int64)
    y_val = np.asarray(fold["y_val"], dtype=np.int64)
    X_train_flat = np.asarray(fold["X_train"], dtype=np.float32)
    X_val_flat = np.asarray(fold["X_val"], dtype=np.float32)

    if family == "conv2d":
        X_train = X_train_flat.reshape(-1, 11, 11, 1)
        X_val = X_val_flat.reshape(-1, 11, 11, 1)
    else:
        X_train = X_train_flat.reshape(-1, 121, 1)
        X_val = X_val_flat.reshape(-1, 121, 1)

    alpha, alpha_counts = core.effective_number_alpha(
        y_train,
        beta=float(configuration["beta"]),
        num_classes=5,
    )
    loss = ClassBalancedFocalLoss(
        alpha=alpha,
        gamma=float(configuration["focal_gamma"]),
    )

    if family == "conv1d":
        backbone = FIXED_BACKBONES["conv1d"]
        model = build_opt_cnn_1d(
            loss=loss,
            groups=backbone["groups"],
            base_filters=backbone["base_filters"],
            dense_units=backbone["dense_units"],
            dropout1=backbone["dropout1"],
            dropout2=backbone["dropout2"],
            use_batch_norm=backbone["batch_norm"],
            use_residual=backbone["residual"],
        )
    elif family == "conv2d":
        backbone = FIXED_BACKBONES["conv2d"]
        model = build_opt_cnn(
            loss=loss,
            groups=backbone["groups"],
            base_filters=backbone["base_filters"],
            dense_units=backbone["dense_units"],
            dropout1=backbone["dropout1"],
            dropout2=backbone["dropout2"],
            use_batch_norm=backbone["batch_norm"],
            use_residual=backbone["residual"],
        )
    elif family == "transformer":
        backbone = FIXED_BACKBONES["transformer"]
        model = build_vanilla_transformer(
            loss=loss,
            d_model=backbone["d_model"],
            num_heads=backbone["num_heads"],
            num_blocks=backbone["blocks"],
            ff_dim=backbone["ff_dim"],
            dense_units=backbone["dense_units"],
            transformer_dropout=backbone["dropout"],
            head_dropout=backbone["head_dropout"],
        )
    else:
        raise ValueError(f"Unsupported neural family: {family}")

    model_parameters = int(model.count_params())
    expected_parameters = int(FIXED_BACKBONES[family]["expected_parameters"])
    if model_parameters != expected_parameters:
        raise ValueError(
            f"{family} parameter count changed: expected {expected_parameters}, "
            f"got {model_parameters}."
        )

    training_data = BalancedBatchSequence(
        X_train,
        y_train,
        batch_size=args.batch_size,
        minority_per_batch=args.minority_per_batch,
        seed=seed,
    )
    with tempfile.TemporaryDirectory(prefix=f"{args.worker_run_name}_") as temp_dir:
        weights_path = Path(temp_dir) / "best.weights.h5"
        callbacks: List[tf.keras.callbacks.Callback] = [
            ValF1Callback(X_val, y_val, args.batch_size),
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
                patience=args.early_stopping_patience,
                restore_best_weights=False,
            ),
        ]
        history = model.fit(
            training_data,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        if not weights_path.exists():
            raise RuntimeError("Best validation checkpoint was not created.")
        model.load_weights(weights_path)

    validation_probabilities = np.asarray(
        model.predict(X_val, batch_size=args.batch_size, verbose=0),
        dtype=np.float32,
    )
    val_history = history.history.get("val_macro_f1", [])
    metadata = {
        "optimizer": "keras_adam_defaults",
        "model_parameters": model_parameters,
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "best_epoch": int(np.argmax(val_history)) + 1 if val_history else None,
        "checkpoint_metric": "validation_macro_f1_raw_argmax",
        "early_stopping_patience": int(args.early_stopping_patience),
        "batch_size": int(args.batch_size),
        "minority_per_batch": int(args.minority_per_batch),
        "beta": float(configuration["beta"]),
        "focal_gamma": float(configuration["focal_gamma"]),
        "alpha": np.asarray(alpha, dtype=float).tolist(),
        "alpha_counts": np.asarray(alpha_counts, dtype=int).tolist(),
        "deterministic_ops_enabled": deterministic_ops_enabled,
        "deterministic_ops_requested": bool(args.deterministic_ops),
        "tensorflow_visible_gpu_count": len(visible_gpus),
        "tensorflow_version": core.package_version("tensorflow"),
        "keras_version": core.package_version("keras"),
    }
    tf.keras.backend.clear_session()
    return validation_probabilities, metadata


def train_xgboost_configuration(
    configuration: Dict[str, Any],
    fold: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, Dict[str, Any]]:
    adapter = {
        "family": "xgboost",
        "class_weighting": configuration["class_weighting"],
    }
    _model, validation_probabilities, metadata = core.train_xgboost_model(
        adapter,
        fold,
        args,
    )
    return np.asarray(validation_probabilities, dtype=np.float32), metadata


def configuration_from_worker_args(args: argparse.Namespace) -> Dict[str, Any]:
    family = args.worker_family
    if family in NEURAL_FAMILIES:
        beta: float | None = float(args.worker_beta)
        focal_gamma: float | None = float(args.worker_focal_gamma)
        weighting = "effective_number"
        loss_description = (
            "class_balanced_cross_entropy"
            if np.isclose(focal_gamma, 0.0)
            else "class_balanced_focal"
        )
    else:
        beta = None
        focal_gamma = None
        weighting = "balanced" if family == "xgboost_cost_sensitive" else "none"
        loss_description = "xgboost_multiclass_logloss"
    return {
        "training_id": args.worker_training_id,
        "configuration_key": args.config_key,
        "family": family,
        "beta": beta,
        "focal_gamma": focal_gamma,
        "class_weighting": weighting,
        "minority_batches": family in NEURAL_FAMILIES,
        "loss_description": loss_description,
        "backbone": (
            FIXED_BACKBONES[family]
            if family in NEURAL_FAMILIES
            else FIXED_BACKBONES["xgboost"]
        ),
    }


def run_training_worker(args: argparse.Namespace, repo_root: Path) -> None:
    configuration = configuration_from_worker_args(args)
    start = time.perf_counter()
    fold = core.prepare_fold_data(repo_root, int(args.worker_seed), args.val_split)

    if configuration["family"] in NEURAL_FAMILIES:
        validation_probabilities, training_metadata = train_neural_configuration(
            configuration,
            fold,
            args,
        )
    else:
        validation_probabilities, training_metadata = train_xgboost_configuration(
            configuration,
            fold,
            args,
        )

    validation_labels = np.asarray(fold["y_val"], dtype=np.int64)
    raw_predictions = np.argmax(validation_probabilities, axis=1).astype(np.int64)
    raw_metrics = core.calculate_metrics(validation_labels, raw_predictions)
    result_path = Path(args.worker_result_path)
    prediction_path = result_path.with_name(
        f"{args.worker_run_name}_{args.worker_attempt_id}_validation.npz"
    )
    core.atomic_npz(
        prediction_path,
        train_indices=np.asarray(fold["train_indices"], dtype=np.int64),
        validation_indices=np.asarray(fold["val_indices"], dtype=np.int64),
        validation_labels=validation_labels,
        validation_probabilities=np.asarray(validation_probabilities, dtype=np.float32),
        raw_validation_predictions=raw_predictions,
    )

    result = {
        "schema_version": 1,
        "experiment_key": args.experiment_key,
        "configuration_key": args.config_key,
        "attempt_id": args.worker_attempt_id,
        "run_name": args.worker_run_name,
        "training_id": configuration["training_id"],
        "family": configuration["family"],
        "seed": int(args.worker_seed),
        "beta": configuration["beta"],
        "focal_gamma": configuration["focal_gamma"],
        "loss_description": configuration["loss_description"],
        "class_weighting": configuration["class_weighting"],
        "minority_batches": configuration["minority_batches"],
        "backbone": configuration["backbone"],
        "no_ctgan": True,
        "synthetic_rows": 0,
        "dataset": "KDDTrain+ only",
        "evaluation_partition": "real held-out KDDTrain+ validation fold",
        "kddtest_accessed": False,
        "split_before_preprocessing": True,
        "val_split": float(args.val_split),
        "train_indices_sha256": fold["train_indices_sha256"],
        "validation_indices_sha256": fold["val_indices_sha256"],
        "train_counts": fold["train_counts"].astype(int).tolist(),
        "validation_counts": fold["validation_counts"].astype(int).tolist(),
        "feature_count": 121,
        "feature_order_sha256": fold["feature_order_sha256"],
        "scaler_state_sha256": fold["scaler_state_sha256"],
        "raw_validation_metrics": raw_metrics,
        "prediction_path": str(prediction_path),
        "prediction_sha256": core.sha256_file(prediction_path),
        "runtime_seconds": time.perf_counter() - start,
        "assigned_gpu": os.environ.get("EXPERIMENT_GPU_ID", ""),
        "training_metadata": training_metadata,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    core.atomic_json(result_path, result)
    print(
        f"Completed {args.worker_run_name}: "
        f"validation macro-F1={raw_metrics['macro_f1']:.6f}",
        flush=True,
    )


def validation_artifact_is_complete(path: Path, result: Dict[str, Any]) -> bool:
    if not path.is_file() or core.sha256_file(path) != result.get("prediction_sha256"):
        return False
    required = {
        "train_indices",
        "validation_indices",
        "validation_labels",
        "validation_probabilities",
        "raw_validation_predictions",
    }
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if not required.issubset(artifact.files):
                return False
            arrays = {name: np.asarray(artifact[name]) for name in required}
    except (OSError, ValueError, KeyError):
        return False

    train_count = int(sum(result["train_counts"]))
    validation_count = int(sum(result["validation_counts"]))
    if arrays["train_indices"].shape != (train_count,):
        return False
    if arrays["validation_indices"].shape != (validation_count,):
        return False
    if arrays["validation_labels"].shape != (validation_count,):
        return False
    if arrays["validation_probabilities"].shape != (validation_count, 5):
        return False
    if arrays["raw_validation_predictions"].shape != (validation_count,):
        return False
    if (
        core.sha256_indices(arrays["train_indices"]) != result["train_indices_sha256"]
        or core.sha256_indices(arrays["validation_indices"])
        != result["validation_indices_sha256"]
    ):
        return False
    if not np.isfinite(arrays["validation_probabilities"]).all():
        return False
    if (
        np.bincount(arrays["validation_labels"], minlength=5).astype(int).tolist()
        != result["validation_counts"]
    ):
        return False
    if np.any(
        (arrays["raw_validation_predictions"] < 0)
        | (arrays["raw_validation_predictions"] >= 5)
    ):
        return False
    if not core.metrics_match_predictions(
        result["raw_validation_metrics"],
        arrays["validation_labels"],
        arrays["raw_validation_predictions"],
    ):
        return False
    return True


def result_is_complete(
    result_path: Path,
    plan: Dict[str, Any],
    expected_attempt_id: str | None = None,
) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = core.read_json(result_path)
        expected = {
            "schema_version": 1,
            "experiment_key": plan["experiment_key"],
            "configuration_key": plan["configuration_key"],
            "run_name": plan["run_name"],
            "training_id": plan["training_id"],
            "family": plan["family"],
            "seed": int(plan["seed"]),
            "beta": plan["beta"],
            "focal_gamma": plan["focal_gamma"],
            "class_weighting": plan["class_weighting"],
            "minority_batches": plan["minority_batches"],
            "no_ctgan": True,
            "synthetic_rows": 0,
            "kddtest_accessed": False,
            "train_indices_sha256": plan["train_indices_sha256"],
            "validation_indices_sha256": plan["validation_indices_sha256"],
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        if (
            expected_attempt_id is not None
            and result.get("attempt_id") != expected_attempt_id
        ):
            return False
        if not core.metrics_are_complete(result.get("raw_validation_metrics")):
            return False
        artifact_path = Path(str(result.get("prediction_path", "")))
        return validation_artifact_is_complete(artifact_path, result)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_worker_command(
    script_path: Path,
    plan: Dict[str, Any],
    args: argparse.Namespace,
    attempt_id: str,
) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--worker-mode",
        "train",
        "--worker-training-id",
        str(plan["training_id"]),
        "--worker-family",
        str(plan["family"]),
        "--worker-seed",
        str(plan["seed"]),
        "--worker-beta",
        str(plan["beta"] if plan["beta"] is not None else "nan"),
        "--worker-focal-gamma",
        str(plan["focal_gamma"] if plan["focal_gamma"] is not None else "nan"),
        "--worker-run-name",
        str(plan["run_name"]),
        "--worker-result-path",
        str(plan["result_path"]),
        "--worker-attempt-id",
        attempt_id,
        "--experiment-key",
        str(plan["experiment_key"]),
        "--config-key",
        str(plan["configuration_key"]),
        "--val-split",
        str(args.val_split),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--minority-per-batch",
        str(args.minority_per_batch),
        "--xgb-device",
        str(args.xgb_device),
        "--xgb-n-estimators",
        str(args.xgb_n_estimators),
        "--xgb-early-stopping-rounds",
        str(args.xgb_early_stopping_rounds),
        "--xgb-max-depth",
        str(args.xgb_max_depth),
        "--xgb-learning-rate",
        str(args.xgb_learning_rate),
        "--xgb-subsample",
        str(args.xgb_subsample),
        "--xgb-colsample-bytree",
        str(args.xgb_colsample_bytree),
        "--xgb-min-child-weight",
        str(args.xgb_min_child_weight),
        "--xgb-reg-lambda",
        str(args.xgb_reg_lambda),
        "--xgb-gamma",
        str(args.xgb_gamma),
        "--xgb-n-jobs",
        str(args.xgb_n_jobs),
        "--xgb-verbose-eval",
        str(args.xgb_verbose_eval),
    ]
    if args.allow_cpu:
        command.append("--allow-cpu")
    if args.deterministic_ops:
        command.append("--deterministic-ops")
    return command


def score_result_is_complete(
    csv_path: Path,
    metadata_path: Path,
    score_key: str,
    expected_rows: int,
) -> bool:
    if not csv_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = core.read_json(metadata_path)
        return bool(
            metadata.get("schema_version") == 1
            and metadata.get("score_key") == score_key
            and metadata.get("rows") == expected_rows
            and core.sha256_file(csv_path) == metadata.get("csv_sha256")
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def score_training_configuration(task: Dict[str, Any]) -> Dict[str, Any]:
    configuration = task["configuration"]
    csv_path = Path(task["csv_path"])
    metadata_path = Path(task["metadata_path"])
    score_key = task["score_key"]
    expected_rows = len(task["coefficient_values"]) ** 2
    if score_result_is_complete(
        csv_path,
        metadata_path,
        score_key,
        expected_rows,
    ):
        return {"training_id": configuration["training_id"], "skipped": True}

    r2l_values = np.asarray(task["coefficient_values"], dtype=np.float64)
    u2r_values = np.asarray(task["coefficient_values"], dtype=np.float64)
    r2l_grid, u2r_grid = np.meshgrid(r2l_values, u2r_values, indexing="ij")
    pair_r2l = r2l_grid.ravel()
    pair_u2r = u2r_grid.ravel()
    per_seed_metrics: List[np.ndarray] = []
    observed_seeds: List[int] = []
    input_artifacts: List[Dict[str, Any]] = []

    for result_path_raw in task["result_paths"]:
        result_path = Path(result_path_raw)
        result = core.read_json(result_path)
        artifact_path = Path(result["prediction_path"])
        with np.load(artifact_path, allow_pickle=False) as artifact:
            labels = np.asarray(artifact["validation_labels"], dtype=np.int64)
            probabilities = np.asarray(
                artifact["validation_probabilities"],
                dtype=np.float64,
            )
        confusions = score_pair_confusions(
            labels,
            probabilities,
            pair_r2l,
            pair_u2r,
            int(task["score_chunk_size"]),
        )
        metrics = metrics_from_confusions(confusions)
        per_seed_metrics.append(metrics[core.METRICS].to_numpy(dtype=np.float64))
        observed_seeds.append(int(result["seed"]))
        input_artifacts.append(
            {
                "seed": int(result["seed"]),
                "result_path": str(result_path),
                "result_sha256": core.sha256_file(result_path),
                "prediction_path": str(artifact_path),
                "prediction_sha256": result["prediction_sha256"],
            }
        )

    expected_seeds = [int(seed) for seed in task["seeds"]]
    if sorted(observed_seeds) != sorted(expected_seeds):
        raise ValueError(
            f"{configuration['training_id']} has seeds {observed_seeds}; "
            f"expected {expected_seeds}."
        )
    order = np.argsort(observed_seeds)
    stacked = np.stack([per_seed_metrics[index] for index in order], axis=0)
    means = stacked.mean(axis=0)
    standard_deviations = (
        stacked.std(axis=0, ddof=1) if len(observed_seeds) > 1 else np.zeros_like(means)
    )

    output = pd.DataFrame(
        {
            "family": configuration["family"],
            "display_name": FAMILY_DISPLAY_NAMES[configuration["family"]],
            "training_id": configuration["training_id"],
            "configuration_key": configuration["configuration_key"],
            "loss_description": configuration["loss_description"],
            "class_weighting": configuration["class_weighting"],
            "beta": configuration["beta"],
            "focal_gamma": configuration["focal_gamma"],
            "r2l_score_coefficient": pair_r2l,
            "u2r_score_coefficient": pair_u2r,
            "is_raw_argmax": np.isclose(pair_r2l, 1.0) & np.isclose(pair_u2r, 1.0),
            "scaling_log_distance": np.abs(np.log(pair_r2l)) + np.abs(np.log(pair_u2r)),
            "runs": len(observed_seeds),
            "seeds": ",".join(str(seed) for seed in sorted(observed_seeds)),
        }
    )
    for metric_index, metric in enumerate(core.METRICS):
        output[f"{metric}_mean"] = means[:, metric_index]
        output[f"{metric}_std"] = standard_deviations[:, metric_index]

    atomic_csv(csv_path, output, compression="gzip")
    metadata = {
        "schema_version": 1,
        "score_key": score_key,
        "training_id": configuration["training_id"],
        "family": configuration["family"],
        "rows": len(output),
        "seeds": sorted(observed_seeds),
        "input_artifacts": input_artifacts,
        "csv_path": str(csv_path),
        "csv_sha256": core.sha256_file(csv_path),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    core.atomic_json(metadata_path, metadata)
    return {"training_id": configuration["training_id"], "skipped": False}


def rank_joint_configurations(
    combined: pd.DataFrame,
    macro_f1_retention: float,
) -> pd.DataFrame:
    ranked_parts: List[pd.DataFrame] = []
    for family in ALL_FAMILIES:
        family_frame = combined[combined["family"] == family].copy()
        if family_frame.empty:
            raise ValueError(f"No score rows were produced for {family}.")
        best_macro_f1 = float(family_frame["macro_f1_mean"].max())
        macro_f1_floor = float(best_macro_f1 * macro_f1_retention)
        family_frame["family_best_macro_f1_mean"] = best_macro_f1
        family_frame["macro_f1_eligibility_floor"] = macro_f1_floor
        family_frame["eligible"] = (
            family_frame["macro_f1_mean"] >= macro_f1_floor - 1e-15
        )
        family_frame["beta_sort"] = family_frame["beta"].fillna(-1.0)
        family_frame["focal_gamma_sort"] = family_frame["focal_gamma"].fillna(-1.0)
        family_frame = family_frame.sort_values(
            by=[
                "eligible",
                "minimum_minority_recall_mean",
                "rare_f1_mean",
                "macro_f1_mean",
                "minimum_minority_recall_std",
                "rare_f1_std",
                "macro_f1_std",
                "scaling_log_distance",
                "beta_sort",
                "focal_gamma_sort",
                "r2l_score_coefficient",
                "u2r_score_coefficient",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ],
            kind="mergesort",
        ).reset_index(drop=True)
        family_frame["rank_within_family"] = np.arange(1, len(family_frame) + 1)
        family_frame["eligible_rank"] = pd.array(
            np.where(
                family_frame["eligible"],
                np.arange(1, int(family_frame["eligible"].sum()) + 1).tolist()
                + [0] * int((~family_frame["eligible"]).sum()),
                0,
            ),
            dtype="Int64",
        )
        family_frame.loc[~family_frame["eligible"], "eligible_rank"] = pd.NA
        ranked_parts.append(
            family_frame.drop(columns=["beta_sort", "focal_gamma_sort"])
        )
    return pd.concat(ranked_parts, ignore_index=True)


def format_mean_std(mean: float, std: float) -> str:
    return f"{100.0 * float(mean):.2f}% +/- {100.0 * float(std):.2f}%"


def formatted_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "display_name",
        "policy",
        "rank_within_family",
        "runs",
        "seeds",
        "loss_description",
        "beta",
        "focal_gamma",
        "r2l_score_coefficient",
        "u2r_score_coefficient",
    ]
    output = frame[[column for column in columns if column in frame.columns]].copy()
    labels = {
        "accuracy": "Accuracy",
        "mcc": "MCC",
        "macro_f1": "Macro-F1",
        "macro_recall": "Macro Recall",
        "minimum_minority_recall": "Min Minority Recall",
        "rare_f1": "Rare F1",
        "r2l_precision": "R2L Precision",
        "r2l_recall": "R2L Recall",
        "r2l_f1": "R2L F1",
        "u2r_precision": "U2R Precision",
        "u2r_recall": "U2R Recall",
        "u2r_f1": "U2R F1",
    }
    for metric in TABLE_METRICS:
        output[labels[metric]] = [
            format_mean_std(mean, std)
            for mean, std in zip(
                frame[f"{metric}_mean"],
                frame[f"{metric}_std"],
                strict=True,
            )
        ]
    return output


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument(
        "--focal-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_FOCAL_GAMMAS,
    )
    parser.add_argument("--coefficient-min", type=float, default=0.05)
    parser.add_argument("--coefficient-max", type=float, default=2.00)
    parser.add_argument("--coefficient-step", type=float, default=0.05)
    parser.add_argument("--macro-f1-retention", type=float, default=0.95)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument("--score-workers", type=int, default=4)
    parser.add_argument("--score-chunk-size", type=int, default=512)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--name-prefix", default="fixed_backbone_imbalance_sweep")

    xgb = FIXED_BACKBONES["xgboost"]
    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--xgb-n-estimators", type=int, default=xgb["n_estimators"])
    parser.add_argument(
        "--xgb-early-stopping-rounds",
        type=int,
        default=xgb["early_stopping_rounds"],
    )
    parser.add_argument("--xgb-max-depth", type=int, default=xgb["max_depth"])
    parser.add_argument("--xgb-learning-rate", type=float, default=xgb["learning_rate"])
    parser.add_argument("--xgb-subsample", type=float, default=xgb["subsample"])
    parser.add_argument(
        "--xgb-colsample-bytree",
        type=float,
        default=xgb["colsample_bytree"],
    )
    parser.add_argument(
        "--xgb-min-child-weight",
        type=float,
        default=xgb["min_child_weight"],
    )
    parser.add_argument("--xgb-reg-lambda", type=float, default=xgb["reg_lambda"])
    parser.add_argument("--xgb-gamma", type=float, default=xgb["split_gamma"])
    parser.add_argument("--xgb-n-jobs", type=int, default=4)
    parser.add_argument("--xgb-verbose-eval", type=int, default=50)

    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument(
        "--deterministic-ops",
        action="store_true",
        help=(
            "Request TensorFlow deterministic kernels. This is opt-in because "
            "some GPU/TensorFlow builds reject unsupported deterministic ops."
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--recompute-scores", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-all-commands", action="store_true")

    parser.add_argument(
        "--worker-mode",
        choices=["none", "train"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-training-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-family", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-beta", type=float, default=float("nan"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-focal-gamma",
        type=float,
        default=float("nan"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-run-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-attempt-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--config-key", default="", help=argparse.SUPPRESS)


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds cannot contain negative values.")
    if not args.betas or len(args.betas) != len(set(args.betas)):
        parser.error("--betas must contain unique values.")
    if any(not 0.0 < beta < 1.0 for beta in args.betas):
        parser.error("Every beta must be strictly between zero and one.")
    if not args.focal_gammas or len(args.focal_gammas) != len(set(args.focal_gammas)):
        parser.error("--focal-gammas must contain unique values.")
    if any(not np.isfinite(gamma) or gamma < 0.0 for gamma in args.focal_gammas):
        parser.error("Focal gammas must be finite and nonnegative.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between zero and one.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")
    if args.early_stopping_patience < 0 or args.minority_per_batch <= 0:
        parser.error(
            "Patience cannot be negative; minority-per-batch must be positive."
        )
    if not 0.0 < args.macro_f1_retention <= 1.0:
        parser.error("--macro-f1-retention must be in (0, 1].")
    if args.score_workers <= 0 or args.score_chunk_size <= 0 or args.top_n <= 0:
        parser.error("Score workers, score chunk size, and top-n must be positive.")
    if (
        not np.isfinite(args.coefficient_min)
        or not np.isfinite(args.coefficient_max)
        or not np.isfinite(args.coefficient_step)
        or args.coefficient_min <= 0.0
        or args.coefficient_max < args.coefficient_min
        or args.coefficient_step <= 0.0
    ):
        parser.error("Score grid needs 0 < min <= max and a positive step.")
    try:
        coefficient_values = core.score_coefficient_values(
            args.coefficient_min,
            args.coefficient_max,
            args.coefficient_step,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if not any(np.isclose(value, 1.0) for value in coefficient_values):
        parser.error("The score grid must include coefficient 1.0.")
    if len(coefficient_values) ** 2 > 10_000:
        parser.error("The score grid exceeds the memory-safe limit of 10,000 pairs.")
    if args.xgb_n_estimators <= 0 or args.xgb_early_stopping_rounds < 0:
        parser.error("XGBoost estimators must be positive; patience nonnegative.")
    if args.xgb_max_depth <= 0 or args.xgb_learning_rate <= 0.0:
        parser.error("XGBoost depth and learning rate must be positive.")
    if not 0.0 < args.xgb_subsample <= 1.0:
        parser.error("--xgb-subsample must be in (0, 1].")
    if not 0.0 < args.xgb_colsample_bytree <= 1.0:
        parser.error("--xgb-colsample-bytree must be in (0, 1].")
    if args.xgb_min_child_weight < 0.0 or args.xgb_reg_lambda < 0.0:
        parser.error("XGBoost child weight and lambda cannot be negative.")
    if args.xgb_gamma < 0.0 or args.xgb_verbose_eval < 0:
        parser.error("XGBoost split gamma and verbose interval cannot be negative.")
    fixed_xgb = FIXED_BACKBONES["xgboost"]
    fixed_values = {
        "--xgb-n-estimators": (args.xgb_n_estimators, fixed_xgb["n_estimators"]),
        "--xgb-early-stopping-rounds": (
            args.xgb_early_stopping_rounds,
            fixed_xgb["early_stopping_rounds"],
        ),
        "--xgb-max-depth": (args.xgb_max_depth, fixed_xgb["max_depth"]),
        "--xgb-learning-rate": (
            args.xgb_learning_rate,
            fixed_xgb["learning_rate"],
        ),
        "--xgb-subsample": (args.xgb_subsample, fixed_xgb["subsample"]),
        "--xgb-colsample-bytree": (
            args.xgb_colsample_bytree,
            fixed_xgb["colsample_bytree"],
        ),
        "--xgb-min-child-weight": (
            args.xgb_min_child_weight,
            fixed_xgb["min_child_weight"],
        ),
        "--xgb-reg-lambda": (args.xgb_reg_lambda, fixed_xgb["reg_lambda"]),
        "--xgb-gamma": (args.xgb_gamma, fixed_xgb["split_gamma"]),
    }
    changed_fixed_values = [
        name
        for name, (actual, expected) in fixed_values.items()
        if not np.isclose(float(actual), float(expected))
    ]
    if changed_fixed_values:
        parser.error(
            "Backbone hyperparameters are fixed a priori in this experiment; "
            f"do not override {changed_fixed_values}."
        )


def experiment_settings(
    args: argparse.Namespace,
    coefficient_values: Sequence[float],
) -> Dict[str, Any]:
    return {
        "library_versions": {
            "tensorflow": core.package_version("tensorflow"),
            "keras": core.package_version("keras"),
            "xgboost": core.package_version("xgboost"),
            "scikit_learn": core.package_version("scikit-learn"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seeds": [int(seed) for seed in args.seeds],
        "betas": [float(beta) for beta in args.betas],
        "focal_gammas": [float(gamma) for gamma in args.focal_gammas],
        "score_coefficients": [float(value) for value in coefficient_values],
        "score_pair_count": len(coefficient_values) ** 2,
        "macro_f1_retention": float(args.macro_f1_retention),
        "val_split": float(args.val_split),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "early_stopping_patience": int(args.early_stopping_patience),
        "minority_per_batch": int(args.minority_per_batch),
        "deterministic_ops": bool(args.deterministic_ops),
        "fixed_backbones": FIXED_BACKBONES,
        "xgboost_runtime": {
            "device": args.xgb_device,
            "n_estimators": int(args.xgb_n_estimators),
            "early_stopping_rounds": int(args.xgb_early_stopping_rounds),
            "max_depth": int(args.xgb_max_depth),
            "learning_rate": float(args.xgb_learning_rate),
            "subsample": float(args.xgb_subsample),
            "colsample_bytree": float(args.xgb_colsample_bytree),
            "min_child_weight": float(args.xgb_min_child_weight),
            "reg_lambda": float(args.xgb_reg_lambda),
            "split_gamma": float(args.xgb_gamma),
            "n_jobs": int(args.xgb_n_jobs),
        },
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=("Fixed-backbone, three-seed beta/gamma and R2L/U2R score sweep.")
    )
    add_arguments(parser)
    args = parser.parse_args()
    validate_arguments(parser, args)

    if args.worker_mode == "train":
        required = {
            "--worker-training-id": args.worker_training_id,
            "--worker-family": args.worker_family,
            "--worker-run-name": args.worker_run_name,
            "--worker-result-path": args.worker_result_path,
            "--worker-attempt-id": args.worker_attempt_id,
            "--experiment-key": args.experiment_key,
            "--config-key": args.config_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"Worker invocation is missing: {missing}")
        if args.worker_family not in ALL_FAMILIES:
            parser.error(f"Unknown worker family: {args.worker_family}")
        if args.worker_family in NEURAL_FAMILIES and (
            not np.isfinite(args.worker_beta)
            or not np.isfinite(args.worker_focal_gamma)
        ):
            parser.error("Neural workers require finite beta and focal gamma.")
        run_training_worker(args, repo_root)
        return

    try:
        gpus = core.parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    prefix = args.name_prefix.strip() or "fixed_backbone_imbalance_sweep"
    if Path(prefix).name != prefix:
        parser.error("--name-prefix must be a filename-safe name, not a path.")

    train_path = repo_root / "data" / "KDDTrain+.txt"
    required_paths = [
        train_path,
        script_path,
        repo_root / "src" / "run_no_ctgan_model_ablation_4gpu.py",
        repo_root / "src" / "cnn_opt.py",
        repo_root / "src" / "cnn_opt_1d_4gpu.py",
        repo_root / "src" / "cnn_gan_foc.py",
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise SystemExit(f"Required files are missing: {missing_paths}")

    coefficient_values = core.score_coefficient_values(
        args.coefficient_min,
        args.coefficient_max,
        args.coefficient_step,
    )
    configurations = build_training_configurations(args.betas, args.focal_gammas)
    settings = experiment_settings(args, coefficient_values)
    source_fingerprint = core.fingerprint_files(required_paths)
    identity = {
        "schema_version": 1,
        "settings": settings,
        "training_configurations": configurations,
        "source_and_train_data_fingerprint": source_fingerprint,
    }
    experiment_key = stable_hash(identity, 12)

    raw_train = core.load_collapsed_nsl_kdd(train_path, is_train=True)
    labels = raw_train["class"].to_numpy(dtype=np.int64)
    fold_protocol: Dict[str, Dict[str, Any]] = {}
    for seed in args.seeds:
        train_indices, validation_indices = core.split_raw_indices(
            labels,
            int(seed),
            args.val_split,
        )
        fold_protocol[str(seed)] = {
            "train_indices_sha256": core.sha256_indices(train_indices),
            "validation_indices_sha256": core.sha256_indices(validation_indices),
            "train_counts": np.bincount(labels[train_indices], minlength=5)
            .astype(int)
            .tolist(),
            "validation_counts": np.bincount(labels[validation_indices], minlength=5)
            .astype(int)
            .tolist(),
        }

    results_dir = repo_root / "results"
    stem = f"{prefix}_{experiment_key}"
    run_dir = results_dir / f"{stem}_runs"
    log_dir = results_dir / f"{stem}_logs"
    score_dir = results_dir / f"{stem}_score_parts"
    plan_path = results_dir / f"{stem}_plan.csv"
    protocol_path = results_dir / f"{stem}_protocol.json"

    plans: List[Dict[str, Any]] = []
    for configuration in configurations:
        for seed in args.seeds:
            fold = fold_protocol[str(seed)]
            run_name = f"{stem}_{configuration['training_id']}_s{seed}"
            plans.append(
                {
                    **configuration,
                    "experiment_key": experiment_key,
                    "seed": int(seed),
                    "run_name": run_name,
                    "train_indices_sha256": fold["train_indices_sha256"],
                    "validation_indices_sha256": fold["validation_indices_sha256"],
                    "result_path": str(run_dir / f"{run_name}.json"),
                    "log_path": str(log_dir / f"{run_name}.log"),
                }
            )

    neural_configuration_count = (
        len(NEURAL_FAMILIES) * len(args.betas) * len(args.focal_gammas)
    )
    expected_configuration_count = neural_configuration_count + len(XGBOOST_FAMILIES)
    if len(configurations) != expected_configuration_count:
        raise RuntimeError("Training configuration construction is inconsistent.")
    if len(plans) != len(configurations) * len(args.seeds):
        raise RuntimeError("Training plan construction is inconsistent.")
    averaged_configuration_count = (
        neural_configuration_count + len(XGBOOST_FAMILIES)
    ) * len(coefficient_values) ** 2

    print("Fixed-backbone imbalance-control sweep")
    print(f"Experiment key: {experiment_key}")
    print(f"Seeds per training configuration: {args.seeds}")
    print(f"GPUs: {gpus}")
    print(
        f"Neural beta/gamma settings per family: {len(args.betas) * len(args.focal_gammas)}"
    )
    print(f"Unique training configurations: {len(configurations)}")
    print(f"Total seeded fits: {len(plans)}")
    print(f"Score pairs per training configuration: {len(coefficient_values) ** 2}")
    print(f"Averaged configurations ranked: {averaged_configuration_count}")
    print("Evaluation during sweep: real KDDTrain+ validation folds only")
    print("KDDTest+ accessed: NO")

    if args.dry_run:
        shown = plans if args.print_all_commands else plans[:8]
        for index, plan in enumerate(shown):
            command = build_worker_command(
                script_path,
                plan,
                args,
                attempt_id="dry_run",
            )
            print(f"[GPU {gpus[index % len(gpus)]}] {shlex.join(command)}")
        if len(shown) < len(plans):
            print(f"... {len(plans) - len(shown)} more training commands")
        print("Dry run complete; no files were written and nothing was trained.")
        return

    core.validate_runtime_dependencies(parser, args)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "experiment_key": experiment_key,
        "title": "Fixed-backbone seeded imbalance-control sweep",
        "source_and_train_data_fingerprint": source_fingerprint,
        "kddtrain_sha256": core.sha256_file(train_path),
        "kddtest_accessed": False,
        "no_ctgan": True,
        "synthetic_rows": 0,
        "split_protocol": (
            "raw KDDTrain+ stratified split before encoder/scaler fitting"
        ),
        "preprocessing_fit_data": "80 percent real training fold only",
        "evaluation_data": "20 percent real KDDTrain+ validation fold only",
        "ranking_scope": "separate ranking within each of five model families",
        "shared_seed_policy": (
            "each complete beta/gamma/score configuration uses the same score "
            "pair across every requested seed; metrics are calculated per "
            "seed and then averaged, never pooled across overlapping folds"
        ),
        "validation_reuse": (
            "the same seed-specific validation fold is used for neural "
            "checkpointing or XGBoost early stopping and subsequent score-policy "
            "ranking; it is development data, not final evaluation data"
        ),
        "ranking_rule": (
            "eligible when mean Macro-F1 is at least the configured fraction "
            "of the best observed mean Macro-F1 in that family; then maximize "
            "mean minimum minority recall, rare F1, and Macro-F1, followed by "
            "lower sample standard deviations and smaller scaling distance"
        ),
        "gamma_zero_interpretation": (
            "gamma=0 is class-balanced cross-entropy, not active focal focusing"
        ),
        "minority_precision_reported": True,
        "settings": settings,
        "folds": fold_protocol,
        "training_configurations": configurations,
        "unique_training_configurations": len(configurations),
        "total_seeded_fits": len(plans),
        "averaged_ranked_configurations": averaged_configuration_count,
    }
    core.atomic_json(protocol_path, protocol)
    pd.DataFrame(plans).to_csv(plan_path, index=False)
    print(f"Protocol: {protocol_path}")
    print(f"Plan: {plan_path}")

    task_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    for plan in plans:
        task_queue.put(plan)
    print_lock = threading.Lock()
    data_lock = threading.Lock()
    stop_event = threading.Event()
    statuses: Dict[str, str] = {}
    assigned_gpus: Dict[str, str] = {}
    runtimes: Dict[str, float] = {}
    failures: List[str] = []

    def gpu_worker(gpu: str) -> None:
        while True:
            if stop_event.is_set():
                return
            try:
                plan = task_queue.get_nowait()
            except queue.Empty:
                return
            run_name = plan["run_name"]
            result_path = Path(plan["result_path"])
            if not args.rerun and result_is_complete(result_path, plan):
                with data_lock:
                    statuses[run_name] = "skipped"
                    assigned_gpus[run_name] = gpu
                    runtimes[run_name] = 0.0
                with print_lock:
                    print(f"[GPU {gpu}] SKIP {run_name}", flush=True)
                task_queue.task_done()
                continue

            attempt_id = hashlib.sha256(
                f"{run_name}:{time.time_ns()}:{os.getpid()}".encode("utf-8")
            ).hexdigest()[:16]
            command = build_worker_command(script_path, plan, args, attempt_id)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["EXPERIMENT_GPU_ID"] = gpu
            environment["PYTHONHASHSEED"] = str(plan["seed"])
            environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            log_path = Path(plan["log_path"])
            with print_lock:
                print(f"[GPU {gpu}] START {run_name}", flush=True)
            start = time.perf_counter()
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    f"\n\n=== attempt {time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"id={attempt_id} ===\n"
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
            complete = result_is_complete(result_path, plan, attempt_id)
            status = "completed" if completed.returncode == 0 and complete else "failed"
            with data_lock:
                statuses[run_name] = status
                assigned_gpus[run_name] = gpu
                runtimes[run_name] = runtime
                if status == "failed":
                    failures.append(
                        f"{run_name}: exit={completed.returncode}, log={log_path}"
                    )
                    stop_event.set()
            with print_lock:
                print(
                    f"[GPU {gpu}] {status.upper()} {run_name} "
                    f"({runtime / 60.0:.1f} min)",
                    flush=True,
                )
            task_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(gpu_worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()

    plan_rows = [
        {
            **plan,
            "status": statuses.get(plan["run_name"], "unknown"),
            "gpu": assigned_gpus.get(plan["run_name"]),
            "runtime_seconds": runtimes.get(plan["run_name"]),
        }
        for plan in plans
    ]
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)
    if failures:
        print(f"\nFailed trainings: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(
            "At least one fit failed. Fix the logged error and rerun the same "
            "command; completed fits will be resumed."
        )

    results_by_training: Dict[str, List[str]] = {
        configuration["training_id"]: [] for configuration in configurations
    }
    for plan in plans:
        if not result_is_complete(Path(plan["result_path"]), plan):
            raise RuntimeError(
                f"Incomplete result before scoring: {plan['result_path']}"
            )
        results_by_training[plan["training_id"]].append(plan["result_path"])

    score_tasks: List[Dict[str, Any]] = []
    score_paths: List[Path] = []
    for configuration in configurations:
        result_paths = sorted(
            results_by_training[configuration["training_id"]],
            key=lambda path: int(core.read_json(Path(path))["seed"]),
        )
        artifact_hashes = [
            core.read_json(Path(path))["prediction_sha256"] for path in result_paths
        ]
        score_identity = {
            "experiment_key": experiment_key,
            "configuration": configuration,
            "seeds": [int(seed) for seed in args.seeds],
            "coefficient_values": coefficient_values,
            "artifact_hashes": artifact_hashes,
            "metrics": core.METRICS,
        }
        score_key = stable_hash(score_identity, 16)
        csv_path = score_dir / (
            f"{configuration['training_id']}_{score_key}_scores.csv.gz"
        )
        metadata_path = score_dir / (
            f"{configuration['training_id']}_{score_key}_scores.json"
        )
        if args.recompute_scores:
            for path in (csv_path, metadata_path):
                if path.exists():
                    path.unlink()
        score_tasks.append(
            {
                "configuration": configuration,
                "seeds": [int(seed) for seed in args.seeds],
                "coefficient_values": coefficient_values,
                "score_chunk_size": int(args.score_chunk_size),
                "result_paths": result_paths,
                "score_key": score_key,
                "csv_path": str(csv_path),
                "metadata_path": str(metadata_path),
            }
        )
        score_paths.append(csv_path)

    print(
        f"\nScoring {len(score_tasks)} training configurations with "
        f"{args.score_workers} CPU workers...",
        flush=True,
    )
    completed_scores = 0
    with ProcessPoolExecutor(max_workers=args.score_workers) as executor:
        future_to_task = {
            executor.submit(score_training_configuration, task): task
            for task in score_tasks
        }
        for future in as_completed(future_to_task):
            outcome = future.result()
            completed_scores += 1
            action = "SKIP" if outcome["skipped"] else "DONE"
            print(
                f"[SCORE] {action} {outcome['training_id']} "
                f"({completed_scores}/{len(score_tasks)})",
                flush=True,
            )

    combined = pd.concat(
        [pd.read_csv(path, compression="gzip") for path in score_paths],
        ignore_index=True,
    )
    ranked = rank_joint_configurations(combined, args.macro_f1_retention)
    full_ranking_path = results_dir / f"{stem}_all_ranked_configurations.csv.gz"
    atomic_csv(full_ranking_path, ranked, compression="gzip")

    family_ranking_paths: Dict[str, str] = {}
    for family in ALL_FAMILIES:
        family_path = results_dir / f"{stem}_{family}_ranking.csv.gz"
        atomic_csv(
            family_path,
            ranked[ranked["family"] == family],
            compression="gzip",
        )
        family_ranking_paths[family] = str(family_path)

    top = (
        ranked.groupby("family", sort=False, group_keys=False)
        .head(args.top_n)
        .reset_index(drop=True)
    )
    top_path = results_dir / f"{stem}_top{args.top_n}_per_family.csv"
    atomic_csv(top_path, top)

    winners = (
        ranked[ranked["rank_within_family"] == 1]
        .copy()
        .sort_values("family", key=lambda values: values.map(ALL_FAMILIES.index))
        .reset_index(drop=True)
    )
    if len(winners) != len(ALL_FAMILIES):
        raise RuntimeError("Exactly one winner per model family was expected.")
    winners["policy"] = np.where(
        winners["is_raw_argmax"],
        "selected_raw_argmax",
        "selected_score_scaling",
    )
    winners_path = results_dir / f"{stem}_winners.csv"
    atomic_csv(winners_path, winners)

    raw_companions: List[pd.DataFrame] = []
    for _, winner in winners.iterrows():
        if bool(winner["is_raw_argmax"]):
            continue
        matches = ranked[
            (ranked["training_id"] == winner["training_id"]) & ranked["is_raw_argmax"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Raw companion missing for {winner['training_id']}: {len(matches)} rows."
            )
        companion = matches.copy()
        companion["policy"] = "raw_argmax_companion"
        raw_companions.append(companion)
    comparison_parts = [winners, *raw_companions]
    winner_comparison = pd.concat(comparison_parts, ignore_index=True).sort_values(
        ["family", "policy"]
    )
    winner_comparison_path = results_dir / f"{stem}_winner_raw_comparison.csv"
    atomic_csv(winner_comparison_path, winner_comparison)

    raw_rankings = ranked[ranked["is_raw_argmax"]].copy()
    raw_rankings = raw_rankings.sort_values(
        ["family", "macro_f1_mean", "rare_f1_mean"],
        ascending=[True, False, False],
    )
    raw_rankings["raw_rank_within_family"] = (
        raw_rankings.groupby("family").cumcount() + 1
    )
    raw_ranking_path = results_dir / f"{stem}_raw_beta_gamma_ranking.csv"
    atomic_csv(raw_ranking_path, raw_rankings)

    winner_per_seed_rows: List[Dict[str, Any]] = []
    for _, winner in winners.iterrows():
        result_paths = sorted(
            results_by_training[winner["training_id"]],
            key=lambda path: int(core.read_json(Path(path))["seed"]),
        )
        for result_path_raw in result_paths:
            result_path = Path(result_path_raw)
            result = core.read_json(result_path)
            prediction_path = Path(result["prediction_path"])
            with np.load(prediction_path, allow_pickle=False) as artifact:
                labels = np.asarray(artifact["validation_labels"], dtype=np.int64)
                probabilities = np.asarray(
                    artifact["validation_probabilities"],
                    dtype=np.float64,
                )
            raw_predictions = np.argmax(probabilities, axis=1).astype(np.int64)
            policies = [
                (
                    str(winner["policy"]),
                    float(winner["r2l_score_coefficient"]),
                    float(winner["u2r_score_coefficient"]),
                )
            ]
            if not bool(winner["is_raw_argmax"]):
                policies.append(("raw_argmax_companion", 1.0, 1.0))
            for policy, r2l_coefficient, u2r_coefficient in policies:
                if np.isclose(r2l_coefficient, 1.0) and np.isclose(
                    u2r_coefficient, 1.0
                ):
                    predictions = raw_predictions
                else:
                    predictions = core.apply_class_score_scaling(
                        probabilities,
                        {2: r2l_coefficient, 3: u2r_coefficient},
                    )
                metrics = core.calculate_metrics(labels, predictions)
                winner_per_seed_rows.append(
                    {
                        "family": winner["family"],
                        "display_name": winner["display_name"],
                        "training_id": winner["training_id"],
                        "seed": int(result["seed"]),
                        "policy": policy,
                        "beta": winner["beta"],
                        "focal_gamma": winner["focal_gamma"],
                        "r2l_score_coefficient": r2l_coefficient,
                        "u2r_score_coefficient": u2r_coefficient,
                        "result_path": str(result_path),
                        "prediction_path": str(prediction_path),
                        **metrics,
                    }
                )
    winner_per_seed = pd.DataFrame(winner_per_seed_rows)
    for _, winner in winners.iterrows():
        selected_runs = winner_per_seed[
            (winner_per_seed["family"] == winner["family"])
            & (winner_per_seed["policy"] == winner["policy"])
        ]
        if len(selected_runs) != len(args.seeds):
            raise RuntimeError(
                f"Winner {winner['family']} does not have every requested seed."
            )
        for metric in core.METRICS:
            values = selected_runs[metric].to_numpy(dtype=np.float64)
            expected_mean = float(winner[f"{metric}_mean"])
            expected_std = float(winner[f"{metric}_std"])
            actual_mean = float(values.mean())
            actual_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            if not np.isclose(actual_mean, expected_mean, atol=1e-12) or not np.isclose(
                actual_std,
                expected_std,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Winner per-seed audit mismatch for {winner['family']} {metric}."
                )
    winner_per_seed_path = results_dir / f"{stem}_winner_per_seed_metrics.csv"
    atomic_csv(winner_per_seed_path, winner_per_seed)

    formatted_winners = formatted_table(winners)
    formatted_winners_path = results_dir / f"{stem}_winners_formatted.csv"
    atomic_csv(formatted_winners_path, formatted_winners)
    formatted_comparison = formatted_table(winner_comparison)
    formatted_comparison_path = (
        results_dir / f"{stem}_winner_raw_comparison_formatted.csv"
    )
    atomic_csv(formatted_comparison_path, formatted_comparison)

    readable_path = results_dir / f"{stem}_summary.txt"
    readable_lines = [
        "Fixed-backbone imbalance-control sweep",
        f"Experiment key: {experiment_key}",
        f"Seeds: {args.seeds}",
        f"Seeded fits: {len(plans)}",
        f"Ranked averaged configurations: {len(ranked)}",
        "Selection data: KDDTrain+ validation only",
        "KDDTest+ accessed: NO",
        "",
        "Winning configuration per family:",
        formatted_winners.to_string(index=False),
        "",
        "Winning policy versus its exact raw-argmax companion when scaling was selected:",
        formatted_comparison.to_string(index=False),
    ]
    readable_path.write_text("\n".join(readable_lines) + "\n", encoding="utf-8")

    completed_protocol = {
        **protocol,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_paths": {
            "full_ranking": str(full_ranking_path),
            "family_rankings": family_ranking_paths,
            "top_per_family": str(top_path),
            "winners": str(winners_path),
            "winner_raw_comparison": str(winner_comparison_path),
            "winner_per_seed_metrics": str(winner_per_seed_path),
            "raw_beta_gamma_ranking": str(raw_ranking_path),
            "formatted_winners": str(formatted_winners_path),
            "formatted_winner_raw_comparison": str(formatted_comparison_path),
            "readable_summary": str(readable_path),
        },
    }
    core.atomic_json(protocol_path, completed_protocol)

    print("\n=== Winning validation configuration per family ===")
    print(formatted_winners.to_string(index=False))
    print(f"\nFull ranking: {full_ranking_path}")
    print(f"Top configurations: {top_path}")
    print(f"Winners: {winners_path}")
    print(f"Winner/raw comparison: {winner_comparison_path}")
    print(f"Readable summary: {readable_path}")
    print("KDDTest+ was not accessed. Use a separate frozen-winner evaluator later.")


if __name__ == "__main__":
    main()
