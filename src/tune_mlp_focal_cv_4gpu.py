"""Tune MLP class-balanced focal loss with fixed four-fold OOF validation.

Default experiment
------------------
* Model: the fixed 99,845-parameter MLP used by ``cnn_opt_1d_4gpu.py``.
* Data: KDDTrain+ only. KDDTest+ and synthetic data are never accessed.
* Grid: beta in {0.99, 0.999, 0.9999} and
  gamma in {0.25, 0.5, 0.75, 1.0, 1.5, 2.0}.
* Repetition: training seeds {0, 1, 2} on one frozen four-fold split.
* Training: 25 fixed epochs, ordinary shuffled batches, raw argmax.

The four folds are assigned to four GPUs. For every beta/gamma/seed setting,
GPU i trains on three folds and predicts fold i. The four predictions are then
placed back in original row order to form one complete out-of-fold prediction
vector. Metrics are calculated once on that vector for each seed, followed by
mean and sample standard deviation across the three seeds.

Each fit runs in a fresh subprocess with exactly one visible GPU. Completed
artifacts are validated and reused when the same command is rerun.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

import run_no_ctgan_model_ablation_4gpu as core


SCHEMA_VERSION = 1
FOLD_COUNT = 4
DEFAULT_FOLD_SEED = 0
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_BETAS = [0.99, 0.999, 0.9999]
DEFAULT_FOCAL_GAMMAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

FIXED_BACKBONE: Dict[str, Any] = {
    "dense_units": 256,
    "dropout1": 0.25,
    "dropout2": 0.30,
    "batch_norm": True,
    "expected_parameters": 99_845,
}

METRICS = list(core.METRICS)


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def value_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def progress_bar(completed: int, total: int, width: int = 30) -> str:
    """Return a compact, dependency-free overall progress indicator."""
    if total <= 0:
        return "[------------------------------]   0.0% (0/0)"
    completed = max(0, min(int(completed), int(total)))
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    percentage = 100.0 * completed / total
    return f"[{bar}] {percentage:6.2f}% ({completed}/{total})"


def configurations(
    betas: Sequence[float], focal_gammas: Sequence[float]
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for beta in betas:
        for gamma in focal_gammas:
            identity = {
                "beta": float(beta),
                "focal_gamma": float(gamma),
                "backbone": FIXED_BACKBONE,
                "batching": "ordinary_shuffled",
                "decision_policy": "raw_argmax",
            }
            config_key = stable_hash(identity, 10)
            output.append(
                {
                    "config_key": config_key,
                    "config_id": (
                        f"b{value_token(beta)}_g{value_token(gamma)}_{config_key}"
                    ),
                    "beta": float(beta),
                    "focal_gamma": float(gamma),
                }
            )
    return output


def make_fixed_folds(
    labels: np.ndarray,
    fold_seed: int,
) -> tuple[List[Dict[str, Any]], np.ndarray]:
    from sklearn.model_selection import StratifiedKFold

    labels = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=FOLD_COUNT,
        shuffle=True,
        random_state=int(fold_seed),
    )
    fold_ids = np.full(len(labels), -1, dtype=np.int64)
    folds: List[Dict[str, Any]] = []
    for fold_id, (train_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(labels), dtype=np.uint8), labels)
    ):
        train_indices = np.sort(np.asarray(train_indices, dtype=np.int64))
        validation_indices = np.sort(np.asarray(validation_indices, dtype=np.int64))
        if np.any(fold_ids[validation_indices] != -1):
            raise RuntimeError("Validation folds overlap.")
        fold_ids[validation_indices] = fold_id
        folds.append(
            {
                "fold_id": fold_id,
                "fold_number": fold_id + 1,
                "train_indices": train_indices,
                "validation_indices": validation_indices,
                "train_indices_sha256": core.sha256_indices(train_indices),
                "validation_indices_sha256": core.sha256_indices(validation_indices),
                "train_counts": np.bincount(labels[train_indices], minlength=5).astype(
                    int
                ),
                "validation_counts": np.bincount(
                    labels[validation_indices], minlength=5
                ).astype(int),
            }
        )

    if np.any(fold_ids < 0):
        raise RuntimeError("The fixed folds do not cover every KDDTrain+ row.")
    if len(folds) != FOLD_COUNT:
        raise RuntimeError(f"Expected {FOLD_COUNT} folds, got {len(folds)}.")
    validation_union = np.concatenate([fold["validation_indices"] for fold in folds])
    if not np.array_equal(
        np.sort(validation_union), np.arange(len(labels), dtype=np.int64)
    ):
        raise RuntimeError("Validation folds do not form an exact row partition.")
    return folds, fold_ids


def fold_protocol_rows(folds: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "fold_id": int(fold["fold_id"]),
            "fold_number": int(fold["fold_number"]),
            "train_size": int(len(fold["train_indices"])),
            "validation_size": int(len(fold["validation_indices"])),
            "train_indices_sha256": fold["train_indices_sha256"],
            "validation_indices_sha256": fold["validation_indices_sha256"],
            "train_counts": np.asarray(fold["train_counts"], dtype=int).tolist(),
            "validation_counts": np.asarray(
                fold["validation_counts"], dtype=int
            ).tolist(),
        }
        for fold in folds
    ]


def fold_cache_is_complete(
    cache_path: Path,
    metadata_path: Path,
    experiment_key: str,
    fold: Dict[str, Any],
    master_labels: np.ndarray,
) -> bool:
    if not cache_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = core.read_json(metadata_path)
        expected = {
            "schema_version": SCHEMA_VERSION,
            "experiment_key": experiment_key,
            "fold_id": int(fold["fold_id"]),
            "train_indices_sha256": fold["train_indices_sha256"],
            "validation_indices_sha256": fold["validation_indices_sha256"],
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            return False
        if metadata.get("cache_sha256") != core.sha256_file(cache_path):
            return False
        required = {
            "X_train",
            "y_train",
            "X_validation",
            "y_validation",
            "train_indices",
            "validation_indices",
        }
        with np.load(cache_path, allow_pickle=False) as artifact:
            if not required.issubset(artifact.files):
                return False
            arrays = {name: np.asarray(artifact[name]) for name in required}
        train_indices = arrays["train_indices"].astype(np.int64, copy=False)
        validation_indices = arrays["validation_indices"].astype(np.int64, copy=False)
        if core.sha256_indices(train_indices) != fold["train_indices_sha256"]:
            return False
        if core.sha256_indices(validation_indices) != fold["validation_indices_sha256"]:
            return False
        if arrays["X_train"].shape != (len(train_indices), 121):
            return False
        if arrays["X_validation"].shape != (len(validation_indices), 121):
            return False
        if arrays["y_train"].shape != (len(train_indices),):
            return False
        if arrays["y_validation"].shape != (len(validation_indices),):
            return False
        if not np.array_equal(arrays["y_train"], master_labels[train_indices]):
            return False
        if not np.array_equal(
            arrays["y_validation"], master_labels[validation_indices]
        ):
            return False
        return bool(
            np.isfinite(arrays["X_train"]).all()
            and np.isfinite(arrays["X_validation"]).all()
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def prepare_fold_cache(
    raw_train: pd.DataFrame,
    master_labels: np.ndarray,
    fold: Dict[str, Any],
    cache_path: Path,
    metadata_path: Path,
    experiment_key: str,
) -> Dict[str, Any]:
    if fold_cache_is_complete(
        cache_path,
        metadata_path,
        experiment_key,
        fold,
        master_labels,
    ):
        return core.read_json(metadata_path)

    train_indices = np.asarray(fold["train_indices"], dtype=np.int64)
    validation_indices = np.asarray(fold["validation_indices"], dtype=np.int64)
    train_frame = raw_train.iloc[train_indices].copy().reset_index(drop=True)
    validation_frame = raw_train.iloc[validation_indices].copy().reset_index(drop=True)
    preprocessor = core.fit_fold_preprocessor(train_frame)
    X_train, y_train = core.transform_with_fold_preprocessor(train_frame, preprocessor)
    X_validation, y_validation = core.transform_with_fold_preprocessor(
        validation_frame, preprocessor
    )
    if not np.array_equal(y_train, master_labels[train_indices]):
        raise RuntimeError("Cached training labels lost their original row order.")
    if not np.array_equal(y_validation, master_labels[validation_indices]):
        raise RuntimeError("Cached validation labels lost their original row order.")

    core.atomic_npz(
        cache_path,
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_validation=np.asarray(X_validation, dtype=np.float32),
        y_validation=np.asarray(y_validation, dtype=np.int64),
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
    scaler = preprocessor["scaler"]
    scaler_state = np.concatenate(
        [
            np.asarray(scaler.data_min_, dtype=np.float64),
            np.asarray(scaler.data_max_, dtype=np.float64),
        ]
    )
    feature_order = list(preprocessor["ordered_features"])
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "fold_id": int(fold["fold_id"]),
        "fold_number": int(fold["fold_number"]),
        "train_size": int(len(train_indices)),
        "validation_size": int(len(validation_indices)),
        "train_indices_sha256": fold["train_indices_sha256"],
        "validation_indices_sha256": fold["validation_indices_sha256"],
        "train_counts": np.bincount(y_train, minlength=5).astype(int).tolist(),
        "validation_counts": np.bincount(y_validation, minlength=5)
        .astype(int)
        .tolist(),
        "feature_count": 121,
        "feature_order_sha256": hashlib.sha256(
            "\n".join(feature_order).encode("utf-8")
        ).hexdigest(),
        "scaler_state_sha256": core.sha256_array(scaler_state),
        "preprocessor_fit_partition": "outer_training_fold_only",
        "cache_path": str(cache_path),
        "cache_sha256": core.sha256_file(cache_path),
    }
    core.atomic_json(metadata_path, metadata)
    if not fold_cache_is_complete(
        cache_path,
        metadata_path,
        experiment_key,
        fold,
        master_labels,
    ):
        raise RuntimeError(f"Prepared fold cache failed validation: {cache_path}")
    return metadata


def worker_configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "config_key": args.worker_config_key,
        "config_id": args.worker_config_id,
        "beta": float(args.worker_beta),
        "focal_gamma": float(args.worker_focal_gamma),
    }


def run_training_worker(args: argparse.Namespace) -> None:
    import tensorflow as tf

    from cnn_gan_foc import ClassBalancedFocalLoss  # type: ignore
    from cnn_opt_1d_4gpu import build_opt_mlp  # type: ignore

    cache_path = Path(args.worker_cache_path)
    cache_metadata_path = Path(args.worker_cache_metadata_path)
    cache_metadata = core.read_json(cache_metadata_path)
    if cache_metadata.get("cache_sha256") != core.sha256_file(cache_path):
        raise RuntimeError("Fold cache hash does not match its metadata.")
    if cache_metadata.get("experiment_key") != args.experiment_key:
        raise RuntimeError("Fold cache belongs to a different experiment.")
    if int(cache_metadata.get("fold_id", -1)) != int(args.worker_fold_id):
        raise RuntimeError("Fold cache ID does not match the worker fold.")

    with np.load(cache_path, allow_pickle=False) as artifact:
        X_train_flat = np.asarray(artifact["X_train"], dtype=np.float32)
        y_train = np.asarray(artifact["y_train"], dtype=np.int64)
        X_validation_flat = np.asarray(artifact["X_validation"], dtype=np.float32)
        y_validation = np.asarray(artifact["y_validation"], dtype=np.int64)
        validation_indices = np.asarray(artifact["validation_indices"], dtype=np.int64)

    visible_gpus = tf.config.list_physical_devices("GPU")
    if len(visible_gpus) != 1:
        raise RuntimeError(
            "Each worker must see exactly one GPU; TensorFlow sees "
            f"{len(visible_gpus)}: {visible_gpus}."
        )
    for device in visible_gpus:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    deterministic_enabled = False
    if args.deterministic_ops:
        try:
            tf.config.experimental.enable_op_determinism()
            deterministic_enabled = True
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError(
                "Deterministic TensorFlow operations were requested but could "
                "not be enabled."
            ) from error

    model_seed = int(args.worker_seed)
    tf.keras.utils.set_random_seed(model_seed)
    np.random.seed(model_seed)
    alpha, alpha_counts = core.effective_number_alpha(
        y_train,
        beta=float(args.worker_beta),
        num_classes=5,
    )
    loss = ClassBalancedFocalLoss(
        alpha=alpha,
        gamma=float(args.worker_focal_gamma),
    )
    model = build_opt_mlp(
        loss=loss,
        dense_units=FIXED_BACKBONE["dense_units"],
        dropout1=FIXED_BACKBONE["dropout1"],
        dropout2=FIXED_BACKBONE["dropout2"],
        use_batch_norm=FIXED_BACKBONE["batch_norm"],
    )
    parameter_count = int(model.count_params())
    if parameter_count != FIXED_BACKBONE["expected_parameters"]:
        raise RuntimeError(
            "MLP parameter count changed: expected "
            f"{FIXED_BACKBONE['expected_parameters']}, got {parameter_count}."
        )

    X_train = X_train_flat
    X_validation = X_validation_flat
    started = time.perf_counter()
    history = model.fit(
        X_train,
        y_train,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        shuffle=True,
        verbose=int(args.fit_verbose),
    )
    probabilities = np.asarray(
        model.predict(
            X_validation,
            batch_size=int(args.batch_size),
            verbose=0,
        ),
        dtype=np.float32,
    )
    training_seconds = time.perf_counter() - started
    if probabilities.shape != (len(y_validation), 5):
        raise RuntimeError(
            f"Unexpected validation probability shape: {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("Validation probabilities contain non-finite values.")
    predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    metrics = core.calculate_metrics(y_validation, predictions)

    result_path = Path(args.worker_result_path)
    prediction_path = result_path.with_name(
        f"{args.worker_run_name}_{args.worker_attempt_id}_predictions.npz"
    )
    core.atomic_npz(
        prediction_path,
        validation_indices=validation_indices,
        validation_labels=y_validation,
        validation_probabilities=probabilities,
        raw_validation_predictions=predictions,
    )
    configuration = worker_configuration(args)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": args.experiment_key,
        "attempt_id": args.worker_attempt_id,
        "run_name": args.worker_run_name,
        **configuration,
        "seed": model_seed,
        "fold_id": int(args.worker_fold_id),
        "fold_number": int(args.worker_fold_id) + 1,
        "train_indices_sha256": cache_metadata["train_indices_sha256"],
        "validation_indices_sha256": cache_metadata["validation_indices_sha256"],
        "train_counts": cache_metadata["train_counts"],
        "validation_counts": cache_metadata["validation_counts"],
        "feature_order_sha256": cache_metadata["feature_order_sha256"],
        "scaler_state_sha256": cache_metadata["scaler_state_sha256"],
        "fold_cache_sha256": cache_metadata["cache_sha256"],
        "model": "fixed_mlp",
        "backbone": FIXED_BACKBONE,
        "model_parameters": parameter_count,
        "loss": "class_balanced_focal",
        "class_weighting": "effective_number_from_outer_training_fold",
        "alpha": np.asarray(alpha, dtype=float).tolist(),
        "alpha_counts": np.asarray(alpha_counts, dtype=int).tolist(),
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "batch_size": int(args.batch_size),
        "batching": "ordinary_shuffled",
        "minority_guaranteed_batches": False,
        "ctgan_used": False,
        "synthetic_rows": 0,
        "decision_policy": "raw_argmax",
        "score_scaling_used": False,
        "dataset": "KDDTrain+ only",
        "evaluation_partition": "one fixed outer validation fold",
        "kddtest_accessed": False,
        "validation_used_during_training": False,
        "checkpointing": "none_fixed_epoch_budget",
        "raw_validation_metrics": metrics,
        "prediction_path": str(prediction_path),
        "prediction_sha256": core.sha256_file(prediction_path),
        "training_seconds": float(training_seconds),
        "assigned_gpu": os.environ.get("EXPERIMENT_GPU_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tensorflow_visible_gpu_count": len(visible_gpus),
        "deterministic_ops_requested": bool(args.deterministic_ops),
        "deterministic_ops_enabled": deterministic_enabled,
        "tensorflow_version": core.package_version("tensorflow"),
        "keras_version": core.package_version("keras"),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    core.atomic_json(result_path, result)
    tf.keras.backend.clear_session()
    print(
        f"Completed {args.worker_run_name}: "
        f"rare-F1={metrics['rare_f1']:.6f}, "
        f"macro-F1={metrics['macro_f1']:.6f}",
        flush=True,
    )


def prediction_artifact_is_complete(
    path: Path,
    result: Dict[str, Any],
) -> bool:
    if not path.is_file() or core.sha256_file(path) != result.get("prediction_sha256"):
        return False
    try:
        with np.load(path, allow_pickle=False) as artifact:
            required = {
                "validation_indices",
                "validation_labels",
                "validation_probabilities",
                "raw_validation_predictions",
            }
            if not required.issubset(artifact.files):
                return False
            indices = np.asarray(artifact["validation_indices"], dtype=np.int64)
            labels = np.asarray(artifact["validation_labels"], dtype=np.int64)
            probabilities = np.asarray(
                artifact["validation_probabilities"], dtype=np.float32
            )
            predictions = np.asarray(
                artifact["raw_validation_predictions"], dtype=np.int64
            )
        expected_count = int(sum(result["validation_counts"]))
        if indices.shape != (expected_count,):
            return False
        if labels.shape != (expected_count,) or predictions.shape != (expected_count,):
            return False
        if probabilities.shape != (expected_count, 5):
            return False
        if core.sha256_indices(indices) != result["validation_indices_sha256"]:
            return False
        if not np.isfinite(probabilities).all():
            return False
        if not np.array_equal(predictions, np.argmax(probabilities, axis=1)):
            return False
        if (
            np.bincount(labels, minlength=5).astype(int).tolist()
            != result["validation_counts"]
        ):
            return False
        return core.metrics_match_predictions(
            result["raw_validation_metrics"], labels, predictions
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


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
            "schema_version": SCHEMA_VERSION,
            "experiment_key": plan["experiment_key"],
            "run_name": plan["run_name"],
            "config_key": plan["config_key"],
            "config_id": plan["config_id"],
            "beta": float(plan["beta"]),
            "focal_gamma": float(plan["focal_gamma"]),
            "seed": int(plan["seed"]),
            "fold_id": int(plan["fold_id"]),
            "train_indices_sha256": plan["train_indices_sha256"],
            "validation_indices_sha256": plan["validation_indices_sha256"],
            "kddtest_accessed": False,
            "validation_used_during_training": False,
            "minority_guaranteed_batches": False,
            "score_scaling_used": False,
            "synthetic_rows": 0,
            "deterministic_ops_requested": bool(plan["deterministic_ops_requested"]),
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
        prediction_path = Path(str(result.get("prediction_path", "")))
        return prediction_artifact_is_complete(prediction_path, result)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
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
        "--worker-run-name",
        str(plan["run_name"]),
        "--worker-result-path",
        str(plan["result_path"]),
        "--worker-attempt-id",
        attempt_id,
        "--worker-config-key",
        str(plan["config_key"]),
        "--worker-config-id",
        str(plan["config_id"]),
        "--worker-beta",
        str(plan["beta"]),
        "--worker-focal-gamma",
        str(plan["focal_gamma"]),
        "--worker-seed",
        str(plan["seed"]),
        "--worker-fold-id",
        str(plan["fold_id"]),
        "--worker-cache-path",
        str(plan["cache_path"]),
        "--worker-cache-metadata-path",
        str(plan["cache_metadata_path"]),
        "--experiment-key",
        str(plan["experiment_key"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--fit-verbose",
        str(args.fit_verbose),
    ]
    if args.deterministic_ops:
        command.append("--deterministic-ops")
    return command


def confusion_values(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    matrix = np.bincount(
        np.asarray(y_true, dtype=np.int64) * 5 + np.asarray(y_pred, dtype=np.int64),
        minlength=25,
    ).reshape(5, 5)
    return {
        f"confusion_{row}_{column}": int(matrix[row, column])
        for row in range(5)
        for column in range(5)
    }


def aggregate_results(
    plans: Sequence[Dict[str, Any]],
    configs: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    master_labels: np.ndarray,
    fold_ids: np.ndarray,
    oof_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_by_key: Dict[tuple[str, int, int], Dict[str, Any]] = {}
    fold_rows: List[Dict[str, Any]] = []
    for plan in plans:
        result_path = Path(plan["result_path"])
        if not result_is_complete(result_path, plan):
            raise RuntimeError(f"Incomplete fit during aggregation: {result_path}")
        result = core.read_json(result_path)
        key = (plan["config_key"], int(plan["seed"]), int(plan["fold_id"]))
        if key in result_by_key:
            raise RuntimeError(f"Duplicate completed fit: {key}")
        result_by_key[key] = result
        fold_rows.append(
            {
                "config_key": plan["config_key"],
                "config_id": plan["config_id"],
                "beta": float(plan["beta"]),
                "focal_gamma": float(plan["focal_gamma"]),
                "seed": int(plan["seed"]),
                "fold_id": int(plan["fold_id"]),
                "fold_number": int(plan["fold_id"]) + 1,
                "gpu": result["assigned_gpu"],
                "runtime_seconds": float(result["training_seconds"]),
                "result_path": str(result_path),
                "prediction_path": result["prediction_path"],
                **{
                    metric: float(result["raw_validation_metrics"][metric])
                    for metric in METRICS
                },
            }
        )

    seed_rows: List[Dict[str, Any]] = []
    row_count = len(master_labels)
    oof_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        for seed in seeds:
            probabilities = np.full((row_count, 5), np.nan, dtype=np.float32)
            coverage = np.zeros(row_count, dtype=np.uint8)
            source_hashes: List[str] = []
            for fold_id in range(FOLD_COUNT):
                key = (config["config_key"], int(seed), fold_id)
                if key not in result_by_key:
                    raise RuntimeError(f"Missing fold result for {key}.")
                result = result_by_key[key]
                prediction_path = Path(result["prediction_path"])
                with np.load(prediction_path, allow_pickle=False) as artifact:
                    indices = np.asarray(artifact["validation_indices"], dtype=np.int64)
                    labels = np.asarray(artifact["validation_labels"], dtype=np.int64)
                    fold_probabilities = np.asarray(
                        artifact["validation_probabilities"], dtype=np.float32
                    )
                if np.any(coverage[indices] != 0):
                    raise RuntimeError(
                        f"OOF overlap for {config['config_id']}, seed {seed}."
                    )
                if not np.array_equal(labels, master_labels[indices]):
                    raise RuntimeError("OOF labels do not match original rows.")
                probabilities[indices] = fold_probabilities
                coverage[indices] += 1
                source_hashes.append(str(result["prediction_sha256"]))

            if not np.all(coverage == 1):
                missing = int(np.count_nonzero(coverage == 0))
                duplicate = int(np.count_nonzero(coverage > 1))
                raise RuntimeError(
                    f"Invalid OOF coverage: missing={missing}, duplicate={duplicate}."
                )
            if not np.isfinite(probabilities).all():
                raise RuntimeError("Assembled OOF probabilities are not finite.")
            predictions = np.argmax(probabilities, axis=1).astype(np.int64)
            metrics = core.calculate_metrics(master_labels, predictions)
            lineage_sha256 = hashlib.sha256(
                "\n".join(source_hashes).encode("ascii")
            ).hexdigest()
            oof_path = oof_dir / (
                f"{config['config_id']}_s{int(seed)}_oof_predictions.npz"
            )
            core.atomic_npz(
                oof_path,
                row_indices=np.arange(row_count, dtype=np.int64),
                fold_ids=np.asarray(fold_ids, dtype=np.int64),
                labels=np.asarray(master_labels, dtype=np.int64),
                probabilities=probabilities,
                raw_predictions=predictions,
            )
            seed_rows.append(
                {
                    "config_key": config["config_key"],
                    "config_id": config["config_id"],
                    "beta": float(config["beta"]),
                    "focal_gamma": float(config["focal_gamma"]),
                    "seed": int(seed),
                    "completed_folds": FOLD_COUNT,
                    "oof_rows": row_count,
                    "oof_coverage_min": int(coverage.min()),
                    "oof_coverage_max": int(coverage.max()),
                    "source_predictions_sha256": lineage_sha256,
                    "oof_path": str(oof_path),
                    "oof_sha256": core.sha256_file(oof_path),
                    **metrics,
                    **confusion_values(master_labels, predictions),
                }
            )

    fold_frame = pd.DataFrame(fold_rows).sort_values(
        ["beta", "focal_gamma", "seed", "fold_id"]
    )
    seed_frame = pd.DataFrame(seed_rows).sort_values(["beta", "focal_gamma", "seed"])
    expected_seed_rows = len(configs) * len(seeds)
    if len(seed_frame) != expected_seed_rows:
        raise RuntimeError(
            f"Expected {expected_seed_rows} seed-level rows, got {len(seed_frame)}."
        )

    summary_rows: List[Dict[str, Any]] = []
    expected_seeds = {int(seed) for seed in seeds}
    for config in configs:
        group = seed_frame[seed_frame["config_key"] == config["config_key"]]
        observed_seeds = set(group["seed"].astype(int).tolist())
        if observed_seeds != expected_seeds or len(group) != len(expected_seeds):
            raise RuntimeError(
                f"Incomplete seed aggregation for {config['config_id']}."
            )
        row: Dict[str, Any] = {
            **config,
            "runs": len(group),
            "seeds": ",".join(str(seed) for seed in sorted(observed_seeds)),
            "folds_per_seed": FOLD_COUNT,
            "total_fits": len(group) * FOLD_COUNT,
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(
        [
            "rare_f1_mean",
            "macro_f1_mean",
            "focal_gamma",
            "beta",
            "config_key",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary = summary.reset_index(drop=True)
    if len(summary) != len(configs):
        raise RuntimeError("Final ranking has the wrong number of configurations.")
    return fold_frame.reset_index(drop=True), seed_frame.reset_index(drop=True), summary


def formatted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary[["rank", "beta", "focal_gamma", "runs", "total_fits"]].copy()
    labels = {
        "rare_f1": "Rare Macro-F1",
        "macro_f1": "Macro-F1",
        "macro_recall": "Macro Recall",
        "accuracy": "Accuracy",
        "mcc": "MCC",
        "r2l_precision": "R2L Precision",
        "r2l_recall": "R2L Recall",
        "r2l_f1": "R2L F1",
        "u2r_precision": "U2R Precision",
        "u2r_recall": "U2R Recall",
        "u2r_f1": "U2R F1",
    }
    for metric, label in labels.items():
        output[label] = [
            f"{100.0 * mean:.2f}% +/- {100.0 * std:.2f}%"
            for mean, std in zip(
                summary[f"{metric}_mean"],
                summary[f"{metric}_std"],
                strict=True,
            )
        ]
    return output


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0", "1", "2", "3"],
        help="Exactly four GPU IDs, mapped in order to folds 1..4 (default: 0 1 2 3).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Paired model-training seeds (default: 0 1 2).",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=DEFAULT_BETAS,
        help="Effective-number beta grid (default: 0.99 0.999 0.9999).",
    )
    parser.add_argument(
        "--focal-gammas",
        type=float,
        nargs="+",
        default=DEFAULT_FOCAL_GAMMAS,
        help="Active focal gamma grid (default: 0.25 0.5 0.75 1.0 1.5 2.0).",
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=DEFAULT_FOLD_SEED,
        help="Seed used once to freeze the shared four-fold partition (default: 0).",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Fixed epochs per fit.")
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Ordinary shuffled batch size."
    )
    parser.add_argument(
        "--fit-verbose",
        type=int,
        choices=[0, 1, 2],
        default=2,
        help="Keras verbosity written to each per-fit log (default: 2).",
    )
    parser.add_argument(
        "--name-prefix",
        default="mlp_focal_stage1",
        type=str,
        help="Prefix for artifacts written under results/.",
    )
    parser.add_argument(
        "--deterministic-ops",
        action="store_true",
        help="Request deterministic TensorFlow GPU kernels when supported.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Force every fit to retrain. For normal resume, omit this flag; "
            "after an interrupted forced run, resume without it."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without writing or training.",
    )
    parser.add_argument(
        "--print-all-commands",
        action="store_true",
        help="With --dry-run, print all worker commands instead of the first eight.",
    )

    parser.add_argument(
        "--worker-mode",
        choices=["none", "train"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-run-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-attempt-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-config-id", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-beta", type=float, default=float("nan"), help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-focal-gamma",
        type=float,
        default=float("nan"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-fold-id", type=int, default=-1, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-cache-path", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-cache-metadata-path", default="", help=argparse.SUPPRESS
    )
    parser.add_argument("--experiment-key", default="", help=argparse.SUPPRESS)


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds cannot be negative.")
    if not args.betas or len(args.betas) != len(set(args.betas)):
        parser.error("--betas must contain unique values.")
    if any(not np.isfinite(beta) or not 0.0 < beta < 1.0 for beta in args.betas):
        parser.error("Every beta must be finite and strictly between 0 and 1.")
    if not args.focal_gammas or len(args.focal_gammas) != len(set(args.focal_gammas)):
        parser.error("--focal-gammas must contain unique values.")
    if any(not np.isfinite(gamma) or gamma <= 0.0 for gamma in args.focal_gammas):
        parser.error("Every focal gamma must be finite and greater than zero.")
    if args.fold_seed < 0:
        parser.error("--fold-seed cannot be negative.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")
    prefix = args.name_prefix.strip()
    if not prefix or Path(prefix).name != prefix:
        parser.error("--name-prefix must be a nonempty filename-safe name.")


def validate_parent_dependencies(parser: argparse.ArgumentParser) -> None:
    missing = [
        package
        for package in ("tensorflow", "scikit-learn", "matplotlib", "seaborn")
        if core.package_version(package) == "not-installed"
    ]
    if missing:
        parser.error(f"Missing required packages: {missing}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=("Four-GPU, four-fold MLP focal beta/gamma tuning on KDDTrain+.")
    )
    add_arguments(parser)
    args = parser.parse_args()
    validate_arguments(parser, args)

    if args.worker_mode == "train":
        required = {
            "--worker-run-name": args.worker_run_name,
            "--worker-result-path": args.worker_result_path,
            "--worker-attempt-id": args.worker_attempt_id,
            "--worker-config-key": args.worker_config_key,
            "--worker-config-id": args.worker_config_id,
            "--worker-cache-path": args.worker_cache_path,
            "--worker-cache-metadata-path": args.worker_cache_metadata_path,
            "--experiment-key": args.experiment_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"Worker invocation is missing: {missing}")
        if args.worker_fold_id not in range(FOLD_COUNT):
            parser.error(f"Worker fold must be 0..{FOLD_COUNT - 1}.")
        if not np.isfinite(args.worker_beta) or not np.isfinite(
            args.worker_focal_gamma
        ):
            parser.error("Worker beta and gamma must be finite.")
        run_training_worker(args)
        return

    try:
        gpus = core.parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    if len(gpus) != FOLD_COUNT:
        parser.error(
            f"This runner maps one fold to each GPU and therefore requires "
            f"exactly {FOLD_COUNT} GPU IDs."
        )

    train_path = repo_root / "data" / "KDDTrain+.txt"
    required_paths = [
        train_path,
        script_path,
        repo_root / "src" / "run_no_ctgan_model_ablation_4gpu.py",
        repo_root / "src" / "cnn_opt.py",
        repo_root / "src" / "cnn_opt_1d_4gpu.py",
        repo_root / "src" / "cnn_gan_foc.py",
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise SystemExit(f"Required files are missing: {missing_paths}")

    raw_train = core.load_collapsed_nsl_kdd(train_path, is_train=True)
    master_labels = raw_train["class"].to_numpy(dtype=np.int64)
    folds, fold_ids = make_fixed_folds(master_labels, args.fold_seed)
    configs = configurations(args.betas, args.focal_gammas)
    library_versions = {
        "tensorflow": core.package_version("tensorflow"),
        "keras": core.package_version("keras"),
        "scikit_learn": core.package_version("scikit-learn"),
        "matplotlib": core.package_version("matplotlib"),
        "seaborn": core.package_version("seaborn"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    settings = {
        "model": "MLP",
        "backbone": FIXED_BACKBONE,
        "betas": [float(value) for value in args.betas],
        "focal_gammas": [float(value) for value in args.focal_gammas],
        "training_seeds": [int(seed) for seed in args.seeds],
        "fold_count": FOLD_COUNT,
        "fold_seed": int(args.fold_seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "batching": "ordinary_shuffled",
        "validation_used_during_training": False,
        "checkpointing": "none_fixed_epoch_budget",
        "decision_policy": "raw_argmax",
        "ctgan": False,
        "minority_guaranteed_batches": False,
        "score_scaling": False,
        "deterministic_ops": bool(args.deterministic_ops),
        "library_versions": library_versions,
    }
    source_fingerprint = core.fingerprint_files(required_paths)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "settings": settings,
        "configurations": configs,
        "folds": fold_protocol_rows(folds),
        "source_and_data_fingerprint": source_fingerprint,
    }
    experiment_key = stable_hash(identity, 12)
    prefix = args.name_prefix.strip()
    stem = f"{prefix}_{experiment_key}"
    results_dir = repo_root / "results"
    run_dir = results_dir / f"{stem}_runs"
    log_dir = results_dir / f"{stem}_logs"
    cache_dir = results_dir / f"{stem}_fold_cache"
    oof_dir = results_dir / f"{stem}_oof"
    protocol_path = results_dir / f"{stem}_protocol.json"
    folds_path = results_dir / f"{stem}_folds.csv"
    plan_path = results_dir / f"{stem}_plan.csv"

    fold_cache_paths: Dict[int, tuple[Path, Path]] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        fold_cache_paths[fold_id] = (
            cache_dir / f"fold_{fold_id + 1}.npz",
            cache_dir / f"fold_{fold_id + 1}_metadata.json",
        )

    plans: List[Dict[str, Any]] = []
    for config in configs:
        for seed in args.seeds:
            for fold in folds:
                fold_id = int(fold["fold_id"])
                cache_path, cache_metadata_path = fold_cache_paths[fold_id]
                run_name = f"{stem}_{config['config_id']}_s{int(seed)}_f{fold_id + 1}"
                plans.append(
                    {
                        **config,
                        "experiment_key": experiment_key,
                        "seed": int(seed),
                        "fold_id": fold_id,
                        "fold_number": fold_id + 1,
                        "assigned_gpu": gpus[fold_id],
                        "deterministic_ops_requested": bool(args.deterministic_ops),
                        "run_name": run_name,
                        "train_indices_sha256": fold["train_indices_sha256"],
                        "validation_indices_sha256": fold["validation_indices_sha256"],
                        "cache_path": str(cache_path),
                        "cache_metadata_path": str(cache_metadata_path),
                        "result_path": str(run_dir / f"{run_name}.json"),
                        "log_path": str(log_dir / f"{run_name}.log"),
                        "status": "pending",
                    }
                )

    expected_configurations = len(args.betas) * len(args.focal_gammas)
    expected_fits = expected_configurations * len(args.seeds) * FOLD_COUNT
    if len(configs) != expected_configurations or len(plans) != expected_fits:
        raise RuntimeError("Experiment plan count is inconsistent.")
    plan_keys = {
        (plan["config_key"], int(plan["seed"]), int(plan["fold_id"])) for plan in plans
    }
    if len(plan_keys) != len(plans):
        raise RuntimeError("Experiment plan contains duplicate fits.")

    print("MLP focal-loss Stage-1 sweep")
    print(f"Experiment key: {experiment_key}")
    print(f"GPUs (fold 1..4): {gpus}")
    print(f"Betas: {args.betas}")
    print(f"Focal gammas: {args.focal_gammas}")
    print(f"Training seeds: {args.seeds}")
    print(f"Fixed folds: {FOLD_COUNT} (fold seed {args.fold_seed})")
    print(f"Configurations: {len(configs)}")
    print(f"Fits: {len(plans)}")
    print("Training: fixed epochs, ordinary shuffled batches")
    print("Evaluation: pooled KDDTrain+ out-of-fold raw argmax")
    print("CTGAN/minority batches/score scaling: OFF")
    print("KDDTest+ accessed: NO")

    if args.dry_run:
        shown = plans if args.print_all_commands else plans[:8]
        for plan in shown:
            command = build_worker_command(
                script_path, plan, args, attempt_id="dry_run"
            )
            print(
                f"[GPU {plan['assigned_gpu']} -> fold {plan['fold_number']}] "
                f"{shlex.join(command)}"
            )
        if len(shown) < len(plans):
            print(f"... {len(plans) - len(shown)} more commands")
        print("Dry run complete; no output files were written.")
        return

    validate_parent_dependencies(parser)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing four leakage-free fold caches...", flush=True)
    cache_metadata: Dict[int, Dict[str, Any]] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        cache_path, metadata_path = fold_cache_paths[fold_id]
        cache_metadata[fold_id] = prepare_fold_cache(
            raw_train,
            master_labels,
            fold,
            cache_path,
            metadata_path,
            experiment_key,
        )
        print(
            f"  fold {fold_id + 1}: train={len(fold['train_indices'])}, "
            f"validation={len(fold['validation_indices'])}, "
            f"validation U2R={int(fold['validation_counts'][3])}",
            flush=True,
        )

    folds_frame = pd.DataFrame(
        {
            "row_index": np.arange(len(master_labels), dtype=np.int64),
            "class_id": master_labels,
            "fold_id": fold_ids,
            "fold_number": fold_ids + 1,
        }
    )
    core.atomic_csv(folds_path, folds_frame)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "title": "MLP class-balanced focal-loss four-fold tuning",
        "source_and_data_fingerprint": source_fingerprint,
        "kddtrain_sha256": core.sha256_file(train_path),
        "kddtest_accessed": False,
        "no_ctgan": True,
        "synthetic_rows": 0,
        "settings": settings,
        "folds": fold_protocol_rows(folds),
        "fold_assignment_path": str(folds_path),
        "fold_assignment_sha256": core.sha256_file(folds_path),
        "fold_cache_metadata": cache_metadata,
        "configurations": configs,
        "expected_configurations": len(configs),
        "expected_fits": len(plans),
        "expected_seed_level_oof_rows": len(configs) * len(args.seeds),
        "expected_ranking_rows": len(configs),
        "preprocessing_protocol": (
            "encoder and MinMax scaler fitted independently on each outer "
            "training partition only"
        ),
        "training_protocol": (
            "25 fixed epochs by default; ordinary shuffled batches; outer "
            "validation fold is not passed to model.fit and does not select an epoch"
        ),
        "oof_protocol": (
            "for each beta/gamma and training seed, four mutually exclusive "
            "validation predictions are restored to original row order; metrics "
            "are computed once on all KDDTrain+ rows"
        ),
        "aggregation_protocol": (
            "one pooled OOF metric vector per training seed, followed by mean "
            "and sample standard deviation across seeds"
        ),
        "ranking_rule": (
            "descending mean Rare Macro-F1; then descending mean Macro-F1; "
            "then lower gamma, lower beta, and config key"
        ),
        "rare_macro_f1_definition": "(R2L F1 + U2R F1) / 2",
        "final_test_policy": (
            "KDDTest+ is not accessed by this tuning script; evaluate it only "
            "after beta and gamma are frozen"
        ),
    }
    core.atomic_json(protocol_path, protocol)
    core.atomic_csv(plan_path, pd.DataFrame(plans))
    print(f"Protocol: {protocol_path}")
    print(f"Plan: {plan_path}")

    print_lock = threading.Lock()
    state_lock = threading.Lock()
    stop_event = threading.Event()
    statuses: Dict[str, str] = {}
    runtimes: Dict[str, float] = {}
    failures: List[str] = []
    pending_plans: List[Dict[str, Any]] = []
    for plan in plans:
        run_name = str(plan["run_name"])
        if not args.rerun and result_is_complete(Path(plan["result_path"]), plan):
            statuses[run_name] = "skipped_complete"
            runtimes[run_name] = 0.0
        else:
            pending_plans.append(plan)
    completed_counter = len(plans) - len(pending_plans)
    plans_by_fold = {
        fold_id: [plan for plan in pending_plans if int(plan["fold_id"]) == fold_id]
        for fold_id in range(FOLD_COUNT)
    }
    if completed_counter:
        print(
            f"Resume check: {completed_counter} verified fits already complete; "
            f"{len(pending_plans)} remain.",
            flush=True,
        )
    print(
        f"Overall progress: {progress_bar(completed_counter, len(plans))}",
        flush=True,
    )

    def execute_plan(
        gpu: str,
        fold_id: int,
        plan: Dict[str, Any],
    ) -> tuple[str, float, str | None]:
        run_name = str(plan["run_name"])
        result_path = Path(plan["result_path"])
        attempt_id = hashlib.sha256(
            f"{run_name}:{time.time_ns()}:{os.getpid()}".encode("utf-8")
        ).hexdigest()[:16]
        command = build_worker_command(script_path, plan, args, attempt_id)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["EXPERIMENT_GPU_ID"] = gpu
        environment["PYTHONHASHSEED"] = str(plan["seed"])
        environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        log_path = Path(plan["log_path"])
        with print_lock:
            print(
                f"[GPU {gpu} | fold {fold_id + 1}] START {run_name}",
                flush=True,
            )
        started = time.perf_counter()
        try:
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
            runtime = time.perf_counter() - started
            complete = result_is_complete(result_path, plan, attempt_id)
            if completed.returncode == 0 and complete:
                return "completed", runtime, None
            return (
                "failed",
                runtime,
                f"{run_name}: exit={completed.returncode}, log={log_path}",
            )
        except Exception as error:  # Preserve resumability for controller failures.
            runtime = time.perf_counter() - started
            return (
                "failed",
                runtime,
                f"{run_name}: controller error={error!r}, log={log_path}",
            )

    def gpu_worker(gpu: str, fold_id: int) -> None:
        nonlocal completed_counter
        for plan in plans_by_fold[fold_id]:
            if stop_event.is_set():
                return
            run_name = str(plan["run_name"])
            try:
                status, runtime, failure = execute_plan(gpu, fold_id, plan)
            except Exception as error:  # Catch failures before log creation too.
                status = "failed"
                runtime = 0.0
                failure = f"{run_name}: controller error={error!r}"
            with state_lock:
                statuses[run_name] = status
                runtimes[run_name] = runtime
                if status == "failed":
                    failures.append(failure or f"{run_name}: unknown failure")
                    stop_event.set()
                else:
                    completed_counter += 1
            with print_lock:
                with state_lock:
                    progress = completed_counter
                print(
                    f"[GPU {gpu} | fold {fold_id + 1}] {status.upper()} "
                    f"{run_name} ({runtime / 60.0:.1f} min)\n"
                    f"Overall progress: {progress_bar(progress, len(plans))}",
                    flush=True,
                )

    with ThreadPoolExecutor(max_workers=FOLD_COUNT) as executor:
        futures = [
            executor.submit(gpu_worker, gpus[fold_id], fold_id)
            for fold_id in range(FOLD_COUNT)
        ]
        for future in futures:
            future.result()

    updated_plans = [
        {
            **plan,
            "status": statuses.get(plan["run_name"], "not_started"),
            "runtime_seconds": runtimes.get(plan["run_name"]),
        }
        for plan in plans
    ]
    core.atomic_csv(plan_path, pd.DataFrame(updated_plans))
    if failures:
        print(f"\nFailed fits: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(
            "Sweep stopped after a failed fit. Fix the logged error and rerun "
            "without --rerun; verified completed fits will be skipped."
        )

    fold_frame, seed_frame, summary = aggregate_results(
        plans,
        configs,
        args.seeds,
        master_labels,
        fold_ids,
        oof_dir,
    )
    fold_runs_path = results_dir / f"{stem}_fold_runs.csv"
    seed_metrics_path = results_dir / f"{stem}_seed_metrics.csv"
    summary_path = results_dir / f"{stem}_summary.csv"
    formatted_path = results_dir / f"{stem}_summary_formatted.csv"
    text_path = results_dir / f"{stem}_summary.txt"
    best_path = results_dir / f"{stem}_best_config.json"
    latest_path = results_dir / f"{prefix}_latest.json"
    core.atomic_csv(fold_runs_path, fold_frame)
    core.atomic_csv(seed_metrics_path, seed_frame)
    core.atomic_csv(summary_path, summary)
    pretty = formatted_summary(summary)
    core.atomic_csv(formatted_path, pretty)
    text_path.write_text(
        "MLP focal-loss Stage-1 sweep\n"
        f"Experiment key: {experiment_key}\n"
        f"Seeds: {args.seeds}\n"
        f"Fixed folds: {FOLD_COUNT}, fold seed: {args.fold_seed}\n"
        "Metric unit: percentage; variability: sample SD across training seeds\n"
        "Ranking: Rare Macro-F1, Macro-F1, lower gamma, lower beta\n"
        "KDDTest+ accessed: NO\n\n" + pretty.to_string(index=False) + "\n",
        encoding="utf-8",
    )
    winner = summary.iloc[0]
    best = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "rank": 1,
        "config_key": str(winner["config_key"]),
        "config_id": str(winner["config_id"]),
        "beta": float(winner["beta"]),
        "focal_gamma": float(winner["focal_gamma"]),
        "training_seeds": [int(seed) for seed in args.seeds],
        "fold_count": FOLD_COUNT,
        "runs": int(winner["runs"]),
        "total_fits": int(winner["total_fits"]),
        "selection_data": "KDDTrain+ pooled out-of-fold predictions only",
        "kddtest_accessed": False,
        "ranking_rule": (
            "highest mean Rare Macro-F1; then highest mean Macro-F1; "
            "then lower gamma and lower beta"
        ),
        "metrics": {
            metric: {
                "mean": float(winner[f"{metric}_mean"]),
                "sample_std": float(winner[f"{metric}_std"]),
            }
            for metric in METRICS
        },
        "summary_path": str(summary_path),
        "seed_metrics_path": str(seed_metrics_path),
    }
    core.atomic_json(best_path, best)
    latest = {
        "experiment_key": experiment_key,
        "protocol": str(protocol_path),
        "plan": str(plan_path),
        "fold_runs": str(fold_runs_path),
        "seed_metrics": str(seed_metrics_path),
        "summary": str(summary_path),
        "formatted_summary": str(formatted_path),
        "readable_summary": str(text_path),
        "best_config": str(best_path),
        "oof_directory": str(oof_dir),
    }
    core.atomic_json(latest_path, latest)

    print("\n=== Ranked MLP focal settings ===")
    print(pretty.to_string(index=False))
    print(f"\nBest beta: {best['beta']}")
    print(f"Best focal gamma: {best['focal_gamma']}")
    print(f"Fold runs: {fold_runs_path}")
    print(f"Seed-level pooled OOF metrics: {seed_metrics_path}")
    print(f"Numeric ranking: {summary_path}")
    print(f"Readable ranking: {text_path}")
    print(f"Best configuration: {best_path}")
    print(f"Latest-results pointer: {latest_path}")


if __name__ == "__main__":
    main()
