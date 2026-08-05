"""Evaluate raw or score-scaled neural models from leakage-free OOF predictions.

The shared runner supports pure cross-entropy baselines, batch-only ablations,
and the focal-loss plus minority-batch pipeline.  For score-scaling searches it
trains one model for each of four fixed folds and three seeds, restores each
seed's folds to original row order, and searches R2L/U2R coefficient pairs from
the saved probabilities without retraining.

KDDTest+ and synthetic data are never accessed. The held-out fold is not
passed to model.fit: training uses a fixed epoch budget with no validation
checkpoint or early stopping.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

import run_no_ctgan_model_ablation_4gpu as core
import tune_conv1d_focal_cv_4gpu as conv1d_stage1
import tune_conv2d_focal_cv_4gpu as conv2d_stage1
import tune_mlp_focal_cv_4gpu as mlp_stage1
import tune_transformer_focal_cv_4gpu as transformer_stage1


SCHEMA_VERSION = 1
FOLD_COUNT = 4
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FOLD_SEED = 0
DEFAULT_BETA = 0.99
DEFAULT_FOCAL_GAMMA = 0.50
DEFAULT_MACRO_F1_RETENTION = 0.90
DEFAULT_MINORITY_PRECISION_RETENTION = 0.80
TRAINING_MODES = {"baseline_ce", "baseline_batch", "focal_balanced"}
BALANCED_BATCH_MODES = {"baseline_batch", "focal_balanced"}
DEFAULT_COEFFICIENTS = [
    0.10,
    0.25,
    0.40,
    0.55,
    0.70,
    0.85,
    1.00,
    1.15,
    1.30,
    1.45,
    1.60,
    1.75,
    1.90,
]
BASELINE_SCALING_COEFFICIENTS = [
    0.10,
    0.25,
    0.40,
    0.55,
    0.70,
    0.85,
    1.00,
    1.15,
    1.30,
    1.45,
    1.60,
    1.75,
    1.90,
    2.20,
    2.50,
    3.00,
    3.50,
    4.00,
    4.50,
    5.00,
    6.00,
    7.00,
    8.00,
    10.00,
]
ARCHITECTURE_DEFAULTS = {
    "conv1d": {
        "label": "Conv1D",
        "focal_gamma": 0.25,
        "name_prefix": "conv1d_balanced_score_scaling",
        "baseline_name_prefix": "conv1d_baseline_cv",
        "batch_baseline_name_prefix": "conv1d_batch_baseline_cv",
        "stage1": conv1d_stage1,
    },
    "conv2d": {
        "label": "Conv2D",
        "focal_gamma": DEFAULT_FOCAL_GAMMA,
        "name_prefix": "conv2d_balanced_score_scaling",
        "baseline_name_prefix": "conv2d_baseline_cv",
        "batch_baseline_name_prefix": "conv2d_batch_baseline_cv",
        "stage1": conv2d_stage1,
    },
    "mlp": {
        "label": "MLP",
        "focal_gamma": 0.25,
        "name_prefix": "mlp_balanced_score_scaling",
        "baseline_name_prefix": "mlp_baseline_cv",
        "batch_baseline_name_prefix": "mlp_batch_baseline_cv",
        "stage1": mlp_stage1,
    },
    "transformer": {
        "label": "Transformer",
        "focal_gamma": 0.75,
        "name_prefix": "transformer_balanced_score_scaling",
        "baseline_name_prefix": "transformer_baseline_cv",
        "batch_baseline_name_prefix": "transformer_batch_baseline_cv",
        "stage1": transformer_stage1,
    },
}
stage1 = conv2d_stage1
FIXED_BACKBONE = dict(conv2d_stage1.FIXED_BACKBONE)
METRICS = list(core.METRICS)
SCORING_METRICS = [*METRICS, "minority_recall"]


def configure_architecture(architecture: str) -> None:
    """Select the matching Stage-1 helpers and fixed backbone."""
    global stage1, FIXED_BACKBONE
    configuration = ARCHITECTURE_DEFAULTS[architecture]
    stage1 = configuration["stage1"]
    FIXED_BACKBONE = dict(stage1.FIXED_BACKBONE)


def architecture_label(architecture: str) -> str:
    return str(ARCHITECTURE_DEFAULTS[architecture]["label"])


def architecture_model_key(architecture: str) -> str:
    return f"fixed_{architecture}"


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[------------------------------]   0.0% (0/0)"
    completed = max(0, min(int(completed), int(total)))
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {100.0 * completed / total:6.2f}% ({completed}/{total})"


def resolve_cuda_tokens(gpus: Sequence[str]) -> Dict[str, str]:
    """Map logical worker IDs to tokens in a parent CUDA allocation."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    try:
        logical = [int(gpu) for gpu in gpus]
    except ValueError:
        logical = []
    if (
        tokens
        and len(logical) == len(gpus)
        and all(0 <= index < len(tokens) for index in logical)
    ):
        return {gpu: tokens[index] for gpu, index in zip(gpus, logical, strict=True)}
    return {gpu: gpu for gpu in gpus}


def worker_result_is_complete(
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
            "training_key": plan["training_key"],
            "run_name": plan["run_name"],
            "seed": int(plan["seed"]),
            "fold_id": int(plan["fold_id"]),
            "train_indices_sha256": plan["train_indices_sha256"],
            "validation_indices_sha256": plan["validation_indices_sha256"],
            "model": architecture_model_key(str(plan["architecture"])),
            "model_parameters": FIXED_BACKBONE["expected_parameters"],
            "training_mode": plan["training_mode"],
            "cb_beta": (
                float(plan["cb_beta"])
                if plan["training_mode"] == "focal_balanced"
                else None
            ),
            "focal_gamma": (
                float(plan["focal_gamma"])
                if plan["training_mode"] == "focal_balanced"
                else None
            ),
            "minority_per_batch": (
                int(plan["minority_per_batch"])
                if plan["training_mode"] in BALANCED_BATCH_MODES
                else 0
            ),
            "minority_guaranteed_batches": (
                plan["training_mode"] in BALANCED_BATCH_MODES
            ),
            "validation_used_during_training": False,
            "score_scaling_used": False,
            "synthetic_rows": 0,
            "kddtest_accessed": False,
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
        return stage1.prediction_artifact_is_complete(prediction_path, result)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def run_training_worker(args: argparse.Namespace) -> None:
    import tensorflow as tf

    if args.training_mode == "focal_balanced":
        from cnn_gan_foc import ClassBalancedFocalLoss  # type: ignore
    if args.training_mode in BALANCED_BATCH_MODES:
        from cnn_opt import BalancedBatchSequence  # type: ignore

    if args.architecture == "conv1d":
        from cnn_opt_1d_4gpu import build_opt_cnn_1d as build_model  # type: ignore
    elif args.architecture == "conv2d":
        from cnn_opt import build_opt_cnn as build_model  # type: ignore
    elif args.architecture == "mlp":
        from cnn_opt_1d_4gpu import build_opt_mlp as build_model  # type: ignore
    else:
        from cnn_opt_1d_4gpu import (  # type: ignore
            build_vanilla_transformer as build_model,
        )

    cache_path = Path(args.worker_cache_path)
    cache_metadata_path = Path(args.worker_cache_metadata_path)
    cache_metadata = core.read_json(cache_metadata_path)
    if cache_metadata.get("cache_sha256") != core.sha256_file(cache_path):
        raise RuntimeError("Fold cache hash does not match its metadata.")
    if cache_metadata.get("experiment_key") != args.training_key:
        raise RuntimeError("Fold cache belongs to a different training experiment.")
    if int(cache_metadata.get("fold_id", -1)) != args.worker_fold_id:
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

    seed = int(args.worker_seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    if args.training_mode != "focal_balanced":
        alpha = None
        alpha_counts = np.bincount(y_train, minlength=5).astype(np.int64)
        loss: Any = tf.keras.losses.SparseCategoricalCrossentropy()
    else:
        alpha, alpha_counts = core.effective_number_alpha(
            y_train,
            beta=float(args.cb_beta),
            num_classes=5,
        )
        loss = ClassBalancedFocalLoss(alpha=alpha, gamma=float(args.focal_gamma))
    if args.architecture == "mlp":
        model = build_model(
            loss=loss,
            dense_units=FIXED_BACKBONE["dense_units"],
            dropout1=FIXED_BACKBONE["dropout1"],
            dropout2=FIXED_BACKBONE["dropout2"],
            use_batch_norm=FIXED_BACKBONE["batch_norm"],
        )
    elif args.architecture == "transformer":
        model = build_model(
            loss=loss,
            d_model=FIXED_BACKBONE["d_model"],
            num_heads=FIXED_BACKBONE["num_heads"],
            num_blocks=FIXED_BACKBONE["blocks"],
            ff_dim=FIXED_BACKBONE["ff_dim"],
            dense_units=FIXED_BACKBONE["dense_units"],
            transformer_dropout=FIXED_BACKBONE["dropout"],
            head_dropout=FIXED_BACKBONE["head_dropout"],
        )
    else:
        model = build_model(
            loss=loss,
            groups=FIXED_BACKBONE["groups"],
            base_filters=FIXED_BACKBONE["base_filters"],
            dense_units=FIXED_BACKBONE["dense_units"],
            dropout1=FIXED_BACKBONE["dropout1"],
            dropout2=FIXED_BACKBONE["dropout2"],
            use_batch_norm=FIXED_BACKBONE["batch_norm"],
            use_residual=FIXED_BACKBONE["residual"],
        )
    parameter_count = int(model.count_params())
    if parameter_count != FIXED_BACKBONE["expected_parameters"]:
        raise RuntimeError(
            f"{architecture_label(args.architecture)} parameter count changed: expected "
            f"{FIXED_BACKBONE['expected_parameters']}, got {parameter_count}."
        )

    if args.architecture == "mlp":
        X_train = X_train_flat
        X_validation = X_validation_flat
    elif args.architecture == "conv2d":
        X_train = X_train_flat.reshape(-1, 11, 11, 1)
        X_validation = X_validation_flat.reshape(-1, 11, 11, 1)
    else:
        X_train = X_train_flat.reshape(-1, 121, 1)
        X_validation = X_validation_flat.reshape(-1, 121, 1)
    started = time.perf_counter()
    if args.training_mode not in BALANCED_BATCH_MODES:
        history = model.fit(
            X_train,
            y_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            shuffle=True,
            verbose=int(args.fit_verbose),
        )
        steps_per_epoch = int(np.ceil(len(y_train) / int(args.batch_size)))
    else:
        training_data = BalancedBatchSequence(
            X_train,
            y_train,
            batch_size=int(args.batch_size),
            minority_per_batch=int(args.minority_per_batch),
            seed=seed,
        )
        history = model.fit(
            training_data,
            epochs=int(args.epochs),
            verbose=int(args.fit_verbose),
        )
        steps_per_epoch = int(len(training_data))
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
    result = {
        "schema_version": SCHEMA_VERSION,
        "training_key": args.training_key,
        "attempt_id": args.worker_attempt_id,
        "run_name": args.worker_run_name,
        "seed": seed,
        "fold_id": int(args.worker_fold_id),
        "fold_number": int(args.worker_fold_id) + 1,
        "train_indices_sha256": cache_metadata["train_indices_sha256"],
        "validation_indices_sha256": cache_metadata["validation_indices_sha256"],
        "train_counts": cache_metadata["train_counts"],
        "validation_counts": cache_metadata["validation_counts"],
        "feature_order_sha256": cache_metadata["feature_order_sha256"],
        "scaler_state_sha256": cache_metadata["scaler_state_sha256"],
        "fold_cache_sha256": cache_metadata["cache_sha256"],
        "model": architecture_model_key(args.architecture),
        "architecture": args.architecture,
        "training_mode": args.training_mode,
        "backbone": FIXED_BACKBONE,
        "model_parameters": parameter_count,
        "loss": (
            "sparse_categorical_crossentropy"
            if args.training_mode != "focal_balanced"
            else "class_balanced_focal"
        ),
        "class_weighting": (
            "none"
            if args.training_mode != "focal_balanced"
            else "effective_number_from_outer_training_fold"
        ),
        "cb_beta": (
            float(args.cb_beta)
            if args.training_mode == "focal_balanced"
            else None
        ),
        "focal_gamma": (
            float(args.focal_gamma)
            if args.training_mode == "focal_balanced"
            else None
        ),
        "alpha": (
            None if alpha is None else np.asarray(alpha, dtype=float).tolist()
        ),
        "alpha_counts": np.asarray(alpha_counts, dtype=int).tolist(),
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "batch_size": int(args.batch_size),
        "batching": (
            "ordinary_shuffled"
            if args.training_mode not in BALANCED_BATCH_MODES
            else "minority_guaranteed_with_replacement"
        ),
        "minority_per_batch": (
            0
            if args.training_mode not in BALANCED_BATCH_MODES
            else int(args.minority_per_batch)
        ),
        "minority_guaranteed_batches": (
            args.training_mode in BALANCED_BATCH_MODES
        ),
        "steps_per_epoch": steps_per_epoch,
        "validation_used_during_training": False,
        "checkpointing": "none_fixed_epoch_budget",
        "ctgan_used": False,
        "synthetic_rows": 0,
        "decision_policy": "raw_argmax_for_saved_oof_probabilities",
        "score_scaling_used": False,
        "dataset": "KDDTrain+ only",
        "evaluation_partition": "one fixed outer validation fold",
        "kddtest_accessed": False,
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
        "--architecture",
        str(args.architecture),
        "--training-mode",
        str(args.training_mode),
        "--worker-mode",
        "train",
        "--worker-run-name",
        str(plan["run_name"]),
        "--worker-result-path",
        str(plan["result_path"]),
        "--worker-attempt-id",
        attempt_id,
        "--worker-seed",
        str(plan["seed"]),
        "--worker-fold-id",
        str(plan["fold_id"]),
        "--worker-cache-path",
        str(plan["cache_path"]),
        "--worker-cache-metadata-path",
        str(plan["cache_metadata_path"]),
        "--training-key",
        str(plan["training_key"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--fit-verbose",
        str(args.fit_verbose),
    ]
    if args.training_mode == "focal_balanced":
        command.extend(
            [
                "--cb-beta",
                str(args.cb_beta),
                "--focal-gamma",
                str(args.focal_gamma),
            ]
        )
    if args.training_mode in BALANCED_BATCH_MODES:
        command.extend(
            ["--minority-per-batch", str(args.minority_per_batch)]
        )
    if args.deterministic_ops:
        command.append("--deterministic-ops")
    return command


def metrics_from_confusions(confusions: np.ndarray) -> pd.DataFrame:
    """Calculate reported metrics from P x 5 x 5 confusion matrices."""
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
    numerator = correct * totals - np.sum(true_totals * predicted_totals, axis=1)
    denominator = np.sqrt(
        (totals**2 - np.sum(predicted_totals**2, axis=1))
        * (totals**2 - np.sum(true_totals**2, axis=1))
    )
    mcc = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0,
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
    """Evaluate every coefficient pair in memory-bounded sample chunks."""
    y_true = np.asarray(labels, dtype=np.int64)
    y_proba = np.asarray(probabilities, dtype=np.float64)
    r2l = np.asarray(pair_r2l, dtype=np.float64)
    u2r = np.asarray(pair_u2r, dtype=np.float64)
    if y_proba.shape != (len(y_true), 5):
        raise ValueError(
            f"Probability/label mismatch: {y_proba.shape} versus {len(y_true)}."
        )
    if not np.isfinite(y_proba).all():
        raise ValueError("OOF probabilities contain a non-finite value.")
    if (
        r2l.shape != u2r.shape
        or r2l.ndim != 1
        or np.any(r2l <= 0.0)
        or np.any(u2r <= 0.0)
    ):
        raise ValueError("Score-pair arrays must align and be positive.")

    pair_count = len(r2l)
    pair_offsets = (25 * np.arange(pair_count, dtype=np.int64))[None, :]
    counts = np.zeros((pair_count, 25), dtype=np.int64)
    majority_classes = np.asarray([0, 1, 4], dtype=np.int8)
    for start in range(0, len(y_true), chunk_size):
        stop = min(len(y_true), start + chunk_size)
        chunk = y_proba[start:stop]
        chunk_labels = y_true[start:stop]
        majority_probabilities = chunk[:, [0, 1, 4]]
        majority_choice = np.argmax(majority_probabilities, axis=1)
        base_predictions = majority_classes[majority_choice]
        base_scores = majority_probabilities[np.arange(len(chunk)), majority_choice]
        predictions = np.broadcast_to(
            base_predictions[:, None], (len(chunk), pair_count)
        ).copy()
        best_scores = np.broadcast_to(
            base_scores[:, None], (len(chunk), pair_count)
        ).copy()

        r2l_scores = chunk[:, 2, None] / r2l[None, :]
        r2l_wins = (r2l_scores > best_scores) | (
            (r2l_scores == best_scores) & (predictions > 2)
        )
        np.copyto(best_scores, r2l_scores, where=r2l_wins)
        predictions[r2l_wins] = 2
        u2r_scores = chunk[:, 3, None] / u2r[None, :]
        u2r_wins = (u2r_scores > best_scores) | (
            (u2r_scores == best_scores) & (predictions > 3)
        )
        predictions[u2r_wins] = 3

        codes = chunk_labels[:, None] * 5 + predictions.astype(np.int64)
        linear_codes = codes + pair_offsets
        counts += np.bincount(linear_codes.ravel(), minlength=pair_count * 25).reshape(
            pair_count, 25
        )
    return counts.reshape(pair_count, 5, 5)


def assemble_oof_predictions(
    plans: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
    master_labels: np.ndarray,
    fold_ids: np.ndarray,
    oof_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[int, Path]]:
    """Restore four held-out folds to one complete OOF matrix per seed."""
    results: Dict[tuple[int, int], Dict[str, Any]] = {}
    fold_rows: List[Dict[str, Any]] = []
    for plan in plans:
        result_path = Path(plan["result_path"])
        if not worker_result_is_complete(result_path, plan):
            raise RuntimeError(f"Incomplete fit during aggregation: {result_path}")
        result = core.read_json(result_path)
        key = (int(plan["seed"]), int(plan["fold_id"]))
        if key in results:
            raise RuntimeError(f"Duplicate completed fit: {key}.")
        results[key] = result
        fold_rows.append(
            {
                "seed": key[0],
                "fold_id": key[1],
                "fold_number": key[1] + 1,
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

    oof_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: List[Dict[str, Any]] = []
    oof_paths: Dict[int, Path] = {}
    row_count = len(master_labels)
    for seed in seeds:
        probabilities = np.full((row_count, 5), np.nan, dtype=np.float32)
        coverage = np.zeros(row_count, dtype=np.uint8)
        source_hashes: List[str] = []
        for fold_id in range(FOLD_COUNT):
            result = results[(int(seed), fold_id)]
            with np.load(result["prediction_path"], allow_pickle=False) as artifact:
                indices = np.asarray(artifact["validation_indices"], dtype=np.int64)
                labels = np.asarray(artifact["validation_labels"], dtype=np.int64)
                fold_probabilities = np.asarray(
                    artifact["validation_probabilities"], dtype=np.float32
                )
            if np.any(coverage[indices] != 0):
                raise RuntimeError(f"OOF overlap for seed {seed}.")
            if not np.array_equal(labels, master_labels[indices]):
                raise RuntimeError("OOF labels do not match original rows.")
            probabilities[indices] = fold_probabilities
            coverage[indices] += 1
            source_hashes.append(str(result["prediction_sha256"]))
        if not np.all(coverage == 1):
            raise RuntimeError(
                f"Invalid OOF coverage for seed {seed}: "
                f"min={coverage.min()}, max={coverage.max()}."
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Assembled OOF probabilities are not finite.")
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
        metrics = core.calculate_metrics(master_labels, predictions)
        lineage = hashlib.sha256("\n".join(source_hashes).encode("ascii")).hexdigest()
        oof_path = oof_dir / f"seed_{int(seed)}_oof_probabilities.npz"
        core.atomic_npz(
            oof_path,
            row_indices=np.arange(row_count, dtype=np.int64),
            fold_ids=np.asarray(fold_ids, dtype=np.int64),
            labels=np.asarray(master_labels, dtype=np.int64),
            probabilities=probabilities,
            raw_predictions=predictions,
        )
        oof_paths[int(seed)] = oof_path
        seed_rows.append(
            {
                "seed": int(seed),
                "completed_folds": FOLD_COUNT,
                "oof_rows": row_count,
                "coverage_min": int(coverage.min()),
                "coverage_max": int(coverage.max()),
                "source_predictions_sha256": lineage,
                "oof_path": str(oof_path),
                "oof_sha256": core.sha256_file(oof_path),
                **metrics,
            }
        )
    return (
        pd.DataFrame(fold_rows).sort_values(["seed", "fold_id"]),
        pd.DataFrame(seed_rows).sort_values("seed"),
        oof_paths,
    )


def score_oof_probabilities(
    oof_paths: Dict[int, Path],
    seeds: Sequence[int],
    coefficients: Sequence[float],
    macro_f1_retention: float,
    minority_precision_retention: float,
    chunk_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Evaluate and rank one shared scaling pair across all seeds."""
    values = np.asarray(coefficients, dtype=np.float64)
    pair_r2l = np.repeat(values, len(values))
    pair_u2r = np.tile(values, len(values))
    per_seed_parts: List[pd.DataFrame] = []
    raw_macro_by_seed: Dict[int, float] = {}
    raw_r2l_precision_by_seed: Dict[int, float] = {}
    raw_u2r_precision_by_seed: Dict[int, float] = {}
    raw_index_matches = np.flatnonzero(
        np.isclose(pair_r2l, 1.0) & np.isclose(pair_u2r, 1.0)
    )
    if len(raw_index_matches) != 1:
        raise RuntimeError("Coefficient grid must contain exactly one (1, 1) pair.")
    raw_pair_index = int(raw_index_matches[0])

    for seed in seeds:
        with np.load(oof_paths[int(seed)], allow_pickle=False) as artifact:
            labels = np.asarray(artifact["labels"], dtype=np.int64)
            probabilities = np.asarray(artifact["probabilities"], dtype=np.float32)
            raw_predictions = np.asarray(artifact["raw_predictions"], dtype=np.int64)
        confusions = score_pair_confusions(
            labels,
            probabilities,
            pair_r2l,
            pair_u2r,
            chunk_size,
        )
        frame = metrics_from_confusions(confusions)
        frame.insert(0, "u2r_score_coefficient", pair_u2r)
        frame.insert(0, "r2l_score_coefficient", pair_r2l)
        frame.insert(0, "seed", int(seed))
        frame["minority_recall"] = (frame["r2l_recall"] + frame["u2r_recall"]) / 2.0
        raw_metrics = core.calculate_metrics(labels, raw_predictions)
        raw_macro_by_seed[int(seed)] = float(raw_metrics["macro_f1"])
        raw_r2l_precision_by_seed[int(seed)] = float(raw_metrics["r2l_precision"])
        raw_u2r_precision_by_seed[int(seed)] = float(raw_metrics["u2r_precision"])
        for metric in METRICS:
            if not np.isclose(
                float(frame.iloc[raw_pair_index][metric]),
                float(raw_metrics[metric]),
                atol=1e-12,
                rtol=1e-12,
            ):
                raise RuntimeError(
                    f"Vectorized raw metric mismatch for seed {seed}: {metric}."
                )
        frame["raw_macro_f1_for_seed"] = raw_macro_by_seed[int(seed)]
        frame["raw_r2l_precision_for_seed"] = raw_r2l_precision_by_seed[int(seed)]
        frame["raw_u2r_precision_for_seed"] = raw_u2r_precision_by_seed[int(seed)]
        frame["macro_f1_retention_ratio"] = (
            frame["macro_f1"] / raw_macro_by_seed[int(seed)]
        )
        frame["r2l_precision_retention_ratio"] = (
            frame["r2l_precision"] / raw_r2l_precision_by_seed[int(seed)]
            if raw_r2l_precision_by_seed[int(seed)] > 0.0
            else 1.0
        )
        frame["u2r_precision_retention_ratio"] = (
            frame["u2r_precision"] / raw_u2r_precision_by_seed[int(seed)]
            if raw_u2r_precision_by_seed[int(seed)] > 0.0
            else 1.0
        )
        frame["meets_seed_macro_f1_retention"] = (
            frame["macro_f1_retention_ratio"] >= macro_f1_retention - 1e-15
        )
        frame["meets_seed_r2l_precision_retention"] = (
            frame["r2l_precision_retention_ratio"]
            >= minority_precision_retention - 1e-15
        )
        frame["meets_seed_u2r_precision_retention"] = (
            frame["u2r_precision_retention_ratio"]
            >= minority_precision_retention - 1e-15
        )
        frame["meets_seed_all_guards"] = (
            frame["meets_seed_macro_f1_retention"]
            & frame["meets_seed_r2l_precision_retention"]
            & frame["meets_seed_u2r_precision_retention"]
        )
        per_seed_parts.append(frame)

    per_seed = pd.concat(per_seed_parts, ignore_index=True)
    metric_columns = list(SCORING_METRICS)
    summary_rows: List[Dict[str, Any]] = []
    for (r2l_value, u2r_value), group in per_seed.groupby(
        ["r2l_score_coefficient", "u2r_score_coefficient"], sort=True
    ):
        observed_seeds = sorted(group["seed"].astype(int).tolist())
        if observed_seeds != sorted(int(seed) for seed in seeds):
            raise RuntimeError("A scaling pair is missing one or more seeds.")
        row: Dict[str, Any] = {
            "r2l_score_coefficient": float(r2l_value),
            "u2r_score_coefficient": float(u2r_value),
            "runs": len(group),
            "seeds": ",".join(str(seed) for seed in observed_seeds),
            "scaling_log_distance": float(
                abs(np.log(float(r2l_value))) + abs(np.log(float(u2r_value)))
            ),
            "eligible_all_seeds": bool(group["meets_seed_all_guards"].all()),
        }
        for metric in metric_columns:
            metric_values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(metric_values.mean())
            row[f"{metric}_std"] = (
                float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    raw_mean_macro_f1 = float(np.mean(list(raw_macro_by_seed.values())))
    raw_mean_r2l_precision = float(np.mean(list(raw_r2l_precision_by_seed.values())))
    raw_mean_u2r_precision = float(np.mean(list(raw_u2r_precision_by_seed.values())))
    macro_floor = raw_mean_macro_f1 * float(macro_f1_retention)
    r2l_precision_floor = raw_mean_r2l_precision * float(minority_precision_retention)
    u2r_precision_floor = raw_mean_u2r_precision * float(minority_precision_retention)
    summary["raw_macro_f1_mean"] = raw_mean_macro_f1
    summary["raw_r2l_precision_mean"] = raw_mean_r2l_precision
    summary["raw_u2r_precision_mean"] = raw_mean_u2r_precision
    summary["macro_f1_eligibility_floor"] = macro_floor
    summary["r2l_precision_eligibility_floor"] = r2l_precision_floor
    summary["u2r_precision_eligibility_floor"] = u2r_precision_floor
    summary["macro_f1_mean_retention_ratio"] = (
        summary["macro_f1_mean"] / raw_mean_macro_f1
    )
    summary["r2l_precision_mean_retention_ratio"] = (
        summary["r2l_precision_mean"] / raw_mean_r2l_precision
        if raw_mean_r2l_precision > 0.0
        else 1.0
    )
    summary["u2r_precision_mean_retention_ratio"] = (
        summary["u2r_precision_mean"] / raw_mean_u2r_precision
        if raw_mean_u2r_precision > 0.0
        else 1.0
    )
    summary["meets_mean_macro_f1_guard"] = (
        summary["macro_f1_mean"] >= macro_floor - 1e-15
    )
    summary["meets_mean_r2l_precision_guard"] = (
        summary["r2l_precision_mean"] >= r2l_precision_floor - 1e-15
    )
    summary["meets_mean_u2r_precision_guard"] = (
        summary["u2r_precision_mean"] >= u2r_precision_floor - 1e-15
    )
    summary["eligible_mean"] = (
        summary["meets_mean_macro_f1_guard"]
        & summary["meets_mean_r2l_precision_guard"]
        & summary["meets_mean_u2r_precision_guard"]
    )
    summary["minority_recall_gap_mean"] = abs(
        summary["r2l_recall_mean"] - summary["u2r_recall_mean"]
    )
    summary = summary.sort_values(
        by=[
            "rare_f1_mean",
            "macro_f1_mean",
            "scaling_log_distance",
            "rare_f1_std",
            "r2l_score_coefficient",
            "u2r_score_coefficient",
        ],
        ascending=[
            False,
            False,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    if len(summary) != len(values) ** 2:
        raise RuntimeError("Scaling summary has the wrong number of pairs.")

    winner = summary.iloc[0]
    best = {
        "r2l_score_coefficient": float(winner["r2l_score_coefficient"]),
        "u2r_score_coefficient": float(winner["u2r_score_coefficient"]),
        "eligible_mean": bool(winner["eligible_mean"]),
        "eligible_all_seeds": bool(winner["eligible_all_seeds"]),
        "raw_macro_f1_mean": raw_mean_macro_f1,
        "raw_r2l_precision_mean": raw_mean_r2l_precision,
        "raw_u2r_precision_mean": raw_mean_u2r_precision,
        "macro_f1_eligibility_floor": macro_floor,
        "r2l_precision_eligibility_floor": r2l_precision_floor,
        "u2r_precision_eligibility_floor": u2r_precision_floor,
        "macro_f1_retention": float(macro_f1_retention),
        "minority_precision_retention": float(minority_precision_retention),
        "metrics": {
            metric: {
                "mean": float(winner[f"{metric}_mean"]),
                "sample_std": float(winner[f"{metric}_std"]),
            }
            for metric in metric_columns
        },
    }
    return per_seed, summary, best


def formatted_summary(summary: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = summary.head(top_n)
    output = selected[
        [
            "rank",
            "r2l_score_coefficient",
            "u2r_score_coefficient",
            "eligible_mean",
            "eligible_all_seeds",
            "meets_mean_macro_f1_guard",
            "meets_mean_r2l_precision_guard",
            "meets_mean_u2r_precision_guard",
        ]
    ].copy()
    labels = {
        "minimum_minority_recall": "Minimum Minority Recall",
        "minority_recall": "Mean Minority Recall",
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
            f"{100.0 * float(mean):.2f}% +/- {100.0 * float(std):.2f}%"
            for mean, std in zip(
                selected[f"{metric}_mean"],
                selected[f"{metric}_std"],
                strict=True,
            )
        ]
    return output


def add_arguments(
    parser: argparse.ArgumentParser,
    default_architecture: str = "conv2d",
    default_training_mode: str = "focal_balanced",
) -> None:
    parser.add_argument(
        "--architecture",
        choices=sorted(ARCHITECTURE_DEFAULTS),
        default=default_architecture,
    )
    parser.add_argument(
        "--training-mode",
        choices=sorted(TRAINING_MODES),
        default=default_training_mode,
        help=(
            "focal_balanced uses class-balanced focal loss and minority-guaranteed "
            "batches; baseline_batch uses ordinary cross-entropy with the same "
            "minority-guaranteed batches; baseline_ce uses ordinary cross-entropy "
            "and shuffled batches"
        ),
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--fold-seed", type=int, default=DEFAULT_FOLD_SEED)
    parser.add_argument("--cb-beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument(
        "--coefficient-values",
        type=float,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--macro-f1-retention", type=float, default=DEFAULT_MACRO_F1_RETENTION
    )
    parser.add_argument(
        "--minority-precision-retention",
        type=float,
        default=DEFAULT_MINORITY_PRECISION_RETENTION,
    )
    parser.add_argument("--score-chunk-size", type=int, default=512)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--fit-verbose", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--name-prefix", default=None)
    parser.add_argument("--deterministic-ops", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--worker-mode",
        choices=["none", "train"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-run-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-attempt-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-fold-id", type=int, default=-1, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-cache-path", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-cache-metadata-path", default="", help=argparse.SUPPRESS
    )
    parser.add_argument("--training-key", default="", help=argparse.SUPPRESS)


def validate_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds cannot be negative.")
    if args.fold_seed < 0:
        parser.error("--fold-seed cannot be negative.")
    if args.training_mode == "focal_balanced":
        if not np.isfinite(args.cb_beta) or not 0.0 < args.cb_beta < 1.0:
            parser.error("--cb-beta must be finite and strictly between 0 and 1.")
        if not np.isfinite(args.focal_gamma) or args.focal_gamma <= 0.0:
            parser.error("--focal-gamma must be finite and greater than zero.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive.")
    if args.training_mode in BALANCED_BATCH_MODES:
        if args.minority_per_batch <= 0:
            parser.error("--minority-per-batch must be positive.")
        if 2 * args.minority_per_batch > args.batch_size:
            parser.error("--batch-size must fit the guaranteed R2L and U2R samples.")
    if not 0.0 < args.macro_f1_retention <= 1.0:
        parser.error("--macro-f1-retention must be in (0, 1].")
    if not 0.0 < args.minority_precision_retention <= 1.0:
        parser.error("--minority-precision-retention must be in (0, 1].")
    if args.score_chunk_size <= 0 or args.top_n <= 0:
        parser.error("--score-chunk-size and --top-n must be positive.")
    values = [float(value) for value in args.coefficient_values]
    if not values or len(values) != len(set(values)):
        parser.error("--coefficient-values must contain unique values.")
    if any(not np.isfinite(value) or value <= 0.0 for value in values):
        parser.error("Every score coefficient must be finite and positive.")
    if sum(np.isclose(value, 1.0) for value in values) != 1:
        parser.error("The coefficient grid must contain 1.0 exactly once.")
    if args.training_mode == "baseline_batch" and (
        len(values) != 1 or not np.isclose(values[0], 1.0)
    ):
        parser.error(
            "The batch-only ablation requires --coefficient-values 1.0 "
            "(raw argmax only)."
        )
    if len(values) ** 2 > 10_000:
        parser.error("The score grid cannot exceed 10,000 pairs.")
    prefix = args.name_prefix.strip()
    if not prefix or Path(prefix).name != prefix:
        parser.error("--name-prefix must be a nonempty filename-safe name.")


def main(
    default_architecture: str = "conv2d",
    default_training_mode: str = "focal_balanced",
    default_coefficient_values: Sequence[float] | None = None,
    default_name_prefix: str | None = None,
) -> None:
    if default_architecture not in ARCHITECTURE_DEFAULTS:
        raise ValueError(f"Unsupported default architecture: {default_architecture}")
    if default_training_mode not in TRAINING_MODES:
        raise ValueError(f"Unsupported default training mode: {default_training_mode}")
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Train MLP, Conv1D, Conv2D, or Transformer OOF models as pure "
            "cross-entropy baselines, cross-entropy plus minority-batch ablations, "
            "or with the focal/batching/scaling pipeline, without accessing "
            "KDDTest+."
        )
    )
    add_arguments(parser, default_architecture, default_training_mode)
    args = parser.parse_args()
    configure_architecture(args.architecture)
    architecture_defaults = ARCHITECTURE_DEFAULTS[args.architecture]
    if args.focal_gamma is None:
        args.focal_gamma = float(architecture_defaults["focal_gamma"])
    if args.coefficient_values is None:
        if default_coefficient_values is not None:
            args.coefficient_values = [
                float(value) for value in default_coefficient_values
            ]
        else:
            args.coefficient_values = (
                [1.0]
                if args.training_mode in {"baseline_ce", "baseline_batch"}
                else list(DEFAULT_COEFFICIENTS)
            )
    if args.name_prefix is None:
        if default_name_prefix is not None:
            args.name_prefix = str(default_name_prefix)
        else:
            prefix_key = (
                "baseline_name_prefix"
                if args.training_mode == "baseline_ce"
                else (
                    "batch_baseline_name_prefix"
                    if args.training_mode == "baseline_batch"
                    else "name_prefix"
                )
            )
            args.name_prefix = str(architecture_defaults[prefix_key])
    validate_arguments(parser, args)
    raw_argmax_only = (
        len(args.coefficient_values) == 1
        and np.isclose(float(args.coefficient_values[0]), 1.0)
    )
    baseline_score_scaling = (
        args.training_mode == "baseline_ce" and not raw_argmax_only
    )

    if args.worker_mode == "train":
        required = {
            "--worker-run-name": args.worker_run_name,
            "--worker-result-path": args.worker_result_path,
            "--worker-attempt-id": args.worker_attempt_id,
            "--worker-cache-path": args.worker_cache_path,
            "--worker-cache-metadata-path": args.worker_cache_metadata_path,
            "--training-key": args.training_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"Worker invocation is missing: {missing}")
        if args.worker_fold_id not in range(FOLD_COUNT):
            parser.error(f"Worker fold must be 0..{FOLD_COUNT - 1}.")
        run_training_worker(args)
        return

    try:
        gpus = core.parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    if len(gpus) > FOLD_COUNT:
        parser.error(f"At most {FOLD_COUNT} GPU workers are supported.")
    cuda_tokens = resolve_cuda_tokens(gpus)
    model_label = architecture_label(args.architecture)

    train_path = repo_root / "data" / "KDDTrain+.txt"
    stage1_path = repo_root / "src" / f"tune_{args.architecture}_focal_cv_4gpu.py"
    training_dependency_paths = [
        train_path,
        stage1_path,
        repo_root / "src" / "run_no_ctgan_model_ablation_4gpu.py",
        repo_root / "src" / "cnn_opt.py",
        repo_root / "src" / "cnn_gan_foc.py",
    ]
    if args.architecture in {"conv1d", "mlp", "transformer"}:
        training_dependency_paths.append(repo_root / "src" / "cnn_opt_1d_4gpu.py")
    required_paths = [script_path, *training_dependency_paths]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise SystemExit(f"Required files are missing: {missing_paths}")

    raw_train = core.load_collapsed_nsl_kdd(train_path, is_train=True)
    master_labels = raw_train["class"].to_numpy(dtype=np.int64)
    folds, fold_ids = stage1.make_fixed_folds(master_labels, args.fold_seed)
    library_versions = {
        "tensorflow": core.package_version("tensorflow"),
        "keras": core.package_version("keras"),
        "scikit_learn": core.package_version("scikit-learn"),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    training_settings = {
        "model": model_label,
        "architecture": args.architecture,
        "backbone": FIXED_BACKBONE,
        "input_representation": (
            "121 semantically ordered features, shape (121,)"
            if args.architecture == "mlp"
            else (
                "121 semantically ordered features, shape (11, 11, 1)"
                if args.architecture == "conv2d"
                else "121 semantically ordered features, shape (121, 1)"
            )
        ),
        "training_mode": args.training_mode,
        "loss": (
            "sparse_categorical_crossentropy"
            if args.training_mode != "focal_balanced"
            else "class_balanced_focal"
        ),
        "class_weighting": (
            "none"
            if args.training_mode != "focal_balanced"
            else "effective_number_from_outer_training_fold"
        ),
        "cb_beta": (
            float(args.cb_beta)
            if args.training_mode == "focal_balanced"
            else None
        ),
        "focal_gamma": (
            float(args.focal_gamma)
            if args.training_mode == "focal_balanced"
            else None
        ),
        "training_seeds": [int(seed) for seed in args.seeds],
        "fold_count": FOLD_COUNT,
        "fold_seed": int(args.fold_seed),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "batching": (
            "ordinary_shuffled"
            if args.training_mode not in BALANCED_BATCH_MODES
            else "minority_guaranteed_with_replacement"
        ),
        "minority_per_batch": (
            0
            if args.training_mode not in BALANCED_BATCH_MODES
            else int(args.minority_per_batch)
        ),
        "validation_used_during_training": False,
        "checkpointing": "none_fixed_epoch_budget",
        "decision_policy_during_training": "none",
        "ctgan": False,
        "deterministic_ops": bool(args.deterministic_ops),
        "library_versions": library_versions,
    }
    training_source_fingerprint = {
        "dependencies_and_data": core.fingerprint_files(training_dependency_paths),
        "worker_code": hashlib.sha256(
            (
                inspect.getsource(run_training_worker)
                + inspect.getsource(build_worker_command)
            ).encode("utf-8")
        ).hexdigest(),
    }
    training_identity = {
        "schema_version": SCHEMA_VERSION,
        "settings": training_settings,
        "folds": stage1.fold_protocol_rows(folds),
        "source_and_data_fingerprint": training_source_fingerprint,
    }
    training_key = stable_hash(training_identity, 12)
    coefficients = sorted(float(value) for value in args.coefficient_values)
    scoring_code_sha256 = hashlib.sha256(
        (
            inspect.getsource(assemble_oof_predictions)
            + inspect.getsource(score_pair_confusions)
            + inspect.getsource(metrics_from_confusions)
            + inspect.getsource(score_oof_probabilities)
        ).encode("utf-8")
    ).hexdigest()
    scoring_settings = {
        "training_mode": args.training_mode,
        "training_key": training_key,
        "scoring_code_sha256": scoring_code_sha256,
        "coefficient_values": coefficients,
        "pair_count": len(coefficients) ** 2,
        "macro_f1_retention": float(args.macro_f1_retention),
        "minority_precision_retention": float(args.minority_precision_retention),
        "retention_indicators": "diagnostic_only_mean_across_training_seeds",
        "pairs_excluded_by_retention": False,
        "ranking": (
            "not applicable; evaluate the single raw-argmax policy (1,1)"
            if raw_argmax_only
            else (
                "rank every pair without exclusion; maximize mean Rare Macro-F1; "
                "then mean Macro-F1; then prefer multiplicative closeness to (1,1); "
                "then lower Rare Macro-F1 sample SD"
            )
        ),
    }
    scoring_key = stable_hash(scoring_settings, 12)

    prefix = args.name_prefix.strip()
    results_dir = repo_root / "results"
    training_stem = f"{prefix}_training_{training_key}"
    scoring_stem = f"{prefix}_{training_key}_{scoring_key}"
    run_dir = results_dir / f"{training_stem}_runs"
    log_dir = results_dir / f"{training_stem}_logs"
    cache_dir = results_dir / f"{training_stem}_fold_cache"
    oof_dir = results_dir / f"{training_stem}_oof"
    protocol_path = results_dir / f"{scoring_stem}_protocol.json"
    folds_path = results_dir / f"{training_stem}_folds.csv"
    plan_path = results_dir / f"{training_stem}_plan.csv"

    cache_paths: Dict[int, tuple[Path, Path]] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        cache_paths[fold_id] = (
            cache_dir / f"fold_{fold_id + 1}.npz",
            cache_dir / f"fold_{fold_id + 1}_metadata.json",
        )
    plans: List[Dict[str, Any]] = []
    for seed in args.seeds:
        for fold in folds:
            fold_id = int(fold["fold_id"])
            cache_path, cache_metadata_path = cache_paths[fold_id]
            run_name = f"{training_stem}_s{int(seed)}_f{fold_id + 1}"
            plans.append(
                {
                    "training_key": training_key,
                    "architecture": args.architecture,
                    "training_mode": args.training_mode,
                    "seed": int(seed),
                    "fold_id": fold_id,
                    "fold_number": fold_id + 1,
                    "cb_beta": (
                        float(args.cb_beta)
                        if args.training_mode == "focal_balanced"
                        else None
                    ),
                    "focal_gamma": (
                        float(args.focal_gamma)
                        if args.training_mode == "focal_balanced"
                        else None
                    ),
                    "minority_per_batch": (
                        0
                        if args.training_mode not in BALANCED_BATCH_MODES
                        else int(args.minority_per_batch)
                    ),
                    "planned_gpu": gpus[len(plans) % len(gpus)],
                    "assigned_gpu": "",
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
    expected_fits = len(args.seeds) * FOLD_COUNT
    if len(plans) != expected_fits:
        raise RuntimeError("Training plan count is inconsistent.")

    if baseline_score_scaling:
        print(f"{model_label} cross-entropy baseline + score-scaling OOF search")
    elif args.training_mode == "baseline_ce":
        print(f"{model_label} pure cross-entropy baseline OOF evaluation")
    elif args.training_mode == "baseline_batch":
        print(f"{model_label} cross-entropy + minority-batch OOF evaluation")
    elif raw_argmax_only:
        print(f"{model_label} focal-loss + minority-batch OOF evaluation")
    else:
        print(f"{model_label} balanced-batch + score-scaling OOF search")
    print(f"Training key: {training_key}")
    print(f"Scoring key: {scoring_key}")
    print(f"GPU workers: {gpus}")
    print(f"CUDA allocation mapping: {cuda_tokens}")
    print(f"Seeds: {args.seeds}; folds: {FOLD_COUNT} (seed {args.fold_seed})")
    if args.training_mode == "baseline_ce":
        print(f"Baseline trainings: {len(plans)}")
        print("Loss: sparse categorical cross-entropy (no class weighting)")
        print("Batches: ordinary shuffled mini-batches")
        if baseline_score_scaling:
            print(f"Score-scaling pairs: {len(args.coefficient_values) ** 2}")
            print("Decision policy: validation-selected R2L/U2R score scaling")
            print("Focal loss: NO; guaranteed batches: NO; CTGAN: NO")
        else:
            print("Decision policy: raw multiclass argmax")
            print("Enhancements: no focal loss, guaranteed batches, CTGAN, or scaling")
    elif args.training_mode == "baseline_batch":
        print(f"Batch-only trainings: {len(plans)}")
        print("Loss: sparse categorical cross-entropy (no class weighting)")
        print(
            f"Guaranteed per batch: {args.minority_per_batch} R2L + "
            f"{args.minority_per_batch} U2R"
        )
        print("Decision policy: raw multiclass argmax")
        print("Focal loss: NO; score scaling: NO; CTGAN: NO")
    else:
        print(f"Frozen focal settings: beta={args.cb_beta}, gamma={args.focal_gamma}")
        print(f"Balanced-batch trainings: {len(plans)}")
        print(
            f"Guaranteed per batch: {args.minority_per_batch} R2L + "
            f"{args.minority_per_batch} U2R"
        )
        if raw_argmax_only:
            print("Decision policy: raw multiclass argmax")
            print("Score scaling: NO; CTGAN: NO")
        else:
            print(f"Coefficient values ({len(coefficients)}): {coefficients}")
            print(f"Offline coefficient pairs: {len(coefficients) ** 2}")
            print(
                "Diagnostic Macro-F1 retention marker: "
                f"{args.macro_f1_retention:.0%} (does not exclude pairs)"
            )
            print(
                "Diagnostic minority-precision retention marker: "
                f"{args.minority_precision_retention:.0%} (does not exclude pairs)"
            )
    print("KDDTest+ accessed: NO")

    if args.dry_run:
        for plan in plans:
            command = build_worker_command(
                script_path, plan, args, attempt_id="dry_run"
            )
            print(
                f"[planned GPU {plan['planned_gpu']} -> fold "
                f"{plan['fold_number']}] {shlex.join(command)}"
            )
        print("Dry run complete; no output files were written.")
        return

    stage1.validate_parent_dependencies(parser)
    for directory in (results_dir, run_dir, log_dir, cache_dir, oof_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cache_metadata: Dict[int, Dict[str, Any]] = {}
    print("Preparing four leakage-free fold caches...", flush=True)
    for fold in folds:
        fold_id = int(fold["fold_id"])
        cache_path, metadata_path = cache_paths[fold_id]
        cache_metadata[fold_id] = stage1.prepare_fold_cache(
            raw_train,
            master_labels,
            fold,
            cache_path,
            metadata_path,
            training_key,
        )
        print(
            f"  fold {fold_id + 1}: train={len(fold['train_indices'])}, "
            f"validation={len(fold['validation_indices'])}",
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
        "training_key": training_key,
        "scoring_key": scoring_key,
        "title": (
            f"{model_label} cross-entropy baseline and score-scaling OOF search"
            if baseline_score_scaling
            else (
                f"{model_label} pure cross-entropy baseline OOF evaluation"
                if args.training_mode == "baseline_ce"
                else (
                    f"{model_label} cross-entropy and minority-guaranteed-batch "
                    "raw-argmax OOF evaluation"
                    if args.training_mode == "baseline_batch"
                    else (
                        f"{model_label} focal-loss and minority-guaranteed-batch "
                        "raw-argmax OOF evaluation"
                        if raw_argmax_only
                        else (
                            f"{model_label} minority-guaranteed batching and "
                            "score-scaling OOF search"
                        )
                    )
                )
            )
        ),
        "training_settings": training_settings,
        "scoring_settings": scoring_settings,
        "source_and_data_fingerprint": training_identity["source_and_data_fingerprint"],
        "kddtrain_sha256": core.sha256_file(train_path),
        "kddtest_accessed": False,
        "synthetic_rows": 0,
        "folds": stage1.fold_protocol_rows(folds),
        "fold_assignment_path": str(folds_path),
        "fold_assignment_sha256": core.sha256_file(folds_path),
        "fold_cache_metadata": cache_metadata,
        "expected_trainings": len(plans),
        "expected_oof_probability_sets": len(args.seeds),
        "expected_per_seed_score_rows": len(args.seeds) * len(coefficients) ** 2,
        "expected_summary_rows": len(coefficients) ** 2,
        "runtime_gpu_workers": gpus,
        "runtime_cuda_mapping": cuda_tokens,
        "preprocessing_protocol": (
            "encoder and MinMax scaler fitted on each outer training partition only"
        ),
        "training_protocol": (
            "fixed epochs; ordinary shuffled mini-batches; held-out fold never "
            "passed to model.fit; no checkpoint or early stopping"
            if args.training_mode == "baseline_ce"
            else (
                "fixed epochs; minority-guaranteed batches; held-out fold never "
                "passed to model.fit; no checkpoint or early stopping"
            )
        ),
        "score_scaling_semantics": (
            "not used; raw multiclass argmax"
            if raw_argmax_only
            else (
                "divide R2L and U2R probability scores by their positive "
                "coefficients before multiclass argmax; below one promotes, "
                "above one suppresses"
            )
        ),
        "selection_protocol": (
            "none; report the single raw-argmax policy across seeds"
            if raw_argmax_only
            else (
                "evaluate each pair separately per seed; average metrics across "
                "seeds; exclude no pair; retain Macro-F1 and per-class "
                "minority-precision markers for diagnostics only; maximize mean "
                "Rare Macro-F1, then mean Macro-F1; use multiplicative distance "
                "to (1,1) and Rare Macro-F1 sample SD as later tie-breaks"
            )
        ),
        "final_test_policy": (
            "freeze the validation-selected score pair before later KDDTest+ evaluation"
            if baseline_score_scaling
            else (
                "use the unchanged architecture and raw argmax for later KDDTest+ evaluation"
                if args.training_mode == "baseline_ce"
                else (
                    "freeze minority-guaranteed batching with ordinary cross-entropy; "
                    "retain raw argmax before later KDDTest+ evaluation"
                    if args.training_mode == "baseline_batch"
                    else (
                        "freeze beta, gamma, and minority-guaranteed batching; retain "
                        "raw argmax before later KDDTest+ evaluation"
                        if raw_argmax_only
                        else (
                            "freeze beta, gamma, batching, and score pair before "
                            "KDDTest+ evaluation"
                        )
                    )
                )
            )
        ),
    }
    core.atomic_json(protocol_path, protocol)
    core.atomic_csv(plan_path, pd.DataFrame(plans))

    pending: List[Dict[str, Any]] = []
    statuses: Dict[str, str] = {}
    runtimes: Dict[str, float] = {}
    assigned_gpus: Dict[str, str] = {}
    for plan in plans:
        if not args.rerun and worker_result_is_complete(
            Path(plan["result_path"]), plan
        ):
            statuses[plan["run_name"]] = "skipped_complete"
            runtimes[plan["run_name"]] = 0.0
            assigned_gpus[plan["run_name"]] = str(
                core.read_json(Path(plan["result_path"])).get("assigned_gpu", "")
            )
        else:
            pending.append(plan)
    completed = len(plans) - len(pending)
    if completed:
        print(f"Resume: {completed}/{len(plans)} verified fits already complete.")

    task_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    for plan in pending:
        task_queue.put(plan)
    print_lock = threading.Lock()
    state_lock = threading.Lock()
    stop_event = threading.Event()
    failures: List[str] = []

    def execute_plan(
        gpu: str, cuda_token: str, plan: Dict[str, Any]
    ) -> tuple[str, float, str | None]:
        run_name = str(plan["run_name"])
        attempt_id = hashlib.sha256(
            f"{run_name}:{time.time_ns()}:{os.getpid()}".encode()
        ).hexdigest()[:16]
        command = build_worker_command(script_path, plan, args, attempt_id)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = cuda_token
        environment["EXPERIMENT_GPU_ID"] = gpu
        environment["PYTHONHASHSEED"] = str(plan["seed"])
        environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        log_path = Path(plan["log_path"])
        with print_lock:
            print(
                f"[GPU {gpu} (CUDA {cuda_token}) | fold "
                f"{plan['fold_number']}] START {run_name}",
                flush=True,
            )
        started = time.perf_counter()
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                completed_process = subprocess.run(
                    command,
                    cwd=repo_root,
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            runtime = time.perf_counter() - started
            if completed_process.returncode != 0:
                return (
                    "failed",
                    runtime,
                    f"{run_name}: exit={completed_process.returncode}, log={log_path}",
                )
            if not worker_result_is_complete(
                Path(plan["result_path"]), plan, expected_attempt_id=attempt_id
            ):
                return (
                    "failed",
                    runtime,
                    f"{run_name}: output validation failed, log={log_path}",
                )
            return "completed", runtime, None
        except Exception as error:
            return "failed", time.perf_counter() - started, f"{run_name}: {error!r}"

    def gpu_worker(gpu: str, cuda_token: str) -> None:
        nonlocal completed
        while not stop_event.is_set():
            try:
                plan = task_queue.get_nowait()
            except queue.Empty:
                return
            status, runtime, failure = execute_plan(gpu, cuda_token, plan)
            run_name = str(plan["run_name"])
            with state_lock:
                statuses[run_name] = status
                runtimes[run_name] = runtime
                assigned_gpus[run_name] = gpu
                if status == "failed":
                    failures.append(failure or f"{run_name}: unknown failure")
                    stop_event.set()
                else:
                    completed += 1
                current = completed
            with print_lock:
                print(
                    f"[GPU {gpu} (CUDA {cuda_token}) | fold "
                    f"{plan['fold_number']}] {status.upper()} {run_name} "
                    f"({runtime / 60.0:.1f} min)\n"
                    f"Training progress: {progress_bar(current, len(plans))}",
                    flush=True,
                )
            task_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(gpu_worker, gpu, cuda_tokens[gpu]) for gpu in gpus]
        for future in futures:
            future.result()

    updated_plans = [
        {
            **plan,
            "status": statuses.get(plan["run_name"], "not_started"),
            "assigned_gpu": assigned_gpus.get(plan["run_name"], ""),
            "runtime_seconds": runtimes.get(plan["run_name"]),
        }
        for plan in plans
    ]
    core.atomic_csv(plan_path, pd.DataFrame(updated_plans))
    if failures:
        raise SystemExit("Training failed:\n" + "\n".join(failures))
    incomplete = [
        plan["run_name"]
        for plan in plans
        if not worker_result_is_complete(Path(plan["result_path"]), plan)
    ]
    if incomplete:
        raise RuntimeError(f"Incomplete training artifacts remain: {incomplete}")

    fold_frame, raw_seed_frame, oof_paths = assemble_oof_predictions(
        plans,
        args.seeds,
        master_labels,
        fold_ids,
        oof_dir,
    )
    if raw_argmax_only:
        print("Training complete. Aggregating raw-argmax OOF metrics...")
    else:
        print("Training complete. Evaluating coefficient pairs without retraining...")
    per_seed_scores, ranking, best = score_oof_probabilities(
        oof_paths,
        args.seeds,
        coefficients,
        args.macro_f1_retention,
        args.minority_precision_retention,
        args.score_chunk_size,
    )
    pretty = formatted_summary(ranking, min(args.top_n, len(ranking)))

    fold_runs_path = results_dir / f"{training_stem}_fold_runs.csv"
    raw_seed_path = results_dir / f"{training_stem}_raw_seed_metrics.csv"
    per_seed_scores_path = results_dir / f"{scoring_stem}_per_seed_scores.csv"
    ranking_path = results_dir / f"{scoring_stem}_ranking.csv"
    formatted_path = results_dir / f"{scoring_stem}_top_formatted.csv"
    readable_path = results_dir / f"{scoring_stem}_summary.txt"
    best_path = results_dir / (
        f"{scoring_stem}_best_scaling.json"
        if baseline_score_scaling
        else (
            f"{scoring_stem}_baseline_summary.json"
            if args.training_mode == "baseline_ce"
            else (
                f"{scoring_stem}_batch_baseline_summary.json"
                if args.training_mode == "baseline_batch"
                else (
                    f"{scoring_stem}_focal_batch_summary.json"
                    if raw_argmax_only
                    else f"{scoring_stem}_best_scaling.json"
                )
            )
        )
    )
    comparison_path = results_dir / (
        f"{scoring_stem}_raw_argmax.csv"
        if raw_argmax_only
        else f"{scoring_stem}_raw_vs_selected.csv"
    )
    latest_path = results_dir / f"{prefix}_latest.json"
    core.atomic_csv(fold_runs_path, fold_frame)
    core.atomic_csv(raw_seed_path, raw_seed_frame)
    core.atomic_csv(per_seed_scores_path, per_seed_scores)
    core.atomic_csv(ranking_path, ranking)
    core.atomic_csv(formatted_path, pretty)

    raw_row = ranking[
        np.isclose(ranking["r2l_score_coefficient"], 1.0)
        & np.isclose(ranking["u2r_score_coefficient"], 1.0)
    ].iloc[0]
    selected_row = ranking.iloc[0]
    comparison_rows: List[Dict[str, Any]] = []
    policies = (
        [("raw_argmax", raw_row)]
        if raw_argmax_only
        else [("raw_argmax", raw_row), ("selected_scaling", selected_row)]
    )
    for policy, row in policies:
        comparison_rows.append(
            {
                "policy": policy,
                "r2l_score_coefficient": float(row["r2l_score_coefficient"]),
                "u2r_score_coefficient": float(row["u2r_score_coefficient"]),
                "eligible_mean": bool(row["eligible_mean"]),
                **{
                    f"{metric}_{suffix}": float(row[f"{metric}_{suffix}"])
                    for metric in SCORING_METRICS
                    for suffix in ("mean", "std")
                },
            }
        )
    core.atomic_csv(comparison_path, pd.DataFrame(comparison_rows))

    best.update(
        {
            "schema_version": SCHEMA_VERSION,
            "architecture": args.architecture,
            "training_mode": args.training_mode,
            "training_key": training_key,
            "scoring_key": scoring_key,
            "cb_beta": (
                float(args.cb_beta)
                if args.training_mode == "focal_balanced"
                else None
            ),
            "focal_gamma": (
                float(args.focal_gamma)
                if args.training_mode == "focal_balanced"
                else None
            ),
            "minority_per_batch": (
                0
                if args.training_mode not in BALANCED_BATCH_MODES
                else int(args.minority_per_batch)
            ),
            "training_seeds": [int(seed) for seed in args.seeds],
            "fold_count": FOLD_COUNT,
            "coefficient_values": coefficients,
            "pair_count": len(coefficients) ** 2,
            "selection_data": "KDDTrain+ pooled OOF predictions per seed only",
            "kddtest_accessed": False,
            "ranking_rule": scoring_settings["ranking"],
            "ranking_path": str(ranking_path),
            "per_seed_scores_path": str(per_seed_scores_path),
        }
    )
    core.atomic_json(best_path, best)
    if baseline_score_scaling:
        readable_text = (
            f"{model_label} cross-entropy baseline + score-scaling OOF search\n"
            f"Training key: {training_key}\n"
            f"Scoring key: {scoring_key}\n"
            f"Seeds: {args.seeds}; folds: {FOLD_COUNT}\n"
            "Loss: sparse categorical cross-entropy; no class weighting\n"
            "Batches: ordinary shuffled; focal loss: NO; "
            "minority-guaranteed batching: NO; CTGAN: NO\n"
            f"Coefficient values: {coefficients}\n"
            "Scaling is evaluated from saved OOF probabilities without retraining. "
            "Ranking: Rare Macro-F1, Macro-F1, distance to (1,1), "
            "Rare Macro-F1 stability\n"
            "KDDTest+ accessed: NO\n\n"
            + pretty.to_string(index=False)
            + "\n"
        )
    elif args.training_mode == "baseline_ce":
        readable_text = (
            f"{model_label} pure cross-entropy baseline OOF evaluation\n"
            f"Training key: {training_key}\n"
            f"Scoring key: {scoring_key}\n"
            f"Seeds: {args.seeds}; folds: {FOLD_COUNT}\n"
            "Loss: sparse categorical cross-entropy; no class weighting\n"
            "Batches: ordinary shuffled; decision: raw argmax\n"
            "Focal loss: NO; minority-guaranteed batching: NO; CTGAN: NO; "
            "score scaling: NO\n"
            "KDDTest+ accessed: NO\n\n"
            + pretty.to_string(index=False)
            + "\n"
        )
    elif args.training_mode == "baseline_batch":
        readable_text = (
            f"{model_label} cross-entropy + minority-guaranteed-batch OOF evaluation\n"
            f"Training key: {training_key}\n"
            f"Scoring key: {scoring_key}\n"
            f"Seeds: {args.seeds}; folds: {FOLD_COUNT}\n"
            "Loss: sparse categorical cross-entropy; no class weighting\n"
            f"Guaranteed per batch: {args.minority_per_batch} R2L + "
            f"{args.minority_per_batch} U2R\n"
            "Focal loss: NO; decision: raw argmax; CTGAN: NO; score scaling: NO\n"
            "KDDTest+ accessed: NO\n\n"
            + pretty.to_string(index=False)
            + "\n"
        )
    elif raw_argmax_only:
        readable_text = (
            f"{model_label} focal-loss + minority-guaranteed-batch OOF evaluation\n"
            f"Training key: {training_key}\n"
            f"Scoring key: {scoring_key}\n"
            f"Focal settings: beta={args.cb_beta}, gamma={args.focal_gamma}\n"
            f"Seeds: {args.seeds}; folds: {FOLD_COUNT}\n"
            f"Guaranteed per batch: {args.minority_per_batch} R2L + "
            f"{args.minority_per_batch} U2R\n"
            "Decision: raw argmax; CTGAN: NO; score scaling: NO\n"
            "KDDTest+ accessed: NO\n\n"
            + pretty.to_string(index=False)
            + "\n"
        )
    else:
        readable_text = (
            f"{model_label} balanced-batch score-scaling search\n"
            f"Training key: {training_key}\n"
            f"Scoring key: {scoring_key}\n"
            f"Focal settings: beta={args.cb_beta}, gamma={args.focal_gamma}\n"
            f"Seeds: {args.seeds}; folds: {FOLD_COUNT}\n"
            f"Coefficient values: {coefficients}\n"
            f"Diagnostic Macro-F1 marker: {args.macro_f1_retention:.0%} of raw mean\n"
            "Diagnostic minority-precision marker: "
            f"{args.minority_precision_retention:.0%} of each raw mean\n"
            "No pair excluded by these markers. Ranking: Rare Macro-F1, "
            "Macro-F1, distance to (1,1), Rare Macro-F1 stability\n"
            "KDDTest+ accessed: NO\n\n"
            + pretty.to_string(index=False)
            + "\n"
        )
    readable_path.write_text(readable_text, encoding="utf-8")
    latest = {
        "architecture": args.architecture,
        "training_key": training_key,
        "scoring_key": scoring_key,
        "protocol": str(protocol_path),
        "plan": str(plan_path),
        "fold_runs": str(fold_runs_path),
        "raw_seed_metrics": str(raw_seed_path),
        "oof_directory": str(oof_dir),
        "per_seed_scores": str(per_seed_scores_path),
        "ranking": str(ranking_path),
        "formatted_top": str(formatted_path),
        "readable_summary": str(readable_path),
        "raw_argmax_or_comparison": str(comparison_path),
        "baseline_summary_or_best_scaling": str(best_path),
    }
    core.atomic_json(latest_path, latest)

    if baseline_score_scaling:
        print(f"\n=== Ranked {model_label} baseline score-scaling pairs ===")
        print(pretty.to_string(index=False))
        print(
            "\nSelected coefficients: "
            f"R2L={best['r2l_score_coefficient']}, "
            f"U2R={best['u2r_score_coefficient']}"
        )
        print(f"Raw versus selected: {comparison_path}")
        print(f"Full numeric ranking: {ranking_path}")
        print(f"Best scaling: {best_path}")
    elif args.training_mode == "baseline_ce":
        print(f"\n=== {model_label} pure baseline OOF results ===")
        print(pretty.to_string(index=False))
        print(f"Per-seed raw metrics: {raw_seed_path}")
        print(f"Numeric mean/std summary: {ranking_path}")
        print(f"Readable summary: {readable_path}")
    elif args.training_mode == "baseline_batch":
        print(f"\n=== {model_label} batch-only OOF results ===")
        print(pretty.to_string(index=False))
        print(f"Per-seed raw metrics: {raw_seed_path}")
        print(f"Numeric mean/std summary: {ranking_path}")
        print(f"Readable summary: {readable_path}")
    elif raw_argmax_only:
        print(f"\n=== {model_label} focal + minority-batch OOF results ===")
        print(pretty.to_string(index=False))
        print(f"Per-seed raw metrics: {raw_seed_path}")
        print(f"Numeric mean/std summary: {ranking_path}")
        print(f"Readable summary: {readable_path}")
    else:
        print(f"\n=== Ranked {model_label} score-scaling pairs ===")
        print(pretty.to_string(index=False))
        print(
            "\nSelected coefficients: "
            f"R2L={best['r2l_score_coefficient']}, "
            f"U2R={best['u2r_score_coefficient']}"
        )
        print(f"Raw versus selected: {comparison_path}")
        print(f"Full numeric ranking: {ranking_path}")
        print(f"Best scaling: {best_path}")
    print(f"Latest-results pointer: {latest_path}")


if __name__ == "__main__":
    main()
