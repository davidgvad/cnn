"""Final KDDTest+ comparison of pure and fully equipped neural models.

All model and imbalance-control settings in this file are frozen constants from
the earlier KDDTrain+ cross-validation experiments.  For every architecture and
seed, the script trains once on all of KDDTrain+ and evaluates once on untouched
KDDTest+.  It does not tune, checkpoint, or select anything using KDDTest+.

Supported variants
------------------
baseline
    Cross-entropy, ordinary shuffled mini-batches, raw multiclass argmax.
focal_only
    Class-balanced focal loss, ordinary shuffled mini-batches, raw argmax.
batch_only
    Cross-entropy, minority-guaranteed mini-batches, raw argmax.
scaling_only
    Cross-entropy, ordinary shuffled mini-batches, frozen score scaling.
full
    Class-balanced focal loss, one guaranteed R2L and U2R example per batch,
    and the architecture-specific frozen class-score coefficients.

The controller runs independent fits concurrently, one process per GPU.  Child
processes see exactly one CUDA device and therefore hold one complete model.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
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


SCHEMA_VERSION = 1
DEFAULT_ARCHITECTURES = ["conv2d", "conv1d", "transformer", "mlp"]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_VARIANTS = ["baseline", "full"]
SUPPORTED_VARIANTS = [
    "baseline",
    "focal_only",
    "batch_only",
    "scaling_only",
    "full",
]
FOCAL_VARIANTS = {"focal_only", "full"}
MINORITY_BATCH_VARIANTS = {"batch_only", "full"}
SCORE_SCALING_VARIANTS = {"scaling_only", "full"}
VARIANT_LABELS = {
    "baseline": "Baseline",
    "focal_only": "Baseline + focal loss",
    "batch_only": "Baseline + minority batching",
    "scaling_only": "Baseline + frozen scaling",
    "full": "Focal + batching + scaling",
}
METRICS = list(core.METRICS)

# These values were selected on KDDTrain+ validation predictions and are now
# frozen. Score scaling divides a class probability by its coefficient.
FROZEN_CONFIG: Dict[str, Dict[str, Any]] = {
    "conv2d": {
        "label": "Conv2D",
        "beta": 0.99,
        "focal_gamma": 0.50,
        "r2l_score_coefficient": 1.00,
        "u2r_score_coefficient": 4.00,
        "backbone": {
            "groups": 1,
            "base_filters": 64,
            "dense_units": 256,
            "dropout1": 0.25,
            "dropout2": 0.30,
            "batch_norm": True,
            "residual": True,
            "expected_parameters": 109_381,
        },
    },
    "conv1d": {
        "label": "Conv1D",
        "beta": 0.99,
        "focal_gamma": 0.25,
        "r2l_score_coefficient": 1.00,
        "u2r_score_coefficient": 7.00,
        "backbone": {
            "groups": 1,
            "base_filters": 64,
            "dense_units": 48,
            "dropout1": 0.25,
            "dropout2": 0.30,
            "batch_norm": True,
            "residual": True,
            "expected_parameters": 109_797,
        },
    },
    "transformer": {
        "label": "Transformer",
        "beta": 0.99,
        "focal_gamma": 0.75,
        "r2l_score_coefficient": 1.00,
        "u2r_score_coefficient": 10.00,
        "backbone": {
            "d_model": 64,
            "num_heads": 4,
            "blocks": 2,
            "ff_dim": 128,
            "dense_units": 512,
            "dropout": 0.10,
            "head_dropout": 0.30,
            "expected_parameters": 110_661,
        },
    },
    "mlp": {
        "label": "MLP",
        "beta": 0.99,
        "focal_gamma": 0.25,
        "r2l_score_coefficient": 0.40,
        "u2r_score_coefficient": 1.90,
        "backbone": {
            "dense_units": 256,
            "dropout1": 0.25,
            "dropout2": 0.30,
            "batch_norm": True,
            "expected_parameters": 99_845,
        },
    },
}


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def progress_bar(completed: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[------------------------------]   0.00% (0/0)"
    completed = max(0, min(int(completed), int(total)))
    filled = int(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {100.0 * completed / total:6.2f}% ({completed}/{total})"


def resolve_cuda_tokens(gpus: Sequence[str]) -> Dict[str, str]:
    """Resolve logical GPU indexes inside a scheduler-provided allocation."""
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
        return {
            gpu: tokens[index]
            for gpu, index in zip(gpus, logical, strict=True)
        }
    return {gpu: gpu for gpu in gpus}


def reshape_features(X: np.ndarray, architecture: str) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if architecture == "conv2d":
        return X.reshape(-1, 11, 11, 1)
    if architecture in {"conv1d", "transformer"}:
        return X.reshape(-1, 121, 1)
    if architecture == "mlp":
        return X.reshape(-1, 121)
    raise ValueError(f"Unsupported architecture: {architecture}")


def build_model(architecture: str, loss: Any) -> Any:
    from cnn_opt import build_opt_cnn  # type: ignore
    from cnn_opt_1d_4gpu import (  # type: ignore
        build_opt_cnn_1d,
        build_opt_mlp,
        build_vanilla_transformer,
    )

    backbone = FROZEN_CONFIG[architecture]["backbone"]
    if architecture == "conv2d":
        return build_opt_cnn(
            loss=loss,
            groups=backbone["groups"],
            base_filters=backbone["base_filters"],
            dense_units=backbone["dense_units"],
            dropout1=backbone["dropout1"],
            dropout2=backbone["dropout2"],
            use_batch_norm=backbone["batch_norm"],
            use_residual=backbone["residual"],
        )
    if architecture == "conv1d":
        return build_opt_cnn_1d(
            loss=loss,
            groups=backbone["groups"],
            base_filters=backbone["base_filters"],
            dense_units=backbone["dense_units"],
            dropout1=backbone["dropout1"],
            dropout2=backbone["dropout2"],
            use_batch_norm=backbone["batch_norm"],
            use_residual=backbone["residual"],
        )
    if architecture == "transformer":
        return build_vanilla_transformer(
            loss=loss,
            d_model=backbone["d_model"],
            num_heads=backbone["num_heads"],
            num_blocks=backbone["blocks"],
            ff_dim=backbone["ff_dim"],
            dense_units=backbone["dense_units"],
            transformer_dropout=backbone["dropout"],
            head_dropout=backbone["head_dropout"],
        )
    if architecture == "mlp":
        return build_opt_mlp(
            loss=loss,
            dense_units=backbone["dense_units"],
            dropout1=backbone["dropout1"],
            dropout2=backbone["dropout2"],
            use_batch_norm=backbone["batch_norm"],
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def prepare_train_test_cache(
    repo_root: Path,
    cache_path: Path,
    metadata_path: Path,
    experiment_key: str,
) -> Dict[str, Any]:
    train_path = repo_root / "data" / "KDDTrain+.txt"
    test_path = repo_root / "data" / "KDDTest+.txt"
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Expected data/KDDTrain+.txt and data/KDDTest+.txt."
        )

    expected_sources = {
        "train_sha256": core.sha256_file(train_path),
        "test_sha256": core.sha256_file(test_path),
    }
    if cache_path.is_file() and metadata_path.is_file():
        metadata = core.read_json(metadata_path)
        if (
            metadata.get("schema_version") == SCHEMA_VERSION
            and metadata.get("experiment_key") == experiment_key
            and metadata.get("cache_sha256") == core.sha256_file(cache_path)
            and all(metadata.get(key) == value for key, value in expected_sources.items())
        ):
            return metadata

    print("Preparing full KDDTrain+ / untouched KDDTest+ feature cache...", flush=True)
    raw_train = core.load_collapsed_nsl_kdd(train_path, is_train=True)
    raw_test = core.load_collapsed_nsl_kdd(test_path, is_train=False)
    preprocessor = core.fit_fold_preprocessor(raw_train)
    X_train, y_train = core.transform_with_fold_preprocessor(raw_train, preprocessor)
    X_test, y_test = core.transform_with_fold_preprocessor(raw_test, preprocessor)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise RuntimeError("Preprocessed features contain a non-finite value.")

    core.atomic_npz(
        cache_path,
        X_train=np.asarray(X_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int64),
        X_test=np.asarray(X_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        **expected_sources,
        "cache_sha256": core.sha256_file(cache_path),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "feature_count": int(X_train.shape[1]),
        "train_class_counts": np.bincount(y_train, minlength=5).astype(int).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=5).astype(int).tolist(),
        "feature_order": list(preprocessor["ordered_features"]),
        "feature_order_sha256": hashlib.sha256(
            "\n".join(preprocessor["ordered_features"]).encode("utf-8")
        ).hexdigest(),
        "preprocessor_fit_partition": "all KDDTrain+ only",
        "test_fit_involvement": "none; transform only",
    }
    core.atomic_json(metadata_path, metadata)
    return metadata


def run_worker(args: argparse.Namespace) -> None:
    # TensorFlow and model modules are deliberately imported only after the
    # controller has restricted this process to one CUDA device.
    import tensorflow as tf

    from cnn_gan_foc import ClassBalancedFocalLoss  # type: ignore
    from cnn_opt import BalancedBatchSequence  # type: ignore

    visible_gpus = tf.config.list_physical_devices("GPU")
    if not args.allow_cpu and len(visible_gpus) != 1:
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
                "Deterministic TensorFlow operations were requested but failed."
            ) from error

    cache_path = Path(args.worker_cache_path)
    metadata_path = Path(args.worker_cache_metadata_path)
    metadata = core.read_json(metadata_path)
    if metadata.get("experiment_key") != args.experiment_key:
        raise RuntimeError("Feature cache belongs to a different experiment.")
    if metadata.get("cache_sha256") != core.sha256_file(cache_path):
        raise RuntimeError("Feature cache hash does not match its metadata.")
    with np.load(cache_path, allow_pickle=False) as artifact:
        X_train_flat = np.asarray(artifact["X_train"], dtype=np.float32)
        y_train = np.asarray(artifact["y_train"], dtype=np.int64)
        X_test_flat = np.asarray(artifact["X_test"], dtype=np.float32)
        y_test = np.asarray(artifact["y_test"], dtype=np.int64)

    architecture = str(args.worker_architecture)
    variant = str(args.worker_variant)
    seed = int(args.worker_seed)
    frozen = FROZEN_CONFIG[architecture]
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)

    if variant in FOCAL_VARIANTS:
        alpha, alpha_counts = core.effective_number_alpha(
            y_train,
            beta=float(frozen["beta"]),
            num_classes=5,
        )
        loss = ClassBalancedFocalLoss(
            alpha=alpha,
            gamma=float(frozen["focal_gamma"]),
        )
    elif variant in SUPPORTED_VARIANTS:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        alpha = None
        alpha_counts = np.bincount(y_train, minlength=5).astype(np.int64)
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    model = build_model(architecture, loss)
    model_parameters = int(model.count_params())
    expected_parameters = int(frozen["backbone"]["expected_parameters"])
    if model_parameters != expected_parameters:
        raise RuntimeError(
            f"{architecture} parameter count changed: expected "
            f"{expected_parameters}, got {model_parameters}."
        )

    X_train = reshape_features(X_train_flat, architecture)
    X_test = reshape_features(X_test_flat, architecture)
    started = time.perf_counter()
    if variant in MINORITY_BATCH_VARIANTS:
        batches = BalancedBatchSequence(
            X_train,
            y_train,
            batch_size=int(args.batch_size),
            minority_per_batch=int(args.minority_per_batch),
            seed=seed,
        )
        history = model.fit(
            batches,
            epochs=int(args.epochs),
            verbose=int(args.fit_verbose),
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            shuffle=True,
            verbose=int(args.fit_verbose),
        )
    probabilities = np.asarray(
        model.predict(X_test, batch_size=int(args.batch_size), verbose=0),
        dtype=np.float32,
    )
    training_seconds = time.perf_counter() - started
    if probabilities.shape != (len(y_test), 5):
        raise RuntimeError(f"Unexpected KDDTest+ probability shape: {probabilities.shape}.")
    if not np.isfinite(probabilities).all():
        raise RuntimeError("KDDTest+ probabilities contain a non-finite value.")

    raw_predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    if variant in SCORE_SCALING_VARIANTS:
        predictions = core.apply_class_score_scaling(
            probabilities,
            {
                2: float(frozen["r2l_score_coefficient"]),
                3: float(frozen["u2r_score_coefficient"]),
            },
        )
    else:
        predictions = raw_predictions
    metrics = core.calculate_metrics(y_test, predictions)
    raw_metrics = core.calculate_metrics(y_test, raw_predictions)

    result_path = Path(args.worker_result_path)
    prediction_path = Path(args.worker_prediction_path)
    core.atomic_npz(
        prediction_path,
        labels=y_test,
        probabilities=probabilities,
        raw_predictions=raw_predictions,
        final_predictions=predictions,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": args.experiment_key,
        "run_name": args.worker_run_name,
        "architecture": architecture,
        "architecture_label": frozen["label"],
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "seed": seed,
        "model_parameters": model_parameters,
        "backbone": frozen["backbone"],
        "loss": (
            "class_balanced_focal"
            if variant in FOCAL_VARIANTS
            else "sparse_categorical_crossentropy"
        ),
        "cb_beta": float(frozen["beta"]) if variant in FOCAL_VARIANTS else None,
        "focal_gamma": (
            float(frozen["focal_gamma"]) if variant in FOCAL_VARIANTS else None
        ),
        "alpha": np.asarray(alpha, dtype=float).tolist() if alpha is not None else None,
        "alpha_counts": np.asarray(alpha_counts, dtype=int).tolist(),
        "batching": (
            "minority_guaranteed_with_replacement"
            if variant in MINORITY_BATCH_VARIANTS
            else "ordinary_shuffled"
        ),
        "minority_per_batch_per_class": (
            int(args.minority_per_batch)
            if variant in MINORITY_BATCH_VARIANTS
            else 0
        ),
        "score_scaling_used": variant in SCORE_SCALING_VARIANTS,
        "score_scaling_operation": (
            "class_score_divided_by_coefficient"
            if variant in SCORE_SCALING_VARIANTS
            else None
        ),
        "r2l_score_coefficient": (
            float(frozen["r2l_score_coefficient"])
            if variant in SCORE_SCALING_VARIANTS
            else 1.0
        ),
        "u2r_score_coefficient": (
            float(frozen["u2r_score_coefficient"])
            if variant in SCORE_SCALING_VARIANTS
            else 1.0
        ),
        "decision_policy": (
            "frozen_score_scaling_then_argmax"
            if variant in SCORE_SCALING_VARIANTS
            else "raw_argmax"
        ),
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "batch_size": int(args.batch_size),
        "optimizer": "adam_keras_defaults",
        "checkpointing": "none_fixed_epoch_budget",
        "validation_used_during_training": False,
        "train_partition": "all KDDTrain+",
        "evaluation_partition": "untouched KDDTest+",
        "preprocessor_fit_partition": "all KDDTrain+ only",
        "kddtest_used_for_selection": False,
        "ctgan_used": False,
        "synthetic_rows": 0,
        "metrics": metrics,
        "raw_argmax_metrics": raw_metrics,
        "prediction_path": str(prediction_path),
        "prediction_sha256": core.sha256_file(prediction_path),
        "feature_cache_sha256": metadata["cache_sha256"],
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
        f"macro-F1={metrics['macro_f1']:.6f}, "
        f"rare-F1={metrics['rare_f1']:.6f}",
        flush=True,
    )


def result_is_complete(result_path: Path, plan: Dict[str, Any]) -> bool:
    if not result_path.is_file():
        return False
    try:
        result = core.read_json(result_path)
        prediction_path = Path(str(result["prediction_path"]))
        expected = {
            "schema_version": SCHEMA_VERSION,
            "experiment_key": plan["experiment_key"],
            "run_name": plan["run_name"],
            "architecture": plan["architecture"],
            "variant": plan["variant"],
            "seed": int(plan["seed"]),
            "epochs_requested": int(plan["epochs"]),
            "batch_size": int(plan["batch_size"]),
            "evaluation_partition": "untouched KDDTest+",
            "kddtest_used_for_selection": False,
        }
        return (
            all(result.get(key) == value for key, value in expected.items())
            and set(result.get("metrics", {})) == set(METRICS)
            and all(np.isfinite(float(result["metrics"][metric])) for metric in METRICS)
            and prediction_path.is_file()
            and result.get("prediction_sha256") == core.sha256_file(prediction_path)
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def build_worker_command(
    script_path: Path,
    plan: Dict[str, Any],
    args: argparse.Namespace,
) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--worker",
        "--experiment-key",
        str(plan["experiment_key"]),
        "--worker-architecture",
        str(plan["architecture"]),
        "--worker-variant",
        str(plan["variant"]),
        "--worker-seed",
        str(plan["seed"]),
        "--worker-run-name",
        str(plan["run_name"]),
        "--worker-cache-path",
        str(plan["cache_path"]),
        "--worker-cache-metadata-path",
        str(plan["cache_metadata_path"]),
        "--worker-result-path",
        str(plan["result_path"]),
        "--worker-prediction-path",
        str(plan["prediction_path"]),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--minority-per-batch",
        str(args.minority_per_batch),
        "--fit-verbose",
        str(args.fit_verbose),
    ]
    if args.allow_cpu:
        command.append("--allow-cpu")
    if args.deterministic_ops:
        command.append("--deterministic-ops")
    return command


def aggregate_results(
    plans: Sequence[Dict[str, Any]],
    results_dir: Path,
    experiment_key: str,
    output_stem: str,
) -> Dict[str, str]:
    rows: List[Dict[str, Any]] = []
    for plan in plans:
        result = core.read_json(Path(plan["result_path"]))
        row: Dict[str, Any] = {
            "architecture": result["architecture"],
            "model": result["architecture_label"],
            "variant": result["variant"],
            "configuration": result["variant_label"],
            "seed": int(result["seed"]),
            "model_parameters": int(result["model_parameters"]),
            "cb_beta": result["cb_beta"],
            "focal_gamma": result["focal_gamma"],
            "minority_per_batch": int(result["minority_per_batch_per_class"]),
            "r2l_score_coefficient": float(result["r2l_score_coefficient"]),
            "u2r_score_coefficient": float(result["u2r_score_coefficient"]),
            "training_seconds": float(result["training_seconds"]),
        }
        row.update({metric: float(result["metrics"][metric]) for metric in METRICS})
        rows.append(row)
    run_frame = pd.DataFrame(rows).sort_values(
        ["architecture", "variant", "seed"]
    ).reset_index(drop=True)

    summary_rows: List[Dict[str, Any]] = []
    for (architecture, variant), group in run_frame.groupby(
        ["architecture", "variant"], sort=False
    ):
        first = group.iloc[0]
        summary: Dict[str, Any] = {
            "architecture": architecture,
            "model": first["model"],
            "variant": variant,
            "configuration": first["configuration"],
            "runs": int(len(group)),
            "seeds": ",".join(str(int(value)) for value in sorted(group["seed"])),
            "model_parameters": int(first["model_parameters"]),
            "cb_beta": first["cb_beta"],
            "focal_gamma": first["focal_gamma"],
            "minority_per_batch": int(first["minority_per_batch"]),
            "r2l_score_coefficient": float(first["r2l_score_coefficient"]),
            "u2r_score_coefficient": float(first["u2r_score_coefficient"]),
        }
        for metric in METRICS:
            summary[f"{metric}_mean"] = float(group[metric].mean())
            summary[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(summary)
    summary_frame = pd.DataFrame(summary_rows)
    architecture_order = {name: index for index, name in enumerate(DEFAULT_ARCHITECTURES)}
    variant_order = {name: index for index, name in enumerate(SUPPORTED_VARIANTS)}
    summary_frame["_architecture_order"] = summary_frame["architecture"].map(architecture_order)
    summary_frame["_variant_order"] = summary_frame["variant"].map(variant_order)
    summary_frame = summary_frame.sort_values(
        ["_architecture_order", "_variant_order"]
    ).drop(columns=["_architecture_order", "_variant_order"]).reset_index(drop=True)

    formatted = summary_frame[
        ["model", "configuration", "runs", "seeds", "model_parameters"]
    ].copy()
    for metric in METRICS:
        formatted[metric] = summary_frame.apply(
            lambda row, name=metric: (
                f"{100.0 * row[f'{name}_mean']:.2f}% +/- "
                f"{100.0 * row[f'{name}_std']:.2f}%"
            ),
            axis=1,
        )

    delta_rows: List[Dict[str, Any]] = []
    for architecture, group in run_frame.groupby("architecture"):
        baseline = group[group["variant"] == "baseline"].set_index("seed")
        full = group[group["variant"] == "full"].set_index("seed")
        common_seeds = sorted(set(baseline.index) & set(full.index))
        for seed in common_seeds:
            row = {
                "architecture": architecture,
                "model": str(full.loc[seed, "model"]),
                "seed": int(seed),
            }
            for metric in METRICS:
                row[f"{metric}_delta_full_minus_baseline"] = float(
                    full.loc[seed, metric] - baseline.loc[seed, metric]
                )
            delta_rows.append(row)
    delta_columns = [
        "architecture",
        "model",
        "seed",
        *[f"{metric}_delta_full_minus_baseline" for metric in METRICS],
    ]
    delta_frame = pd.DataFrame(delta_rows, columns=delta_columns)
    delta_summary_rows: List[Dict[str, Any]] = []
    for architecture, group in delta_frame.groupby("architecture", sort=False):
        row = {
            "architecture": architecture,
            "model": str(group.iloc[0]["model"]),
            "paired_seeds": int(len(group)),
        }
        for metric in METRICS:
            column = f"{metric}_delta_full_minus_baseline"
            row[f"{metric}_delta_mean"] = float(group[column].mean())
            row[f"{metric}_delta_std"] = float(group[column].std(ddof=1))
        delta_summary_rows.append(row)
    delta_summary_columns = [
        "architecture",
        "model",
        "paired_seeds",
        *[
            name
            for metric in METRICS
            for name in (f"{metric}_delta_mean", f"{metric}_delta_std")
        ],
    ]
    delta_summary_frame = pd.DataFrame(
        delta_summary_rows,
        columns=delta_summary_columns,
    )

    prefix = results_dir / f"{output_stem}_{experiment_key}"
    paths = {
        "all_runs": str(prefix.with_name(f"{prefix.name}_all_runs.csv")),
        "summary": str(prefix.with_name(f"{prefix.name}_summary.csv")),
        "formatted_summary": str(prefix.with_name(f"{prefix.name}_summary_formatted.csv")),
        "paired_deltas": str(prefix.with_name(f"{prefix.name}_paired_deltas.csv")),
        "paired_delta_summary": str(prefix.with_name(f"{prefix.name}_paired_delta_summary.csv")),
    }
    core.atomic_csv(Path(paths["all_runs"]), run_frame)
    core.atomic_csv(Path(paths["summary"]), summary_frame)
    core.atomic_csv(Path(paths["formatted_summary"]), formatted)
    core.atomic_csv(Path(paths["paired_deltas"]), delta_frame)
    core.atomic_csv(Path(paths["paired_delta_summary"]), delta_summary_frame)

    display_metrics = [
        "accuracy",
        "mcc",
        "macro_f1",
        "macro_recall",
        "rare_f1",
        "r2l_precision",
        "r2l_recall",
        "u2r_precision",
        "u2r_recall",
    ]
    print("\n=== Final KDDTest+ neural-configuration results ===")
    print(
        formatted[["model", "configuration", "runs", *display_metrics]].to_string(
            index=False
        )
    )
    return paths


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architectures", nargs="+", choices=DEFAULT_ARCHITECTURES, default=DEFAULT_ARCHITECTURES)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=SUPPORTED_VARIANTS,
        default=DEFAULT_VARIANTS,
        help="Final fixed configurations to train and evaluate.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument("--fit-verbose", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--deterministic-ops", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Internal worker arguments. Users run the controller, not these directly.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-architecture", choices=DEFAULT_ARCHITECTURES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-variant", choices=SUPPORTED_VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-name", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cache-path", help=argparse.SUPPRESS)
    parser.add_argument("--worker-cache-metadata-path", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", help=argparse.SUPPRESS)
    parser.add_argument("--worker-prediction-path", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.minority_per_batch <= 0:
        parser.error("--minority-per-batch must be positive.")
    if not args.seeds:
        parser.error("At least one seed is required.")
    if not args.gpus and not args.allow_cpu:
        parser.error("At least one GPU is required unless --allow-cpu is used.")
    return args


def main() -> None:
    args = parse_arguments()
    if args.worker:
        required_worker_values = [
            args.experiment_key,
            args.worker_architecture,
            args.worker_variant,
            args.worker_seed,
            args.worker_run_name,
            args.worker_cache_path,
            args.worker_cache_metadata_path,
            args.worker_result_path,
            args.worker_prediction_path,
        ]
        if any(value is None or value == "" for value in required_worker_values):
            raise ValueError("A required internal worker argument is missing.")
        run_worker(args)
        return

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (args.results_dir or (repo_root / "results")).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    architectures = list(dict.fromkeys(args.architectures))
    variants = list(dict.fromkeys(args.variants))
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    gpus = list(dict.fromkeys(str(gpu) for gpu in args.gpus))
    if args.allow_cpu and not gpus:
        gpus = [""]

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "final frozen neural-configuration evaluation on KDDTest+",
        "architectures": architectures,
        "variants": variants,
        "seeds": seeds,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "minority_per_batch_per_class": int(args.minority_per_batch),
        "frozen_config": {name: FROZEN_CONFIG[name] for name in architectures},
        "training_partition": "all KDDTrain+",
        "evaluation_partition": "untouched KDDTest+",
        "preprocessing_fit": "all KDDTrain+ only",
        "validation_during_final_training": False,
        "checkpointing": "none_fixed_epoch_budget",
        "selection_on_kddtest": False,
        "ctgan_used": False,
    }
    experiment_key = stable_hash(protocol)
    protocol["experiment_key"] = experiment_key
    if variants == DEFAULT_VARIANTS:
        output_stem = "final_baseline_vs_full_kddtest"
    elif variants == ["focal_only", "batch_only", "scaling_only"]:
        output_stem = "final_single_enhancement_kddtest"
    else:
        output_stem = "final_neural_variants_kddtest"
    prefix = f"{output_stem}_{experiment_key}"
    protocol_path = results_dir / f"{prefix}_protocol.json"
    plan_path = results_dir / f"{prefix}_plan.csv"
    cache_path = results_dir / f"{prefix}_feature_cache.npz"
    cache_metadata_path = results_dir / f"{prefix}_feature_cache.json"
    run_dir = results_dir / f"{prefix}_runs"
    log_dir = results_dir / f"{prefix}_logs"
    prediction_dir = results_dir / f"{prefix}_predictions"

    plans: List[Dict[str, Any]] = []
    for seed in seeds:
        for architecture in architectures:
            for variant in variants:
                run_name = f"{prefix}_{architecture}_{variant}_s{seed}"
                plans.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "experiment_key": experiment_key,
                        "run_name": run_name,
                        "architecture": architecture,
                        "variant": variant,
                        "seed": int(seed),
                        "epochs": int(args.epochs),
                        "batch_size": int(args.batch_size),
                        "cache_path": str(cache_path),
                        "cache_metadata_path": str(cache_metadata_path),
                        "result_path": str(run_dir / f"{run_name}.json"),
                        "prediction_path": str(prediction_dir / f"{run_name}.npz"),
                        "log_path": str(log_dir / f"{run_name}.log"),
                    }
                )

    print("Final frozen neural-configuration evaluation on KDDTest+", flush=True)
    print(f"Experiment key: {experiment_key}", flush=True)
    print(f"Architectures: {architectures}", flush=True)
    print(f"Configurations: {[VARIANT_LABELS[name] for name in variants]}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(
        f"Fits: {len(plans)} ({len(architectures)} architectures x "
        f"{len(variants)} configurations x {len(seeds)} seeds)",
        flush=True,
    )
    print(f"GPU workers: {gpus}", flush=True)
    print("Train: all KDDTrain+; final evaluation: untouched KDDTest+", flush=True)
    for variant in variants:
        focal = "focal loss" if variant in FOCAL_VARIANTS else "cross-entropy"
        batching = (
            "minority batches"
            if variant in MINORITY_BATCH_VARIANTS
            else "ordinary batches"
        )
        decision = (
            "frozen score scaling"
            if variant in SCORE_SCALING_VARIANTS
            else "raw argmax"
        )
        print(
            f"{VARIANT_LABELS[variant]}: {focal} + {batching} + {decision}",
            flush=True,
        )
    print("KDDTest+ is not used for tuning or model selection.", flush=True)
    if args.dry_run:
        print("\nDry run; planned fits:")
        print(pd.DataFrame(plans)[["architecture", "variant", "seed", "run_name"]].to_string(index=False))
        return

    for directory in (run_dir, log_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)
    core.atomic_json(protocol_path, protocol)
    prepare_train_test_cache(
        repo_root,
        cache_path,
        cache_metadata_path,
        experiment_key,
    )
    script_path = Path(__file__).resolve()
    gpu_tokens = resolve_cuda_tokens(gpus)
    pending = [
        plan
        for plan in plans
        if args.rerun or not result_is_complete(Path(plan["result_path"]), plan)
    ]
    completed_count = len(plans) - len(pending)
    print(f"CUDA allocation mapping: {gpu_tokens}", flush=True)
    if completed_count:
        print(f"Resume check: {completed_count} verified fits already complete.", flush=True)
    print(f"Overall progress: {progress_bar(completed_count, len(plans))}", flush=True)
    core.atomic_csv(plan_path, pd.DataFrame(plans))

    work_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    for plan in pending:
        work_queue.put(plan)
    print_lock = threading.Lock()
    state_lock = threading.Lock()
    failures: List[str] = []
    statuses = {
        plan["run_name"]: (
            "complete" if plan not in pending else "pending"
        )
        for plan in plans
    }

    def gpu_worker(logical_gpu: str) -> None:
        nonlocal completed_count
        while True:
            try:
                plan = work_queue.get_nowait()
            except queue.Empty:
                return
            run_name = str(plan["run_name"])
            command = build_worker_command(script_path, plan, args)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_tokens[logical_gpu]
            environment["EXPERIMENT_GPU_ID"] = logical_gpu
            environment["PYTHONHASHSEED"] = str(plan["seed"])
            environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            log_path = Path(plan["log_path"])
            with print_lock:
                print(f"[GPU {logical_gpu}] START {run_name}", flush=True)
            started = time.perf_counter()
            try:
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"\n\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                        f"{shlex.join(command)}\n\n"
                    )
                    log_file.flush()
                    completed = subprocess.run(
                        command,
                        cwd=repo_root,
                        env=environment,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                verified = result_is_complete(Path(plan["result_path"]), plan)
                if completed.returncode != 0 or not verified:
                    raise RuntimeError(
                        f"exit={completed.returncode}, verified={verified}, log={log_path}"
                    )
                with state_lock:
                    statuses[run_name] = "complete"
                    completed_count += 1
                    current_count = completed_count
                runtime = (time.perf_counter() - started) / 60.0
                with print_lock:
                    print(
                        f"[GPU {logical_gpu}] DONE {run_name} ({runtime:.1f} min)\n"
                        f"Overall progress: {progress_bar(current_count, len(plans))}",
                        flush=True,
                    )
            except Exception as error:
                with state_lock:
                    statuses[run_name] = "failed"
                    failures.append(f"{run_name}: {error}")
                with print_lock:
                    print(f"[GPU {logical_gpu}] FAILED {run_name}: {error}", flush=True)
            finally:
                work_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(gpu_worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()

    plan_frame = pd.DataFrame([{**plan, "status": statuses[plan["run_name"]]} for plan in plans])
    core.atomic_csv(plan_path, plan_frame)
    if failures:
        raise RuntimeError("One or more fits failed:\n" + "\n".join(failures))
    incomplete = [plan["run_name"] for plan in plans if not result_is_complete(Path(plan["result_path"]), plan)]
    if incomplete:
        raise RuntimeError(f"Result verification failed for: {incomplete}")

    output_paths = aggregate_results(
        plans,
        results_dir,
        experiment_key,
        output_stem,
    )
    latest_path = results_dir / f"{output_stem}_latest.json"
    latest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "protocol": str(protocol_path),
        "plan": str(plan_path),
        **output_paths,
    }
    core.atomic_json(latest_path, latest)
    print("\nSaved results:")
    for label, path in output_paths.items():
        print(f"  {label}: {path}")
    print(f"  latest pointer: {latest_path}")


if __name__ == "__main__":
    main()
