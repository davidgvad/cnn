"""
Leakage-free CTGAN augmentation-amount sweep on KDDTrain+ validation folds.

For every seed, this script first creates a stratified 80/20 split of the
real KDDTrain+ records. One conditional CTGAN is fitted only on the 80%
training fold and produces two separate nested pools: 5,000 R2L rows and
5,000 U2R rows. The following seven Conv2D conditions are then trained:

    no synthetic data;
    U2R-only 500, 1,000, and 5,000;
    R2L-only 500, 1,000, and 5,000.

The held-out 20% fold is always real. KDDTest+ is never loaded by this file.
One independent child process is assigned to each GPU at a time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import inspect
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


CLASS_NAMES = ["DoS", "Probe", "R2L", "U2R", "Normal"]
CLASS_IDS = {"r2l": 2, "u2r": 3}

CANONICAL_CATEGORIES = {
    "protocol_type": ["icmp", "tcp", "udp"],
    "service": [
        "IRC",
        "X11",
        "Z39_50",
        "aol",
        "auth",
        "bgp",
        "courier",
        "csnet_ns",
        "ctf",
        "daytime",
        "discard",
        "domain",
        "domain_u",
        "echo",
        "eco_i",
        "ecr_i",
        "efs",
        "exec",
        "finger",
        "ftp",
        "ftp_data",
        "gopher",
        "harvest",
        "hostnames",
        "http",
        "http_2784",
        "http_443",
        "http_8001",
        "imap4",
        "iso_tsap",
        "klogin",
        "kshell",
        "ldap",
        "link",
        "login",
        "mtp",
        "name",
        "netbios_dgm",
        "netbios_ns",
        "netbios_ssn",
        "netstat",
        "nnsp",
        "nntp",
        "ntp_u",
        "other",
        "pm_dump",
        "pop_2",
        "pop_3",
        "printer",
        "private",
        "red_i",
        "remote_job",
        "rje",
        "shell",
        "smtp",
        "sql_net",
        "ssh",
        "sunrpc",
        "supdup",
        "systat",
        "telnet",
        "tftp_u",
        "tim_i",
        "time",
        "urh_i",
        "urp_i",
        "uucp",
        "uucp_path",
        "vmnet",
        "whois",
    ],
    "flag": [
        "OTH",
        "REJ",
        "RSTO",
        "RSTOS0",
        "RSTR",
        "S0",
        "S1",
        "S2",
        "S3",
        "SF",
        "SH",
    ],
}

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

METRICS = [
    "accuracy",
    "mcc",
    "macro_f1",
    "macro_recall",
    "rare_f1",
    "minimum_minority_recall",
    "r2l_precision",
    "r2l_recall",
    "r2l_f1",
    "u2r_precision",
    "u2r_recall",
    "u2r_f1",
]


def parse_gpus(values: Sequence[str]) -> List[str]:
    gpus: List[str] = []
    for value in values:
        gpus.extend(part.strip() for part in value.split(",") if part.strip())
    if not gpus:
        raise ValueError("Provide at least one GPU ID.")
    if len(gpus) != len(set(gpus)):
        raise ValueError(f"GPU IDs must be unique, got {gpus}.")
    return gpus


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_indices(values: np.ndarray) -> str:
    array = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def source_fingerprint(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        with path.open("rb") as input_file:
            for chunk in iter(
                lambda: input_file.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def atomic_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    os.replace(temporary, path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_real_kddtrain(repo_root: Path) -> pd.DataFrame:
    # Imported only inside GPU workers so the parent does not initialize TF.
    from cnn_gan_foc import (  # type: ignore
        collapse_attack_labels,
        load_nsl_kdd_txt,
    )

    frame = load_nsl_kdd_txt(repo_root / "data" / "KDDTrain+.txt").drop(
        columns=["num_outbound_cmds"]
    )
    return collapse_attack_labels(frame, is_train=True).reset_index(drop=True)


def split_indices(
    labels: np.ndarray,
    seed: int,
    val_split: float,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    all_indices = np.arange(len(labels), dtype=np.int64)
    train_indices, val_indices = train_test_split(
        all_indices,
        test_size=float(val_split),
        random_state=int(seed),
        stratify=labels,
    )
    return (
        np.sort(np.asarray(train_indices, dtype=np.int64)),
        np.sort(np.asarray(val_indices, dtype=np.int64)),
    )


def pool_paths(
    repo_root: Path,
    prefix: str,
    experiment_key: str,
    seed: int,
) -> Dict[str, Path]:
    stem = f"{prefix}_{experiment_key}_s{seed}"
    return {
        "r2l": repo_root / "data" / f"{stem}_r2l_pool.csv",
        "u2r": repo_root / "data" / f"{stem}_u2r_pool.csv",
        "metadata": repo_root / "results" / f"{stem}_ctgan_metadata.json",
        "fold": repo_root / "results" / f"{stem}_fold_indices.npz",
    }


def condition_name(target_class: str | None, amount: int) -> str:
    return "baseline" if target_class is None else f"{target_class}_{amount}"


def condition_order(amounts: Sequence[int]) -> List[str]:
    return [
        "baseline",
        *[f"u2r_{amount}" for amount in amounts],
        *[f"r2l_{amount}" for amount in amounts],
    ]


def effective_ctgan_batch_size(batch_size: int, pac: int) -> int:
    adjusted = int(batch_size) - (int(batch_size) % int(pac))
    return adjusted if adjusted > 0 else int(pac)


def build_conditions(amounts: Sequence[int]) -> List[Dict[str, Any]]:
    conditions: List[Dict[str, Any]] = [
        {"target_class": None, "class_id": None, "amount": 0}
    ]
    for target_class in ("u2r", "r2l"):
        for amount in amounts:
            conditions.append(
                {
                    "target_class": target_class,
                    "class_id": CLASS_IDS[target_class],
                    "amount": int(amount),
                }
            )
    return conditions


def expected_pool_metadata(
    args: argparse.Namespace,
    experiment_key: str,
    train_data_sha256: str,
    seed: int,
) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_key": experiment_key,
        "seed": int(seed),
        "val_split": float(args.val_split),
        "max_amount": int(max(args.amounts)),
        "ctgan_epochs": int(args.ctgan_epochs),
        "ctgan_batch_size_requested": int(args.ctgan_batch_size),
        "ctgan_batch_size_effective": effective_ctgan_batch_size(
            args.ctgan_batch_size,
            args.ctgan_pac,
        ),
        "ctgan_pac": int(args.ctgan_pac),
        "train_data_sha256": train_data_sha256,
        "ctgan_version": package_version("ctgan"),
        "torch_version": package_version("torch"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    }


def pool_is_complete(
    paths: Dict[str, Path],
    expected: Dict[str, Any],
) -> bool:
    if not all(paths[key].exists() for key in paths):
        return False
    try:
        metadata = read_json(paths["metadata"])
        for key, value in expected.items():
            if metadata.get(key) != value:
                return False
        max_amount = int(expected["max_amount"])
        for class_name, class_id in CLASS_IDS.items():
            frame = pd.read_csv(paths[class_name], usecols=["class"])
            if len(frame) != max_amount:
                return False
            labels = pd.to_numeric(frame["class"], errors="raise")
            if not bool((labels == class_id).all()):
                return False
            if metadata.get(f"{class_name}_pool_sha256") != sha256_file(
                paths[class_name]
            ):
                return False
        with np.load(paths["fold"]) as folds:
            train_indices = folds["train_indices"]
            val_indices = folds["val_indices"]
        if metadata.get("train_indices_sha256") != sha256_indices(train_indices):
            return False
        if metadata.get("val_indices_sha256") != sha256_indices(val_indices):
            return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run_generate_worker(args: argparse.Namespace, repo_root: Path) -> None:
    import torch

    visible_gpus = int(torch.cuda.device_count())
    if not args.allow_cpu and (not torch.cuda.is_available() or visible_gpus != 1):
        raise RuntimeError(
            "CTGAN worker expected exactly one usable GPU, but PyTorch sees "
            f"{visible_gpus}. Fix CUDA or pass --allow-cpu intentionally."
        )

    from cnn_gan_foc import _generate_synth_ctgan  # type: ignore

    seed = int(args.worker_seed)
    real = load_real_kddtrain(repo_root)
    labels = real["class"].to_numpy(dtype=np.int64)
    train_indices, val_indices = split_indices(
        labels,
        seed,
        args.val_split,
    )
    train_fold = real.iloc[train_indices].copy().reset_index(drop=True)
    train_counts = np.bincount(
        train_fold["class"].to_numpy(dtype=np.int64),
        minlength=5,
    )
    val_counts = np.bincount(labels[val_indices], minlength=5)
    max_amount = int(max(args.amounts))
    targets = {
        0: 0,
        1: 0,
        2: int(train_counts[2]) + max_amount,
        3: int(train_counts[3]) + max_amount,
        4: 0,
    }

    start = time.perf_counter()
    synthetic = _generate_synth_ctgan(
        train_df_raw=train_fold,
        seed=seed,
        epochs=args.ctgan_epochs,
        batch_size=args.ctgan_batch_size,
        pac=args.ctgan_pac,
        targets=targets,
    )
    runtime = time.perf_counter() - start

    if list(synthetic.columns) != list(train_fold.columns):
        synthetic = synthetic.reindex(columns=train_fold.columns)
    if synthetic.isna().any().any():
        missing = synthetic.columns[synthetic.isna().any()].tolist()
        raise ValueError(f"CTGAN generated missing values in columns: {missing}")

    output_paths = {
        "r2l": Path(args.worker_r2l_pool),
        "u2r": Path(args.worker_u2r_pool),
        "metadata": Path(args.worker_pool_metadata),
        "fold": Path(args.worker_fold_path),
    }
    pool_frames: Dict[str, pd.DataFrame] = {}
    for class_name, class_id in CLASS_IDS.items():
        class_pool = synthetic[synthetic["class"] == class_id].copy()
        if len(class_pool) != max_amount:
            raise ValueError(
                f"Expected {max_amount} generated {class_name.upper()} rows, "
                f"got {len(class_pool)}."
            )
        class_pool = class_pool.sample(
            frac=1.0,
            random_state=seed + class_id * 10_000,
        ).reset_index(drop=True)
        atomic_csv(output_paths[class_name], class_pool)
        pool_frames[class_name] = class_pool

    atomic_npz(
        output_paths["fold"],
        train_indices=train_indices,
        val_indices=val_indices,
    )
    effective_batch = effective_ctgan_batch_size(
        args.ctgan_batch_size,
        args.ctgan_pac,
    )
    metadata: Dict[str, Any] = {
        **expected_pool_metadata(
            args,
            args.experiment_key,
            sha256_file(repo_root / "data" / "KDDTrain+.txt"),
            seed,
        ),
        "protocol": "split_real_first_fit_ctgan_on_train_fold_only",
        "kddtest_accessed": False,
        "train_indices_sha256": sha256_indices(train_indices),
        "val_indices_sha256": sha256_indices(val_indices),
        "real_full_counts": np.bincount(labels, minlength=5).tolist(),
        "real_train_counts": train_counts.tolist(),
        "real_validation_counts": val_counts.tolist(),
        "r2l_pool_count": len(pool_frames["r2l"]),
        "u2r_pool_count": len(pool_frames["u2r"]),
        "r2l_pool_sha256": sha256_file(output_paths["r2l"]),
        "u2r_pool_sha256": sha256_file(output_paths["u2r"]),
        "fold_file_sha256": sha256_file(output_paths["fold"]),
        "ctgan_runtime_seconds": runtime,
        "ctgan_version": package_version("ctgan"),
        "torch_version": package_version("torch"),
        "ctgan_cuda_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES",
            "",
        ),
        "assigned_gpu": os.environ.get("EXPERIMENT_GPU_ID", ""),
        "torch_visible_gpu_count": visible_gpus,
        "ctgan_batch_size_effective": effective_batch,
        "ctgan_discrete_columns": [
            "protocol_type",
            "service",
            "flag",
            "class",
        ],
        "ctgan_constructor_signature": str(
            inspect.signature(__import__("ctgan", fromlist=["CTGAN"]).CTGAN)
        ),
        "nested_pool_rule": "each smaller amount is a prefix of the maximum pool",
        "nested_pool_amounts": [int(amount) for amount in args.amounts],
    }
    atomic_json(output_paths["metadata"], metadata)
    print(
        f"Generated seed {seed}: R2L={len(pool_frames['r2l'])}, "
        f"U2R={len(pool_frames['u2r'])}, runtime={runtime / 60:.1f} min"
    )


def fixed_one_hot_encoders(
    train_fold: pd.DataFrame,
) -> tuple[Dict[str, Any], Dict[str, List[str]]]:
    from sklearn.preprocessing import OneHotEncoder

    encoders: Dict[str, Any] = {}
    feature_names: Dict[str, List[str]] = {}
    for column in CATEGORICAL_COLUMNS:
        encoder = OneHotEncoder(
            categories=[CANONICAL_CATEGORIES[column]],
            handle_unknown="ignore",
            sparse_output=False,
        )
        encoder.fit(train_fold[[column]])
        encoders[column] = encoder
        feature_names[column] = encoder.get_feature_names_out([column]).tolist()
    return encoders, feature_names


def calculate_validation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    labels = np.arange(5)
    recalls = recall_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    precisions = precision_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    f1_values = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_recall": float(np.mean(recalls)),
        "rare_f1": float(np.mean(f1_values[[2, 3]])),
        "minimum_minority_recall": float(np.min(recalls[[2, 3]])),
        "r2l_precision": float(precisions[2]),
        "r2l_recall": float(recalls[2]),
        "r2l_f1": float(f1_values[2]),
        "u2r_precision": float(precisions[3]),
        "u2r_recall": float(recalls[3]),
        "u2r_f1": float(f1_values[3]),
    }


def run_train_worker(args: argparse.Namespace, repo_root: Path) -> None:
    # Heavy imports stay in the child process with one visible GPU.
    import tensorflow as tf

    visible_gpus = tf.config.list_physical_devices("GPU")
    if not args.allow_cpu and len(visible_gpus) != 1:
        raise RuntimeError(
            "CNN worker expected exactly one usable GPU, but TensorFlow sees "
            f"{len(visible_gpus)}. Fix CUDA or pass --allow-cpu intentionally."
        )

    from cnn_gan_foc import (  # type: ignore
        ClassBalancedFocalLoss,
        apply_one_hot,
        apply_scaler,
        compute_cb_alpha_effective_number,
        fit_scaler,
    )
    from cnn_opt import (  # type: ignore
        BalancedBatchSequence,
        ValF1Callback,
        build_opt_cnn,
        build_optimized_feature_order,
    )

    seed = int(args.worker_seed)
    amount = int(args.worker_amount)
    target_class = (
        None if args.worker_target_class == "baseline" else args.worker_target_class
    )
    run_name = str(args.worker_run_name)
    result_path = Path(args.worker_result_path)
    pool_metadata_path = Path(args.worker_pool_metadata)
    fold_path = Path(args.worker_fold_path)
    pool_metadata = read_json(pool_metadata_path)

    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    real = load_real_kddtrain(repo_root)
    with np.load(fold_path) as folds:
        train_indices = np.asarray(folds["train_indices"], dtype=np.int64)
        val_indices = np.asarray(folds["val_indices"], dtype=np.int64)
    if sha256_indices(train_indices) != pool_metadata.get("train_indices_sha256"):
        raise ValueError("Training-fold hash does not match CTGAN metadata.")
    if sha256_indices(val_indices) != pool_metadata.get("val_indices_sha256"):
        raise ValueError("Validation-fold hash does not match CTGAN metadata.")

    train_real = real.iloc[train_indices].copy().reset_index(drop=True)
    validation_real = real.iloc[val_indices].copy().reset_index(drop=True)
    if target_class is None:
        synthetic = train_real.iloc[0:0].copy()
        selected_pool_sha256 = "none"
    else:
        pool_path = Path(
            args.worker_r2l_pool if target_class == "r2l" else args.worker_u2r_pool
        )
        selected_pool_sha256 = sha256_file(pool_path)
        expected_hash = pool_metadata.get(f"{target_class}_pool_sha256")
        if selected_pool_sha256 != expected_hash:
            raise ValueError(
                f"{target_class.upper()} pool hash does not match metadata."
            )
        pool = pd.read_csv(pool_path)
        missing_columns = sorted(set(train_real.columns) - set(pool.columns))
        extra_columns = sorted(set(pool.columns) - set(train_real.columns))
        if missing_columns or extra_columns:
            raise ValueError(
                "Synthetic pool schema does not match the real training data: "
                f"missing={missing_columns}, extra={extra_columns}."
            )
        pool = pool[train_real.columns]
        class_id = CLASS_IDS[target_class]
        if len(pool) < amount:
            raise ValueError(
                f"Pool has {len(pool)} rows, but condition needs {amount}."
            )
        synthetic = pool.iloc[:amount].copy().reset_index(drop=True)
        synth_labels = pd.to_numeric(
            synthetic["class"],
            errors="raise",
        )
        if not bool((synth_labels == class_id).all()):
            raise ValueError(
                f"{target_class.upper()} condition contains another class."
            )

    encoders, feature_names = fixed_one_hot_encoders(train_real)
    train_real_ohe = apply_one_hot(
        train_real,
        encoders,
        feature_names,
        CATEGORICAL_COLUMNS,
    )
    validation_ohe = apply_one_hot(
        validation_real,
        encoders,
        feature_names,
        CATEGORICAL_COLUMNS,
    )
    scaler = fit_scaler(train_real_ohe, COLUMNS_TO_SCALE)
    train_real_proc = apply_scaler(
        train_real_ohe,
        scaler,
        COLUMNS_TO_SCALE,
    )
    validation_proc = apply_scaler(
        validation_ohe,
        scaler,
        COLUMNS_TO_SCALE,
    )
    if synthetic.empty:
        # MinMaxScaler.transform rejects an empty dataframe. Reuse the
        # already-processed schema for the no-augmentation baseline.
        synthetic_proc = train_real_proc.iloc[0:0].copy()
    else:
        synthetic_ohe = apply_one_hot(
            synthetic,
            encoders,
            feature_names,
            CATEGORICAL_COLUMNS,
        )
        synthetic_proc = apply_scaler(
            synthetic_ohe,
            scaler,
            COLUMNS_TO_SCALE,
        )
    training_proc = pd.concat(
        [train_real_proc, synthetic_proc],
        ignore_index=True,
    )
    training_proc = training_proc.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)
    validation_proc = validation_proc[training_proc.columns]

    feature_columns = [column for column in training_proc.columns if column != "class"]
    if len(feature_columns) != 121:
        raise ValueError(
            f"Expected canonical 121 features, got {len(feature_columns)}."
        )
    ordered_features = build_optimized_feature_order(feature_columns)
    ordered_columns = [*ordered_features, "class"]
    training_proc = training_proc[ordered_columns]
    validation_proc = validation_proc[ordered_columns]
    feature_order_sha256 = hashlib.sha256(
        "\n".join(ordered_features).encode("utf-8")
    ).hexdigest()

    y_train = training_proc["class"].to_numpy(dtype=np.int64)
    y_val = validation_proc["class"].to_numpy(dtype=np.int64)
    X_train = (
        training_proc.drop(columns=["class"])
        .to_numpy(dtype=np.float32)
        .reshape(-1, 11, 11, 1)
    )
    X_val = (
        validation_proc.drop(columns=["class"])
        .to_numpy(dtype=np.float32)
        .reshape(-1, 11, 11, 1)
    )

    if args.loss_mode == "class_balanced_focal":
        alpha_labels = (
            train_real_proc["class"].to_numpy(dtype=np.int64)
            if args.alpha_source == "real"
            else y_train
        )
        alpha, alpha_counts = compute_cb_alpha_effective_number(
            alpha_labels,
            beta=args.cb_beta,
            num_classes=5,
        )
        loss: tf.keras.losses.Loss = ClassBalancedFocalLoss(
            alpha=alpha,
            gamma=args.focal_gamma,
        )
    else:
        alpha = np.ones(5, dtype=np.float32)
        alpha_counts = np.bincount(y_train, minlength=5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

    model = build_opt_cnn(
        loss=loss,
        groups=args.groups,
        base_filters=args.base_filters,
        dense_units=args.dense_units,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        use_batch_norm=not args.no_bn,
        use_residual=not args.no_residual,
    )
    model_parameters = int(model.count_params())
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=f"{run_name}_") as temp_dir:
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
        if args.minority_per_batch > 0:
            training_data = BalancedBatchSequence(
                X_train,
                y_train,
                batch_size=args.batch_size,
                minority_per_batch=args.minority_per_batch,
                seed=seed,
            )
            history = model.fit(
                training_data,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                verbose=1,
                callbacks=callbacks,
            )
        else:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                verbose=1,
                callbacks=callbacks,
            )
        if not weights_path.exists():
            raise RuntimeError("Best validation checkpoint was not created.")
        model.load_weights(weights_path)

    probabilities = model.predict(
        X_val,
        batch_size=args.batch_size,
        verbose=0,
    )
    predictions = np.argmax(probabilities, axis=1)
    metrics = calculate_validation_metrics(y_val, predictions)
    runtime = time.perf_counter() - start
    val_history = history.history.get("val_macro_f1", [])
    best_epoch = int(np.argmax(val_history)) + 1 if val_history else None
    result: Dict[str, Any] = {
        "schema_version": 1,
        "experiment_key": args.experiment_key,
        "run_name": run_name,
        "condition": condition_name(target_class, amount),
        "target_class": target_class or "none",
        "synthetic_amount": amount,
        "seed": seed,
        "evaluation_data": "KDDTrain+ held-out real validation fold",
        "kddtest_accessed": False,
        "decision_policy": "raw_argmax",
        "val_split": float(args.val_split),
        "train_indices_sha256": sha256_indices(train_indices),
        "val_indices_sha256": sha256_indices(val_indices),
        "pool_metadata_path": str(pool_metadata_path),
        "selected_pool_sha256": selected_pool_sha256,
        "nested_pool_prefix_rows": amount,
        "real_train_counts": np.bincount(
            train_real["class"].to_numpy(dtype=np.int64),
            minlength=5,
        ).tolist(),
        "synthetic_counts": np.bincount(
            synthetic["class"].to_numpy(dtype=np.int64),
            minlength=5,
        ).tolist(),
        "augmented_training_counts": np.bincount(
            y_train,
            minlength=5,
        ).tolist(),
        "real_validation_counts": np.bincount(
            y_val,
            minlength=5,
        ).tolist(),
        "loss_mode": args.loss_mode,
        "cb_beta": float(args.cb_beta),
        "focal_gamma": float(args.focal_gamma),
        "alpha_source": args.alpha_source,
        "alpha_counts": np.asarray(alpha_counts).astype(int).tolist(),
        "alpha": np.asarray(alpha).astype(float).tolist(),
        "minority_per_batch": int(args.minority_per_batch),
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "best_epoch": best_epoch,
        "batch_size": int(args.batch_size),
        "early_stopping_patience": int(args.early_stopping_patience),
        "optimizer": "keras_adam_defaults",
        "groups": int(args.groups),
        "base_filters": int(args.base_filters),
        "dense_units": int(args.dense_units),
        "dropout1": float(args.dropout1),
        "dropout2": float(args.dropout2),
        "use_batch_norm": not args.no_bn,
        "use_residual": not args.no_residual,
        "model_parameters": model_parameters,
        "feature_count": len(ordered_features),
        "feature_order_sha256": feature_order_sha256,
        "categorical_vocabulary": CANONICAL_CATEGORIES,
        "preprocessing_fit_data": "real training fold only",
        "runtime_seconds": runtime,
        "tensorflow_version": package_version("tensorflow"),
        "keras_version": package_version("keras"),
        "scikit_learn_version": package_version("scikit-learn"),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "assigned_gpu": os.environ.get("EXPERIMENT_GPU_ID", ""),
        "tensorflow_visible_gpu_count": len(visible_gpus),
        **metrics,
    }
    atomic_json(result_path, result)
    text_path = result_path.with_suffix(".txt")
    lines = [
        "Leakage-free CTGAN amount sweep (validation only)",
        "",
        f"run_name: {run_name}",
        f"condition: {result['condition']}",
        f"seed: {seed}",
        f"synthetic_amount: {amount}",
        f"target_class: {result['target_class']}",
        "KDDTest+ accessed: False",
        f"Validation Accuracy: {metrics['accuracy']}",
        f"Validation MCC: {metrics['mcc']}",
        f"Validation Macro F1: {metrics['macro_f1']}",
        f"Validation Macro Recall: {metrics['macro_recall']}",
        f"Validation R2L Precision: {metrics['r2l_precision']}",
        f"Validation R2L Recall: {metrics['r2l_recall']}",
        f"Validation R2L F1: {metrics['r2l_f1']}",
        f"Validation U2R Precision: {metrics['u2r_precision']}",
        f"Validation U2R Recall: {metrics['u2r_recall']}",
        f"Validation U2R F1: {metrics['u2r_f1']}",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if args.save_predictions:
        prediction_path = result_path.with_name(
            f"{result_path.stem}_validation_predictions.npz"
        )
        atomic_npz(
            prediction_path,
            y_true=y_val,
            y_pred=predictions,
            probabilities=probabilities,
            validation_indices=val_indices,
        )
    print(
        f"Completed {run_name}: val_macro_f1={metrics['macro_f1']:.4f}, "
        f"R2L_F1={metrics['r2l_f1']:.4f}, "
        f"U2R_F1={metrics['u2r_f1']:.4f}"
    )


def result_is_complete(
    result_path: Path,
    plan: Dict[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not result_path.exists():
        return False
    try:
        result = read_json(result_path)
        exact_checks = {
            "experiment_key": args.experiment_key,
            "run_name": plan["run_name"],
            "condition": plan["condition"],
            "target_class": plan["target_class"] or "none",
            "synthetic_amount": plan["amount"],
            "seed": plan["seed"],
            "kddtest_accessed": False,
            "decision_policy": "raw_argmax",
            "loss_mode": args.loss_mode,
            "alpha_source": args.alpha_source,
            "minority_per_batch": args.minority_per_batch,
        }
        for key, expected in exact_checks.items():
            if result.get(key) != expected:
                return False
        numeric_checks = {
            "val_split": args.val_split,
            "cb_beta": args.cb_beta,
            "focal_gamma": args.focal_gamma,
            "batch_size": args.batch_size,
            "groups": args.groups,
            "base_filters": args.base_filters,
            "dense_units": args.dense_units,
            "dropout1": args.dropout1,
            "dropout2": args.dropout2,
        }
        for key, expected in numeric_checks.items():
            if not np.isclose(float(result.get(key)), float(expected)):
                return False
        pool_metadata = read_json(plan["pool_metadata"])
        if result.get("train_indices_sha256") != pool_metadata.get(
            "train_indices_sha256"
        ):
            return False
        if result.get("val_indices_sha256") != pool_metadata.get("val_indices_sha256"):
            return False
        synthetic_counts = [int(value) for value in result["synthetic_counts"]]
        expected_synthetic_counts = [0, 0, 0, 0, 0]
        if plan["target_class"] is not None:
            expected_synthetic_counts[CLASS_IDS[plan["target_class"]]] = int(
                plan["amount"]
            )
        if synthetic_counts != expected_synthetic_counts:
            return False
        if plan["target_class"] is not None:
            pool_path = plan[f"{plan['target_class']}_pool"]
            if result.get("selected_pool_sha256") != sha256_file(pool_path):
                return False
        for metric in METRICS:
            if not np.isfinite(float(result.get(metric))):
                return False
        if args.save_predictions:
            prediction_path = result_path.with_name(
                f"{result_path.stem}_validation_predictions.npz"
            )
            if not prediction_path.exists():
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_worker_command(
    script_path: Path,
    args: argparse.Namespace,
    mode: str,
    plan: Dict[str, Any],
) -> List[str]:
    command = [
        sys.executable,
        "-u",
        str(script_path),
        "--worker-mode",
        mode,
        "--experiment-key",
        args.experiment_key,
        "--seed",
        str(plan["seed"]),
        "--amounts",
        *[str(amount) for amount in args.amounts],
        "--val-split",
        str(args.val_split),
        "--ctgan-epochs",
        str(args.ctgan_epochs),
        "--ctgan-batch-size",
        str(args.ctgan_batch_size),
        "--ctgan-pac",
        str(args.ctgan_pac),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--loss-mode",
        args.loss_mode,
        "--cb-beta",
        str(args.cb_beta),
        "--focal-gamma",
        str(args.focal_gamma),
        "--alpha-source",
        args.alpha_source,
        "--minority-per-batch",
        str(args.minority_per_batch),
        "--groups",
        str(args.groups),
        "--base-filters",
        str(args.base_filters),
        "--dense-units",
        str(args.dense_units),
        "--dropout1",
        str(args.dropout1),
        "--dropout2",
        str(args.dropout2),
        "--worker-r2l-pool",
        str(plan["r2l_pool"]),
        "--worker-u2r-pool",
        str(plan["u2r_pool"]),
        "--worker-pool-metadata",
        str(plan["pool_metadata"]),
        "--worker-fold-path",
        str(plan["fold_path"]),
    ]
    if args.no_bn:
        command.append("--no-bn")
    if args.no_residual:
        command.append("--no-residual")
    if args.save_predictions:
        command.append("--save-predictions")
    if args.allow_cpu:
        command.append("--allow-cpu")
    if mode == "train":
        command.extend(
            [
                "--worker-target-class",
                plan["target_class"] or "baseline",
                "--worker-amount",
                str(plan["amount"]),
                "--worker-run-name",
                plan["run_name"],
                "--worker-result-path",
                str(plan["result_path"]),
            ]
        )
    return command


def run_gpu_phase(
    phase_name: str,
    plans: List[Dict[str, Any]],
    gpus: List[str],
    command_builder: Any,
    log_dir: Path,
) -> tuple[Dict[str, float], List[str]]:
    task_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    for plan in plans:
        task_queue.put(plan)
    print_lock = threading.Lock()
    data_lock = threading.Lock()
    runtimes: Dict[str, float] = {}
    failures: List[str] = []

    def worker(gpu: str) -> None:
        while True:
            try:
                plan = task_queue.get_nowait()
            except queue.Empty:
                return
            name = plan["task_name"]
            command = command_builder(plan)
            log_path = log_dir / f"{name}.log"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONHASHSEED"] = str(plan["seed"])
            environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            environment["EXPERIMENT_GPU_ID"] = gpu
            environment["EXPERIMENT_PHASE"] = phase_name
            with print_lock:
                print(f"[GPU {gpu}] START {name}", flush=True)
            start = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(shlex.join(command) + "\n\n")
                log_file.flush()
                completed = subprocess.run(
                    command,
                    cwd=log_dir.parents[1],
                    env=environment,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            runtime = time.perf_counter() - start
            with data_lock:
                runtimes[name] = runtime
                if completed.returncode != 0:
                    failures.append(
                        f"{name}: exit={completed.returncode}, log={log_path}"
                    )
            with print_lock:
                status = "DONE" if completed.returncode == 0 else "FAILED"
                print(
                    f"[GPU {gpu}] {status} {name} ({runtime / 60:.1f} min)",
                    flush=True,
                )
            task_queue.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpus]
        for future in futures:
            future.result()
    return runtimes, failures


def summarize_results(
    plans: List[Dict[str, Any]],
    args: argparse.Namespace,
    results_dir: Path,
    prefix: str,
) -> None:
    ordered_conditions = condition_order(args.amounts)
    rows: List[Dict[str, Any]] = []
    for plan in plans:
        if not result_is_complete(plan["result_path"], plan, args):
            raise ValueError(f"Incomplete result: {plan['result_path']}")
        result = read_json(plan["result_path"])
        rows.append(
            {
                "experiment_key": args.experiment_key,
                "condition": plan["condition"],
                "target_class": result["target_class"],
                "synthetic_amount": result["synthetic_amount"],
                "seed": result["seed"],
                "train_indices_sha256": result["train_indices_sha256"],
                "val_indices_sha256": result["val_indices_sha256"],
                "result_path": str(plan["result_path"]),
                **{metric: result[metric] for metric in METRICS},
            }
        )
    raw = pd.DataFrame(rows)
    raw["_order"] = raw["condition"].map(
        {name: index for index, name in enumerate(ordered_conditions)}
    )
    raw = raw.sort_values(["_order", "seed"]).drop(columns="_order")
    raw_path = results_dir / f"{prefix}_all_runs.csv"
    raw.to_csv(raw_path, index=False)

    summary_rows: List[Dict[str, Any]] = []
    for condition in ordered_conditions:
        group = raw[raw["condition"] == condition]
        if len(group) != len(args.seeds):
            raise ValueError(
                f"{condition} has {len(group)} runs; expected {len(args.seeds)}."
            )
        row: Dict[str, Any] = {
            "condition": condition,
            "target_class": group["target_class"].iloc[0],
            "synthetic_amount": int(group["synthetic_amount"].iloc[0]),
            "runs": len(group),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = results_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    baseline = raw[raw["condition"] == "baseline"].set_index("seed")
    delta_rows: List[Dict[str, Any]] = []
    for condition in ordered_conditions[1:]:
        group = raw[raw["condition"] == condition].set_index("seed")
        for seed in args.seeds:
            delta_rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    **{
                        f"delta_{metric}": (
                            float(group.loc[seed, metric])
                            - float(baseline.loc[seed, metric])
                        )
                        for metric in METRICS
                    },
                }
            )
    deltas = pd.DataFrame(delta_rows)
    delta_path = results_dir / f"{prefix}_paired_deltas_vs_baseline.csv"
    deltas.to_csv(delta_path, index=False)

    selection_rows: List[Dict[str, Any]] = []
    for target_class in ("u2r", "r2l"):
        metric = f"{target_class}_f1_mean"
        candidates = pd.concat(
            [
                summary[summary["condition"] == "baseline"],
                summary[summary["target_class"] == target_class],
            ],
            ignore_index=True,
        ).copy()
        candidates = candidates.sort_values(
            ["macro_f1_mean", metric, "synthetic_amount"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        candidates.insert(0, "validation_rank", np.arange(1, len(candidates) + 1))
        candidates.insert(0, "selection_family", target_class.upper())
        selection_rows.extend(candidates.to_dict(orient="records"))
    selection = pd.DataFrame(selection_rows)
    selection_path = results_dir / f"{prefix}_validation_ranking.csv"
    selection.to_csv(selection_path, index=False)

    def percent(mean: float, std: float) -> str:
        return f"{100 * mean:.2f}% +/- {100 * std:.2f}%"

    formatted_rows: List[Dict[str, Any]] = []
    for row in summary.to_dict(orient="records"):
        formatted_rows.append(
            {
                "Condition": row["condition"],
                "Runs": row["runs"],
                "Accuracy": percent(row["accuracy_mean"], row["accuracy_std"]),
                "MCC": percent(row["mcc_mean"], row["mcc_std"]),
                "Macro-F1": percent(row["macro_f1_mean"], row["macro_f1_std"]),
                "Macro Recall": percent(
                    row["macro_recall_mean"], row["macro_recall_std"]
                ),
                "R2L Precision": percent(
                    row["r2l_precision_mean"], row["r2l_precision_std"]
                ),
                "R2L Recall": percent(row["r2l_recall_mean"], row["r2l_recall_std"]),
                "R2L F1": percent(row["r2l_f1_mean"], row["r2l_f1_std"]),
                "U2R Precision": percent(
                    row["u2r_precision_mean"], row["u2r_precision_std"]
                ),
                "U2R Recall": percent(row["u2r_recall_mean"], row["u2r_recall_std"]),
                "U2R F1": percent(row["u2r_f1_mean"], row["u2r_f1_std"]),
            }
        )
    formatted = pd.DataFrame(formatted_rows)
    formatted_path = results_dir / f"{prefix}_summary_formatted.csv"
    formatted.to_csv(formatted_path, index=False)
    text_path = results_dir / f"{prefix}_summary.txt"
    text_path.write_text(
        "Leakage-free CTGAN amount sweep\n"
        "Evaluation: real held-out KDDTrain+ validation only\n"
        "KDDTest+ accessed: no\n"
        "Selection order: validation Macro-F1, then target-class F1\n\n"
        + formatted.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )
    print("\n=== Validation-only CTGAN amount sweep ===")
    print(formatted.to_string(index=False))
    print(f"\nPer-seed results: {raw_path}")
    print(f"Mean/sample-SD summary: {summary_path}")
    print(f"Formatted summary: {formatted_path}")
    print(f"Paired deltas: {delta_path}")
    print(f"Validation ranking: {selection_path}")
    print(f"Readable table: {text_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-free R2L/U2R CTGAN amount sweep using KDDTrain+ validation only."
        )
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--amounts",
        type=int,
        nargs="+",
        default=[500, 1000, 5000],
    )
    parser.add_argument("--name-prefix", default="ctgan_amount_sweep")
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--ctgan-epochs", type=int, default=200)
    parser.add_argument("--ctgan-batch-size", type=int, default=4096)
    parser.add_argument("--ctgan-pac", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument(
        "--loss-mode",
        choices=["cross_entropy", "class_balanced_focal"],
        default="class_balanced_focal",
    )
    parser.add_argument("--cb-beta", type=float, default=0.99)
    parser.add_argument("--focal-gamma", type=float, default=0.5)
    parser.add_argument(
        "--alpha-source",
        choices=["augmented", "real"],
        default="real",
        help=(
            "For focal loss, freeze class weights from real fold data "
            "(controlled default) or recompute them per augmented condition."
        ),
    )
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout1", type=float, default=0.25)
    parser.add_argument("--dropout2", type=float, default=0.30)
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--no-residual", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow workers to run without exactly one visible GPU.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--regenerate-pools", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    # Internal subprocess options.
    parser.add_argument(
        "--worker-mode",
        choices=["none", "generate", "train"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--experiment-key", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--seed", dest="worker_seed", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-r2l-pool", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-u2r-pool", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-pool-metadata", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-fold-path", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-target-class", default="baseline", help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-amount", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds cannot contain negative values.")
    if not args.amounts or len(args.amounts) != len(set(args.amounts)):
        parser.error("--amounts must contain unique values.")
    args.amounts = sorted(args.amounts)
    if any(amount <= 0 for amount in args.amounts):
        parser.error("Every --amounts value must be greater than zero.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between zero and one.")
    if args.ctgan_epochs <= 0 or args.ctgan_batch_size <= 0:
        parser.error("CTGAN epochs and batch size must be positive.")
    if args.ctgan_pac <= 0:
        parser.error("--ctgan-pac must be positive.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("CNN epochs and batch size must be positive.")
    if args.early_stopping_patience < 0:
        parser.error("Early-stopping patience cannot be negative.")
    if not 0.0 < args.cb_beta < 1.0:
        parser.error("--cb-beta must be between zero and one.")
    if args.focal_gamma < 0.0:
        parser.error("--focal-gamma cannot be negative.")
    if args.minority_per_batch < 0:
        parser.error("--minority-per-batch cannot be negative.")
    if args.groups <= 0 or args.base_filters % args.groups != 0:
        parser.error("--groups must divide --base-filters.")
    if args.prepare_only and args.train_only:
        parser.error("--prepare-only and --train-only are mutually exclusive.")
    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))

    if args.worker_mode != "none":
        if not args.experiment_key:
            parser.error("Internal worker is missing --experiment-key.")
        if args.worker_mode == "generate":
            run_generate_worker(args, repo_root)
        else:
            run_train_worker(args, repo_root)
        return

    train_data_path = repo_root / "data" / "KDDTrain+.txt"
    if not train_data_path.exists():
        raise SystemExit(f"KDDTrain+ not found: {train_data_path}")
    settings_identity = (
        tuple(args.seeds),
        tuple(args.amounts),
        args.val_split,
        args.ctgan_epochs,
        args.ctgan_batch_size,
        args.ctgan_pac,
        args.epochs,
        args.batch_size,
        args.early_stopping_patience,
        args.loss_mode,
        args.cb_beta,
        args.focal_gamma,
        args.alpha_source,
        args.minority_per_batch,
        args.groups,
        args.base_filters,
        args.dense_units,
        args.dropout1,
        args.dropout2,
        not args.no_bn,
        not args.no_residual,
        args.allow_cpu,
        package_version("ctgan"),
        package_version("torch"),
        package_version("tensorflow"),
        package_version("keras"),
        package_version("scikit-learn"),
        np.__version__,
        pd.__version__,
        source_fingerprint(
            [
                script_path,
                repo_root / "src" / "cnn_opt.py",
                repo_root / "src" / "cnn_gan_foc.py",
                train_data_path,
            ]
        ),
    )
    args.experiment_key = hashlib.sha256(
        repr(settings_identity).encode("utf-8")
    ).hexdigest()[:12]
    prefix = args.name_prefix.strip() or "ctgan_amount_sweep"
    results_dir = repo_root / "results"
    log_dir = results_dir / f"{prefix}_{args.experiment_key}_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_data_sha256 = sha256_file(train_data_path)

    pool_plans: List[Dict[str, Any]] = []
    for seed in args.seeds:
        paths = pool_paths(repo_root, prefix, args.experiment_key, seed)
        pool_plans.append(
            {
                "task_name": f"{prefix}_{args.experiment_key}_ctgan_s{seed}",
                "seed": seed,
                "r2l_pool": paths["r2l"],
                "u2r_pool": paths["u2r"],
                "pool_metadata": paths["metadata"],
                "fold_path": paths["fold"],
            }
        )

    training_plans: List[Dict[str, Any]] = []
    pool_by_seed = {plan["seed"]: plan for plan in pool_plans}
    for condition in build_conditions(args.amounts):
        condition_label = condition_name(
            condition["target_class"],
            condition["amount"],
        )
        for seed in args.seeds:
            pool_plan = pool_by_seed[seed]
            run_name = f"{prefix}_{args.experiment_key}_{condition_label}_s{seed}"
            training_plans.append(
                {
                    **condition,
                    **pool_plan,
                    "task_name": run_name,
                    "run_name": run_name,
                    "condition": condition_label,
                    "seed": seed,
                    "result_path": results_dir / f"{run_name}_results.json",
                }
            )

    protocol = {
        "experiment_key": args.experiment_key,
        "purpose": "select R2L and U2R CTGAN amounts on validation only",
        "kddtest_accessed": False,
        "seeds": args.seeds,
        "amounts": args.amounts,
        "conditions_per_seed": len(build_conditions(args.amounts)),
        "ctgan_fits": len(args.seeds),
        "cnn_trainings": len(training_plans),
        "split": "stratified 80% real training / 20% real validation",
        "ctgan_fit_data": "80% real training fold only",
        "preprocessing_fit_data": "80% real training fold only",
        "pool_rule": "separate nested R2L and U2R prefixes",
        "classifier_never_mixes_r2l_and_u2r_synthetic_rows": True,
        "decision_policy": "raw_argmax",
        "primary_selection_metric": "validation macro-F1",
        "secondary_selection_metric": "target-class validation F1",
        "settings": {
            "val_split": args.val_split,
            "ctgan_epochs": args.ctgan_epochs,
            "ctgan_batch_size_requested": args.ctgan_batch_size,
            "ctgan_batch_size_effective": effective_ctgan_batch_size(
                args.ctgan_batch_size,
                args.ctgan_pac,
            ),
            "ctgan_pac": args.ctgan_pac,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "early_stopping_patience": args.early_stopping_patience,
            "loss_mode": args.loss_mode,
            "cb_beta": args.cb_beta,
            "focal_gamma": args.focal_gamma,
            "alpha_source": args.alpha_source,
            "minority_per_batch": args.minority_per_batch,
            "groups": args.groups,
            "base_filters": args.base_filters,
            "dense_units": args.dense_units,
            "dropout1": args.dropout1,
            "dropout2": args.dropout2,
            "batch_norm": not args.no_bn,
            "residual": not args.no_residual,
            "allow_cpu": args.allow_cpu,
            "optimizer": "Adam with Keras defaults",
        },
        "library_versions": {
            "ctgan": package_version("ctgan"),
            "torch": package_version("torch"),
            "tensorflow": package_version("tensorflow"),
            "keras": package_version("keras"),
            "scikit_learn": package_version("scikit-learn"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "canonical_categorical_vocabulary": CANONICAL_CATEGORIES,
        "kddtrain_sha256": train_data_sha256,
    }
    protocol_path = results_dir / f"{prefix}_{args.experiment_key}_protocol.json"
    atomic_json(protocol_path, protocol)
    plan_path = results_dir / f"{prefix}_{args.experiment_key}_plan.csv"
    plan_rows = []
    for index, plan in enumerate(pool_plans):
        plan_rows.append(
            {
                "phase": "ctgan",
                "task_name": plan["task_name"],
                "seed": plan["seed"],
                "planned_gpu": gpus[index % len(gpus)],
                "target_class": "R2L+U2R separate pools",
                "synthetic_amount": max(args.amounts),
                "output": str(plan["pool_metadata"]),
            }
        )
    for index, plan in enumerate(training_plans):
        plan_rows.append(
            {
                "phase": "cnn",
                "task_name": plan["task_name"],
                "seed": plan["seed"],
                "planned_gpu": gpus[index % len(gpus)],
                "target_class": plan["target_class"] or "none",
                "synthetic_amount": plan["amount"],
                "output": str(plan["result_path"]),
            }
        )
    pd.DataFrame(plan_rows).to_csv(plan_path, index=False)

    print("Leakage-free CTGAN amount sweep")
    print(f"Experiment key: {args.experiment_key}")
    print(f"Seeds: {args.seeds}")
    print(f"Amounts per class: {args.amounts}")
    print(f"GPUs: {gpus}")
    print(f"CTGAN fits: {len(pool_plans)}")
    print(f"CNN trainings: {len(training_plans)}")
    print("Evaluation: real KDDTrain+ validation only")
    print("KDDTest+ accessed: NO")
    print(f"Protocol: {protocol_path}")
    print(f"Plan: {plan_path}")

    if args.dry_run:
        for plan in pool_plans:
            print(
                "[CTGAN] "
                + shlex.join(
                    build_worker_command(
                        script_path,
                        args,
                        "generate",
                        plan,
                    )
                )
            )
        for plan in training_plans:
            print(
                "[CNN] "
                + shlex.join(
                    build_worker_command(
                        script_path,
                        args,
                        "train",
                        plan,
                    )
                )
            )
        print("Dry run complete; no CTGAN or CNN was trained.")
        return

    expected_by_seed = {
        seed: expected_pool_metadata(
            args,
            args.experiment_key,
            train_data_sha256,
            seed,
        )
        for seed in args.seeds
    }
    if not args.train_only:
        pending_pools = [
            plan
            for plan in pool_plans
            if args.regenerate_pools
            or not pool_is_complete(
                {
                    "r2l": plan["r2l_pool"],
                    "u2r": plan["u2r_pool"],
                    "metadata": plan["pool_metadata"],
                    "fold": plan["fold_path"],
                },
                expected_by_seed[plan["seed"]],
            )
        ]
        if pending_pools:
            _, failures = run_gpu_phase(
                "ctgan",
                pending_pools,
                gpus,
                lambda plan: build_worker_command(
                    script_path,
                    args,
                    "generate",
                    plan,
                ),
                log_dir,
            )
            if failures:
                print("\nCTGAN failures:")
                for failure in failures:
                    print(f"- {failure}")
                raise SystemExit(1)
        else:
            print("All fold-specific CTGAN pools are already complete.")

    for plan in pool_plans:
        paths = {
            "r2l": plan["r2l_pool"],
            "u2r": plan["u2r_pool"],
            "metadata": plan["pool_metadata"],
            "fold": plan["fold_path"],
        }
        if not pool_is_complete(paths, expected_by_seed[plan["seed"]]):
            raise SystemExit(
                f"Missing or incompatible CTGAN pool for seed {plan['seed']}. "
                "Run without --train-only or use --regenerate-pools."
            )

    if args.prepare_only:
        print("CTGAN preparation complete; CNN training was skipped.")
        return

    pending_trainings = [
        plan
        for plan in training_plans
        if args.rerun or not result_is_complete(plan["result_path"], plan, args)
    ]
    if pending_trainings:
        _, failures = run_gpu_phase(
            "cnn",
            pending_trainings,
            gpus,
            lambda plan: build_worker_command(
                script_path,
                args,
                "train",
                plan,
            ),
            log_dir,
        )
        if failures:
            print("\nCNN failures:")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(1)
    else:
        print("All CNN validation runs are already complete.")

    summarize_results(
        training_plans,
        args,
        results_dir,
        f"{prefix}_{args.experiment_key}",
    )


if __name__ == "__main__":
    main()
