"""
Run a leakage-controlled, no-CTGAN ablation for CNN, Transformer, and XGBoost.

Every seed starts by splitting raw KDDTrain+ rows into a stratified 80/20
training/validation fold. The fixed 121-feature encoder and MinMax scaler are
fit on the 80% fold only. KDDTest+ is not loaded or transformed until the model
has finished training and any score-scaling coefficients have been selected
from validation probabilities.

Eight unique models are trained per seed and expanded into eleven report rows:

  CNN:         CE; focal; focal+minority batches; same batch model+scaling
  Transformer: CE; focal; focal+minority batches; same batch model+scaling
  XGBoost:     standard; balanced weights; same weighted model+scaling

The score-scaled rows reuse the exact trained model and only change the frozen
post-training decision rule. No synthetic file is read and no synthetic row is
used anywhere in this script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import queue
import re
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
CLASS_TO_ID = {"DoS": 0, "Probe": 1, "R2L": 2, "U2R": 3, "normal": 4}

NSL_KDD_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
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
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
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
    "class",
    "difficulty",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
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

TRAINING_CONFIGS: List[Dict[str, Any]] = [
    {
        "training_id": "cnn_ce",
        "family": "cnn",
        "loss_mode": "cross_entropy",
        "minority_batches": False,
        "class_weighting": "none",
        "supports_scaled_row": False,
    },
    {
        "training_id": "cnn_focal",
        "family": "cnn",
        "loss_mode": "class_balanced_focal",
        "minority_batches": False,
        "class_weighting": "effective_number",
        "supports_scaled_row": False,
    },
    {
        "training_id": "cnn_focal_batch",
        "family": "cnn",
        "loss_mode": "class_balanced_focal",
        "minority_batches": True,
        "class_weighting": "effective_number",
        "supports_scaled_row": True,
    },
    {
        "training_id": "transformer_ce",
        "family": "transformer",
        "loss_mode": "cross_entropy",
        "minority_batches": False,
        "class_weighting": "none",
        "supports_scaled_row": False,
    },
    {
        "training_id": "transformer_focal",
        "family": "transformer",
        "loss_mode": "class_balanced_focal",
        "minority_batches": False,
        "class_weighting": "effective_number",
        "supports_scaled_row": False,
    },
    {
        "training_id": "transformer_focal_batch",
        "family": "transformer",
        "loss_mode": "class_balanced_focal",
        "minority_batches": True,
        "class_weighting": "effective_number",
        "supports_scaled_row": True,
    },
    {
        "training_id": "xgboost_standard",
        "family": "xgboost",
        "loss_mode": "not_applicable",
        "minority_batches": False,
        "class_weighting": "none",
        "supports_scaled_row": False,
    },
    {
        "training_id": "xgboost_balanced",
        "family": "xgboost",
        "loss_mode": "not_applicable",
        "minority_batches": False,
        "class_weighting": "balanced",
        "supports_scaled_row": True,
    },
]

REPORT_ROWS: List[Dict[str, str]] = [
    {
        "condition": "cnn_ce_raw",
        "display_name": "CNN baseline (CE)",
        "training_id": "cnn_ce",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "cnn_focal_raw",
        "display_name": "CNN + class-balanced focal",
        "training_id": "cnn_focal",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "cnn_focal_batch_raw",
        "display_name": "CNN + focal + minority batches",
        "training_id": "cnn_focal_batch",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "cnn_focal_batch_scaled",
        "display_name": "CNN + focal + batches + scaling",
        "training_id": "cnn_focal_batch",
        "metrics_key": "scaled_test_metrics",
    },
    {
        "condition": "transformer_ce_raw",
        "display_name": "Transformer baseline (CE)",
        "training_id": "transformer_ce",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "transformer_focal_raw",
        "display_name": "Transformer + class-balanced focal",
        "training_id": "transformer_focal",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "transformer_focal_batch_raw",
        "display_name": "Transformer + focal + minority batches",
        "training_id": "transformer_focal_batch",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "transformer_focal_batch_scaled",
        "display_name": "Transformer + focal + batches + scaling",
        "training_id": "transformer_focal_batch",
        "metrics_key": "scaled_test_metrics",
    },
    {
        "condition": "xgboost_standard_raw",
        "display_name": "XGBoost standard",
        "training_id": "xgboost_standard",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "xgboost_balanced_raw",
        "display_name": "XGBoost cost-sensitive",
        "training_id": "xgboost_balanced",
        "metrics_key": "raw_test_metrics",
    },
    {
        "condition": "xgboost_balanced_scaled",
        "display_name": "XGBoost cost-sensitive + scaling",
        "training_id": "xgboost_balanced",
        "metrics_key": "scaled_test_metrics",
    },
]

PAIRED_COMPARISONS = [
    ("cnn_focal_minus_ce", "cnn_focal_raw", "cnn_ce_raw"),
    (
        "cnn_batches_minus_focal",
        "cnn_focal_batch_raw",
        "cnn_focal_raw",
    ),
    (
        "cnn_scaling_minus_raw",
        "cnn_focal_batch_scaled",
        "cnn_focal_batch_raw",
    ),
    (
        "transformer_focal_minus_ce",
        "transformer_focal_raw",
        "transformer_ce_raw",
    ),
    (
        "transformer_batches_minus_focal",
        "transformer_focal_batch_raw",
        "transformer_focal_raw",
    ),
    (
        "transformer_scaling_minus_raw",
        "transformer_focal_batch_scaled",
        "transformer_focal_batch_raw",
    ),
    (
        "xgboost_weights_minus_standard",
        "xgboost_balanced_raw",
        "xgboost_standard_raw",
    ),
    (
        "xgboost_scaling_minus_raw",
        "xgboost_balanced_scaled",
        "xgboost_balanced_raw",
    ),
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
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        if not path.exists():
            digest.update(b"<missing>")
            continue
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()[:16]


def sha256_indices(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def version_major(version: str) -> int | None:
    match = re.match(r"^(\d+)", version)
    return int(match.group(1)) if match else None


def validate_runtime_dependencies(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    missing = [
        name
        for name in ("tensorflow", "xgboost")
        if package_version(name) == "not-installed"
    ]
    if missing:
        parser.error(
            "Missing runtime packages: "
            f"{missing}. Install them before starting the full ablation."
        )
    xgboost_version = package_version("xgboost")
    xgboost_major = version_major(xgboost_version)
    if xgboost_major is None or xgboost_major < 2:
        parser.error(
            "XGBoost >= 2.0 is required for the configured device API; "
            f"found {xgboost_version}."
        )
    try:
        import xgboost as xgb
    except (ImportError, OSError) as error:
        parser.error(f"XGBoost could not load its native library: {error}")
    if args.xgb_device == "cuda":
        build_info = xgb.build_info() if hasattr(xgb, "build_info") else {}
        cuda_build_flag = build_info.get("USE_CUDA")
        if cuda_build_flag in {False, "OFF", "false", "0", 0}:
            parser.error(
                "--xgb-device cuda was requested, but this XGBoost build has "
                "no CUDA support."
            )


def atomic_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as output_file:
        np.savez_compressed(output_file, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_collapsed_nsl_kdd(path: Path, is_train: bool) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=NSL_KDD_COLUMNS)
    frame = frame.drop(columns=["difficulty", "num_outbound_cmds"])
    frame["su_attempted"] = frame["su_attempted"].replace(2, 1)

    if is_train:
        replacements = {
            "DoS": ["neptune", "smurf", "back", "teardrop", "pod", "land"],
            "Probe": ["satan", "ipsweep", "portsweep", "nmap"],
            "R2L": [
                "warezclient",
                "guess_passwd",
                "warezmaster",
                "imap",
                "ftp_write",
                "multihop",
                "phf",
                "spy",
            ],
            "U2R": ["buffer_overflow", "rootkit", "loadmodule", "perl"],
        }
    else:
        replacements = {
            "DoS": [
                "neptune",
                "apache2",
                "processtable",
                "smurf",
                "back",
                "mailbomb",
                "pod",
                "teardrop",
                "land",
                "udpstorm",
            ],
            "Probe": ["mscan", "satan", "saint", "portsweep", "ipsweep", "nmap"],
            "R2L": [
                "guess_passwd",
                "warezmaster",
                "snmpguess",
                "snmpgetattack",
                "httptunnel",
                "multihop",
                "named",
                "sendmail",
                "xlock",
                "xsnoop",
                "ftp_write",
                "worm",
                "phf",
                "imap",
            ],
            "U2R": [
                "buffer_overflow",
                "ps",
                "rootkit",
                "xterm",
                "loadmodule",
                "perl",
                "sqlattack",
            ],
        }
    for collapsed, raw_labels in replacements.items():
        frame["class"] = frame["class"].replace(raw_labels, collapsed)
    unknown = sorted(set(frame["class"].unique()) - set(CLASS_TO_ID))
    if unknown:
        raise ValueError(f"Found unmapped class labels in {path}: {unknown}")
    frame["class"] = frame["class"].map(CLASS_TO_ID).astype(np.int64)
    return frame.reset_index(drop=True)


def split_raw_indices(
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
        stratify=np.asarray(labels, dtype=np.int64),
    )
    return (
        np.sort(np.asarray(train_indices, dtype=np.int64)),
        np.sort(np.asarray(val_indices, dtype=np.int64)),
    )


def build_optimized_feature_order(feature_columns: Sequence[str]) -> List[str]:
    feature_columns = list(feature_columns)
    feature_set = set(feature_columns)
    if len(feature_columns) != 121 or len(feature_set) != 121:
        raise ValueError(
            f"Expected 121 unique processed features, got {len(feature_columns)}."
        )
    numeric = [*NUM_BASIC, *NUM_CONTENT, *NUM_TRAFFIC, *NUM_HOST]
    missing = [column for column in numeric if column not in feature_set]
    if missing:
        raise ValueError(f"Missing expected numeric features: {missing}")
    protocol = sorted(
        column for column in feature_columns if column.startswith("protocol_type_")
    )
    flag = sorted(column for column in feature_columns if column.startswith("flag_"))
    service = sorted(
        column for column in feature_columns if column.startswith("service_")
    )
    used = set(numeric) | set(protocol) | set(flag) | set(service)
    extras = sorted(feature_set - used)
    ordered = [*numeric, *protocol, *flag, *service, *extras]
    if len(ordered) != 121 or set(ordered) != feature_set:
        raise ValueError("Optimized feature ordering lost or duplicated a feature.")
    return ordered


def fit_fold_preprocessor(
    train_fold: pd.DataFrame,
) -> Dict[str, Any]:
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

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

    train_ohe = apply_fold_preprocessor_one_hot(
        train_fold,
        encoders,
        feature_names,
    )
    scaler = MinMaxScaler()
    scaler.fit(train_ohe[COLUMNS_TO_SCALE])
    feature_columns = [column for column in train_ohe.columns if column != "class"]
    ordered_features = build_optimized_feature_order(feature_columns)
    return {
        "encoders": encoders,
        "feature_names": feature_names,
        "scaler": scaler,
        "ordered_features": ordered_features,
    }


def apply_fold_preprocessor_one_hot(
    frame: pd.DataFrame,
    encoders: Dict[str, Any],
    feature_names: Dict[str, List[str]],
) -> pd.DataFrame:
    output = frame.copy()
    for column in CATEGORICAL_COLUMNS:
        encoded = encoders[column].transform(output[[column]])
        encoded_frame = pd.DataFrame(
            encoded,
            columns=feature_names[column],
            index=output.index,
        )
        output = output.drop(columns=[column]).join(encoded_frame)
    return output


def transform_with_fold_preprocessor(
    frame: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    processed = apply_fold_preprocessor_one_hot(
        frame,
        preprocessor["encoders"],
        preprocessor["feature_names"],
    )
    processed[COLUMNS_TO_SCALE] = preprocessor["scaler"].transform(
        processed[COLUMNS_TO_SCALE]
    )
    ordered_columns = [*preprocessor["ordered_features"], "class"]
    processed = processed[ordered_columns]
    X = processed.drop(columns=["class"]).to_numpy(dtype=np.float32)
    y = processed["class"].to_numpy(dtype=np.int64)
    if X.shape[1] != 121:
        raise ValueError(f"Expected a 121-feature matrix, got {X.shape}.")
    return X, y


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
    )

    labels = np.arange(5)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    precisions = precision_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_values = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
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


def apply_class_score_scaling(
    probabilities: np.ndarray,
    coefficients: Dict[int, float],
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=np.float64).copy()
    if scores.ndim != 2 or scores.shape[1] != 5:
        raise ValueError(f"Expected probability shape (n, 5), got {scores.shape}.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Probability array contains a non-finite value.")
    for class_id, coefficient_raw in coefficients.items():
        coefficient = float(coefficient_raw)
        if class_id < 0 or class_id >= 5:
            raise ValueError(f"Score-scaling class ID is out of range: {class_id}")
        if not np.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError("Score coefficients must be finite and positive.")
        scores[:, class_id] /= coefficient
    return np.argmax(scores, axis=1).astype(np.int64)


def score_coefficient_values(
    minimum: float,
    maximum: float,
    step: float,
) -> List[float]:
    values = [
        minimum + index * step
        for index in range(int(np.floor((maximum - minimum) / step + 1e-9)) + 1)
    ]
    if not np.isclose(values[-1], maximum):
        values.append(maximum)
    if minimum <= 1.0 <= maximum:
        values.append(1.0)
    return sorted({round(float(value), 10) for value in values})


def score_scaling_rank_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    raw_predictions: np.ndarray,
) -> Dict[str, float]:
    metrics = calculate_metrics(y_true, predictions)
    changed = int(np.count_nonzero(predictions != raw_predictions))
    return {
        **metrics,
        "minority_recall": float((metrics["r2l_recall"] + metrics["u2r_recall"]) / 2.0),
        "minority_recall_gap": float(
            abs(metrics["r2l_recall"] - metrics["u2r_recall"])
        ),
        "changed_predictions": float(changed),
        "change_rate": float(changed / len(y_true)),
    }


def search_score_coefficients(
    y_val: np.ndarray,
    probabilities: np.ndarray,
    candidates: Sequence[float],
    macro_f1_retention: float,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    probabilities = np.asarray(probabilities)
    raw_predictions = np.argmax(probabilities, axis=1).astype(np.int64)
    raw_metrics = score_scaling_rank_metrics(
        y_val,
        raw_predictions,
        raw_predictions,
    )
    minimum_macro_f1 = raw_metrics["macro_f1"] * float(macro_f1_retention)
    rows: List[Dict[str, float]] = []
    for r2l_coefficient in candidates:
        for u2r_coefficient in candidates:
            predictions = apply_class_score_scaling(
                probabilities,
                {2: float(r2l_coefficient), 3: float(u2r_coefficient)},
            )
            metrics = score_scaling_rank_metrics(
                y_val,
                predictions,
                raw_predictions,
            )
            rows.append(
                {
                    "r2l_score_coefficient": float(r2l_coefficient),
                    "u2r_score_coefficient": float(u2r_coefficient),
                    "distance_from_argmax": float(
                        abs(np.log(r2l_coefficient)) + abs(np.log(u2r_coefficient))
                    ),
                    "minimum_allowed_macro_f1": float(minimum_macro_f1),
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
    if not rows:
        raise ValueError("The score-coefficient search grid is empty.")
    return rows[0], rows


def balanced_class_weights(y: np.ndarray, num_classes: int = 5) -> np.ndarray:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(
            f"Cannot balance classes; missing IDs {np.flatnonzero(counts == 0).tolist()}."
        )
    return len(y) / (float(num_classes) * counts.astype(np.float64))


def effective_number_alpha(
    y: np.ndarray,
    beta: float,
    num_classes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=num_classes).astype(
        np.float64
    )
    effective = 1.0 - np.power(float(beta), counts)
    weights = (1.0 - float(beta)) / np.maximum(effective, 1e-12)
    weights = np.where(counts > 0, weights, 0.0)
    weights = weights / weights.sum() * num_classes
    return weights.astype(np.float32), counts.astype(np.int64)


def configuration_by_id(training_id: str) -> Dict[str, Any]:
    for configuration in TRAINING_CONFIGS:
        if configuration["training_id"] == training_id:
            return dict(configuration)
    raise ValueError(f"Unknown training ID: {training_id}")


def prepare_fold_data(
    repo_root: Path,
    seed: int,
    val_split: float,
) -> Dict[str, Any]:
    raw_train = load_collapsed_nsl_kdd(
        repo_root / "data" / "KDDTrain+.txt",
        is_train=True,
    )
    labels = raw_train["class"].to_numpy(dtype=np.int64)
    train_indices, val_indices = split_raw_indices(labels, seed, val_split)
    train_fold = raw_train.iloc[train_indices].copy().reset_index(drop=True)
    validation_fold = raw_train.iloc[val_indices].copy().reset_index(drop=True)

    preprocessor = fit_fold_preprocessor(train_fold)
    X_train, y_train = transform_with_fold_preprocessor(
        train_fold,
        preprocessor,
    )
    X_val, y_val = transform_with_fold_preprocessor(
        validation_fold,
        preprocessor,
    )
    scaler = preprocessor["scaler"]
    scaler_state = np.concatenate(
        [
            np.asarray(scaler.data_min_, dtype=np.float64),
            np.asarray(scaler.data_max_, dtype=np.float64),
        ]
    )
    feature_order = list(preprocessor["ordered_features"])
    return {
        "preprocessor": preprocessor,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_indices_sha256": sha256_indices(train_indices),
        "val_indices_sha256": sha256_indices(val_indices),
        "train_counts": np.bincount(y_train, minlength=5),
        "validation_counts": np.bincount(y_val, minlength=5),
        "feature_order": feature_order,
        "feature_order_sha256": hashlib.sha256(
            "\n".join(feature_order).encode("utf-8")
        ).hexdigest(),
        "scaler_state_sha256": sha256_array(scaler_state),
    }


def train_neural_model(
    configuration: Dict[str, Any],
    fold: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, Dict[str, Any]]:
    import tensorflow as tf

    from cnn_gan_foc import ClassBalancedFocalLoss  # type: ignore
    from cnn_opt import (  # type: ignore
        BalancedBatchSequence,
        ValF1Callback,
        build_opt_cnn,
    )
    from cnn_opt_1d_4gpu import build_vanilla_transformer  # type: ignore

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

    seed = int(args.worker_seed)
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)
    y_train = np.asarray(fold["y_train"], dtype=np.int64)
    y_val = np.asarray(fold["y_val"], dtype=np.int64)

    if configuration["family"] == "cnn":
        X_train = np.asarray(fold["X_train"], dtype=np.float32).reshape(-1, 11, 11, 1)
        X_val = np.asarray(fold["X_val"], dtype=np.float32).reshape(-1, 11, 11, 1)
    else:
        X_train = np.asarray(fold["X_train"], dtype=np.float32).reshape(-1, 121, 1)
        X_val = np.asarray(fold["X_val"], dtype=np.float32).reshape(-1, 121, 1)

    if configuration["loss_mode"] == "class_balanced_focal":
        alpha, alpha_counts = effective_number_alpha(
            y_train,
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

    if configuration["family"] == "cnn":
        model = build_opt_cnn(
            loss=loss,
            groups=args.cnn_groups,
            base_filters=args.cnn_base_filters,
            dense_units=args.cnn_dense_units,
            dropout1=args.cnn_dropout1,
            dropout2=args.cnn_dropout2,
            use_batch_norm=True,
            use_residual=True,
        )
        architecture = {
            "groups": int(args.cnn_groups),
            "base_filters": int(args.cnn_base_filters),
            "dense_units": int(args.cnn_dense_units),
            "dropout1": float(args.cnn_dropout1),
            "dropout2": float(args.cnn_dropout2),
            "use_batch_norm": True,
            "use_residual": True,
        }
    else:
        model = build_vanilla_transformer(
            loss=loss,
            d_model=args.transformer_d_model,
            num_heads=args.transformer_num_heads,
            num_blocks=args.transformer_blocks,
            ff_dim=args.transformer_ff_dim,
            dense_units=args.transformer_dense_units,
            transformer_dropout=args.transformer_dropout,
            head_dropout=args.transformer_head_dropout,
        )
        architecture = {
            "d_model": int(args.transformer_d_model),
            "num_heads": int(args.transformer_num_heads),
            "blocks": int(args.transformer_blocks),
            "ff_dim": int(args.transformer_ff_dim),
            "dense_units": int(args.transformer_dense_units),
            "transformer_dropout": float(args.transformer_dropout),
            "head_dropout": float(args.transformer_head_dropout),
        }
    model_parameters = int(model.count_params())
    if (
        configuration["family"] == "transformer"
        and model_parameters != args.expected_transformer_parameters
    ):
        raise ValueError(
            "Transformer parameter count changed: "
            f"expected {args.expected_transformer_parameters}, got "
            f"{model_parameters}."
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
        if configuration["minority_batches"]:
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
                callbacks=callbacks,
                verbose=1,
            )
        else:
            history = model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks,
                verbose=1,
            )
        if not weights_path.exists():
            raise RuntimeError("Best validation checkpoint was not created.")
        model.load_weights(weights_path)

    validation_probabilities = model.predict(
        X_val,
        batch_size=args.batch_size,
        verbose=0,
    )
    val_history = history.history.get("val_macro_f1", [])
    training_metadata = {
        "optimizer": "keras_adam_defaults",
        "epochs_requested": int(args.epochs),
        "epochs_completed": len(history.history.get("loss", [])),
        "best_epoch": (int(np.argmax(val_history)) + 1 if val_history else None),
        "checkpoint_metric": "validation_macro_f1_raw_argmax",
        "early_stopping_patience": int(args.early_stopping_patience),
        "batch_size": int(args.batch_size),
        "minority_per_batch": (
            int(args.minority_per_batch) if configuration["minority_batches"] else 0
        ),
        "alpha": np.asarray(alpha, dtype=float).tolist(),
        "alpha_counts": np.asarray(alpha_counts, dtype=int).tolist(),
        "cb_beta": (
            float(args.cb_beta)
            if configuration["loss_mode"] == "class_balanced_focal"
            else None
        ),
        "focal_gamma": (
            float(args.focal_gamma)
            if configuration["loss_mode"] == "class_balanced_focal"
            else None
        ),
        "model_parameters": model_parameters,
        "architecture": architecture,
        "tensorflow_visible_gpu_count": len(visible_gpus),
        "tensorflow_version": package_version("tensorflow"),
        "keras_version": package_version("keras"),
    }
    return model, validation_probabilities, training_metadata


def train_xgboost_model(
    configuration: Dict[str, Any],
    fold: Dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray, Dict[str, Any]]:
    try:
        import xgboost as xgb
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "XGBoost is required; install the repository's XGBoost version."
        ) from error

    assigned_devices = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    visible_gpu_count: int | None = len(assigned_devices)
    build_info = xgb.build_info() if hasattr(xgb, "build_info") else {}
    if args.xgb_device == "cuda":
        if not args.allow_cpu and visible_gpu_count != 1:
            raise RuntimeError(
                "XGBoost worker expected exactly one assigned CUDA GPU, "
                "but CUDA_VISIBLE_DEVICES assigns "
                f"{visible_gpu_count}: {assigned_devices}."
            )
        cuda_build_flag = build_info.get("USE_CUDA")
        if cuda_build_flag in {False, "OFF", "false", "0", 0}:
            raise RuntimeError(
                "This XGBoost installation was built without CUDA support."
            )

    seed = int(args.worker_seed)
    np.random.seed(seed)
    X_train = np.asarray(fold["X_train"], dtype=np.float32)
    y_train = np.asarray(fold["y_train"], dtype=np.int64)
    X_val = np.asarray(fold["X_val"], dtype=np.float32)
    y_val = np.asarray(fold["y_val"], dtype=np.int64)

    if configuration["class_weighting"] == "balanced":
        class_weights = balanced_class_weights(y_train)
        sample_weight = class_weights[y_train]
    else:
        class_weights = None
        sample_weight = None

    model_arguments: Dict[str, Any] = {
        "objective": "multi:softprob",
        "num_class": 5,
        "n_estimators": int(args.xgb_n_estimators),
        "max_depth": int(args.xgb_max_depth),
        "learning_rate": float(args.xgb_learning_rate),
        "subsample": float(args.xgb_subsample),
        "colsample_bytree": float(args.xgb_colsample_bytree),
        "min_child_weight": float(args.xgb_min_child_weight),
        "reg_lambda": float(args.xgb_reg_lambda),
        "gamma": float(args.xgb_gamma),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": args.xgb_device,
        "fail_on_invalid_gpu_id": args.xgb_device == "cuda",
        "random_state": seed,
        "n_jobs": int(args.xgb_n_jobs),
    }
    if args.xgb_early_stopping_rounds > 0:
        model_arguments["early_stopping_rounds"] = int(args.xgb_early_stopping_rounds)
    model = XGBClassifier(**model_arguments)
    fit_arguments: Dict[str, Any] = {
        "eval_set": [(X_val, y_val)],
        "verbose": (int(args.xgb_verbose_eval) if args.xgb_verbose_eval > 0 else False),
    }
    if sample_weight is not None:
        fit_arguments["sample_weight"] = sample_weight
    model.fit(X_train, y_train, **fit_arguments)
    validation_probabilities = model.predict_proba(X_val)
    try:
        best_iteration = int(model.best_iteration)
    except (AttributeError, TypeError, ValueError):
        best_iteration = None
    training_metadata = {
        "optimizer": "xgboost_hist",
        "class_weights": (
            class_weights.astype(float).tolist() if class_weights is not None else None
        ),
        "validation_weighting": "none",
        "best_iteration": best_iteration,
        "xgboost_version": xgb.__version__,
        "xgboost_build_info": json.loads(json.dumps(build_info, default=str)),
        "parameters": model_arguments,
        "assigned_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "cuda_visible_gpu_count": visible_gpu_count,
    }
    return model, validation_probabilities, training_metadata


def predict_probabilities(
    model: Any,
    family: str,
    X: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if family == "cnn":
        return np.asarray(
            model.predict(
                X.reshape(-1, 11, 11, 1),
                batch_size=batch_size,
                verbose=0,
            )
        )
    if family == "transformer":
        return np.asarray(
            model.predict(
                X.reshape(-1, 121, 1),
                batch_size=batch_size,
                verbose=0,
            )
        )
    return np.asarray(model.predict_proba(X))


def run_training_worker(args: argparse.Namespace, repo_root: Path) -> None:
    configuration = configuration_by_id(args.worker_training_id)
    seed = int(args.worker_seed)
    start = time.perf_counter()
    fold = prepare_fold_data(repo_root, seed, args.val_split)

    if configuration["family"] in {"cnn", "transformer"}:
        model, validation_probabilities, training_metadata = train_neural_model(
            configuration,
            fold,
            args,
        )
    else:
        model, validation_probabilities, training_metadata = train_xgboost_model(
            configuration,
            fold,
            args,
        )

    raw_validation_predictions = np.argmax(
        validation_probabilities,
        axis=1,
    ).astype(np.int64)
    raw_validation_metrics = calculate_metrics(
        fold["y_val"],
        raw_validation_predictions,
    )
    score_search_path: Path | None = None
    selected_validation_metrics: Dict[str, float] | None = None
    coefficient_pair_count = 0
    if configuration["supports_scaled_row"]:
        candidates = score_coefficient_values(
            args.coefficient_min,
            args.coefficient_max,
            args.coefficient_step,
        )
        selected, search_rows = search_score_coefficients(
            fold["y_val"],
            validation_probabilities,
            candidates,
            args.min_validation_macro_f1_retention,
        )
        coefficients = {
            2: float(selected["r2l_score_coefficient"]),
            3: float(selected["u2r_score_coefficient"]),
        }
        selected_validation_predictions = apply_class_score_scaling(
            validation_probabilities,
            coefficients,
        )
        selected_validation_metrics = calculate_metrics(
            fold["y_val"],
            selected_validation_predictions,
        )
        coefficient_pair_count = len(search_rows)
        result_path = Path(args.worker_result_path)
        score_search_path = result_path.with_name(
            f"{args.worker_run_name}_{args.worker_attempt_id}_score_search.csv"
        )
        search_frame = pd.DataFrame(search_rows)
        search_frame.insert(0, "rank", np.arange(1, len(search_frame) + 1))
        atomic_csv(score_search_path, search_frame)
        score_selection = "validation_grid_search"
    else:
        coefficients = {2: 1.0, 3: 1.0}
        score_selection = "disabled"

    training_and_selection_seconds = time.perf_counter() - start

    # KDDTest+ is deliberately loaded only after training and validation-only
    # selection are both complete.
    test_load_started = time.time()
    raw_test = load_collapsed_nsl_kdd(
        repo_root / "data" / "KDDTest+.txt",
        is_train=False,
    )
    X_test, y_test = transform_with_fold_preprocessor(
        raw_test,
        fold["preprocessor"],
    )
    test_probabilities = predict_probabilities(
        model,
        configuration["family"],
        X_test,
        args.batch_size,
    )
    raw_test_predictions = np.argmax(test_probabilities, axis=1).astype(np.int64)
    raw_test_metrics = calculate_metrics(y_test, raw_test_predictions)
    if configuration["supports_scaled_row"]:
        scaled_test_predictions = apply_class_score_scaling(
            test_probabilities,
            coefficients,
        )
        scaled_test_metrics = calculate_metrics(y_test, scaled_test_predictions)
    else:
        selected_validation_predictions = raw_validation_predictions
        scaled_test_predictions = raw_test_predictions
        scaled_test_metrics = None

    result_path = Path(args.worker_result_path)
    prediction_path = result_path.with_name(
        f"{args.worker_run_name}_{args.worker_attempt_id}_predictions.npz"
    )
    atomic_npz(
        prediction_path,
        train_indices=np.asarray(fold["train_indices"], dtype=np.int64),
        validation_indices=np.asarray(fold["val_indices"], dtype=np.int64),
        validation_labels=np.asarray(fold["y_val"], dtype=np.int64),
        validation_probabilities=np.asarray(
            validation_probabilities,
            dtype=np.float32,
        ),
        raw_validation_predictions=np.asarray(
            raw_validation_predictions,
            dtype=np.int64,
        ),
        selected_validation_predictions=np.asarray(
            selected_validation_predictions,
            dtype=np.int64,
        ),
        test_labels=np.asarray(y_test, dtype=np.int64),
        test_probabilities=np.asarray(test_probabilities, dtype=np.float32),
        raw_test_predictions=np.asarray(raw_test_predictions, dtype=np.int64),
        selected_test_predictions=np.asarray(
            scaled_test_predictions,
            dtype=np.int64,
        ),
        score_coefficients=np.asarray(
            [coefficients[2], coefficients[3]],
            dtype=np.float64,
        ),
    )
    prediction_sha256 = sha256_file(prediction_path)

    result: Dict[str, Any] = {
        "schema_version": 1,
        "experiment_key": args.experiment_key,
        "config_key": args.config_key,
        "attempt_id": args.worker_attempt_id,
        "run_name": args.worker_run_name,
        "training_id": configuration["training_id"],
        "family": configuration["family"],
        "seed": seed,
        "no_ctgan": True,
        "synthetic_rows": 0,
        "train_data": "KDDTrain+ real rows only",
        "validation_data": "held-out real KDDTrain+ fold",
        "test_data": "KDDTest+",
        "split_before_preprocessing": True,
        "val_split": float(args.val_split),
        "train_indices_sha256": fold["train_indices_sha256"],
        "validation_indices_sha256": fold["val_indices_sha256"],
        "train_counts": fold["train_counts"].astype(int).tolist(),
        "validation_counts": fold["validation_counts"].astype(int).tolist(),
        "test_counts": np.bincount(y_test, minlength=5).astype(int).tolist(),
        "feature_count": 121,
        "feature_layout": "optimized_fixed_canonical_121",
        "feature_order_sha256": fold["feature_order_sha256"],
        "scaler_fit_data": "80_percent_real_training_fold_only",
        "scaler_state_sha256": fold["scaler_state_sha256"],
        "categorical_vocabulary": CANONICAL_CATEGORIES,
        "loss_mode": configuration["loss_mode"],
        "class_weighting": configuration["class_weighting"],
        "minority_batches": bool(configuration["minority_batches"]),
        "score_scaling_supported": bool(configuration["supports_scaled_row"]),
        "score_scaling_selection": score_selection,
        "score_scaling_selection_data": (
            "real_validation_fold_only"
            if configuration["supports_scaled_row"]
            else "not_applicable"
        ),
        "score_scaling_objective": (
            "retain validation macro-F1 then optimize minority recall/F1"
            if configuration["supports_scaled_row"]
            else "not_applicable"
        ),
        "coefficient_min": float(args.coefficient_min),
        "coefficient_max": float(args.coefficient_max),
        "coefficient_step": float(args.coefficient_step),
        "min_validation_macro_f1_retention": float(
            args.min_validation_macro_f1_retention
        ),
        "coefficient_pairs_searched": coefficient_pair_count,
        "r2l_score_coefficient": float(coefficients[2]),
        "u2r_score_coefficient": float(coefficients[3]),
        "score_search_path": (
            str(score_search_path) if score_search_path is not None else None
        ),
        "score_search_sha256": (
            sha256_file(score_search_path) if score_search_path is not None else None
        ),
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "raw_validation_metrics": raw_validation_metrics,
        "selected_validation_metrics": selected_validation_metrics,
        "raw_test_metrics": raw_test_metrics,
        "scaled_test_metrics": scaled_test_metrics,
        "test_loaded_after_training_and_policy_selection": True,
        "test_load_started_unix": test_load_started,
        "training_and_selection_seconds": training_and_selection_seconds,
        "runtime_seconds": time.perf_counter() - start,
        "assigned_gpu": os.environ.get("EXPERIMENT_GPU_ID", ""),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": package_version("scikit-learn"),
        "training_metadata": training_metadata,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    atomic_json(result_path, result)
    print(
        f"Completed {args.worker_run_name}: "
        f"raw macro-F1={raw_test_metrics['macro_f1']:.6f}",
        flush=True,
    )


def metrics_are_complete(values: Any) -> bool:
    if not isinstance(values, dict):
        return False
    try:
        numbers = np.asarray([float(values[name]) for name in METRICS])
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(numbers).all())


def metrics_match_predictions(
    expected: Any,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> bool:
    if not metrics_are_complete(expected):
        return False
    calculated = calculate_metrics(labels, predictions)
    return all(
        np.isclose(
            float(expected[metric]),
            float(calculated[metric]),
            rtol=1e-12,
            atol=1e-12,
        )
        for metric in METRICS
    )


def prediction_artifact_is_complete(path: Path, data: Dict[str, Any]) -> bool:
    if not path.is_file() or sha256_file(path) != data.get("prediction_sha256"):
        return False
    required_keys = {
        "train_indices",
        "validation_indices",
        "validation_labels",
        "validation_probabilities",
        "raw_validation_predictions",
        "selected_validation_predictions",
        "test_labels",
        "test_probabilities",
        "raw_test_predictions",
        "selected_test_predictions",
        "score_coefficients",
    }
    try:
        with np.load(path, allow_pickle=False) as artifact:
            if not required_keys.issubset(artifact.files):
                return False
            arrays = {key: np.asarray(artifact[key]) for key in required_keys}
    except (OSError, ValueError, KeyError):
        return False

    train_count = int(sum(data["train_counts"]))
    validation_count = int(sum(data["validation_counts"]))
    test_count = int(sum(data["test_counts"]))
    expected_shapes = {
        "train_indices": (train_count,),
        "validation_indices": (validation_count,),
        "validation_labels": (validation_count,),
        "validation_probabilities": (validation_count, 5),
        "raw_validation_predictions": (validation_count,),
        "selected_validation_predictions": (validation_count,),
        "test_labels": (test_count,),
        "test_probabilities": (test_count, 5),
        "raw_test_predictions": (test_count,),
        "selected_test_predictions": (test_count,),
        "score_coefficients": (2,),
    }
    if any(arrays[key].shape != shape for key, shape in expected_shapes.items()):
        return False
    if sha256_indices(arrays["train_indices"]) != data["train_indices_sha256"]:
        return False
    if (
        sha256_indices(arrays["validation_indices"])
        != data["validation_indices_sha256"]
    ):
        return False
    if (
        np.bincount(arrays["validation_labels"], minlength=5).astype(int).tolist()
        != data["validation_counts"]
        or np.bincount(arrays["test_labels"], minlength=5).astype(int).tolist()
        != data["test_counts"]
    ):
        return False
    finite_keys = {
        "validation_probabilities",
        "test_probabilities",
        "score_coefficients",
    }
    if any(not np.isfinite(arrays[key]).all() for key in finite_keys):
        return False
    categorical_keys = {
        "validation_labels",
        "raw_validation_predictions",
        "selected_validation_predictions",
        "test_labels",
        "raw_test_predictions",
        "selected_test_predictions",
    }
    if any(np.any((arrays[key] < 0) | (arrays[key] >= 5)) for key in categorical_keys):
        return False
    expected_coefficients = np.asarray(
        [data["r2l_score_coefficient"], data["u2r_score_coefficient"]],
        dtype=np.float64,
    )
    if not np.array_equal(arrays["score_coefficients"], expected_coefficients):
        return False
    if not metrics_match_predictions(
        data["raw_validation_metrics"],
        arrays["validation_labels"],
        arrays["raw_validation_predictions"],
    ) or not metrics_match_predictions(
        data["raw_test_metrics"],
        arrays["test_labels"],
        arrays["raw_test_predictions"],
    ):
        return False
    if bool(data["score_scaling_supported"]):
        return metrics_match_predictions(
            data["selected_validation_metrics"],
            arrays["validation_labels"],
            arrays["selected_validation_predictions"],
        ) and metrics_match_predictions(
            data["scaled_test_metrics"],
            arrays["test_labels"],
            arrays["selected_test_predictions"],
        )
    return bool(
        np.array_equal(
            arrays["selected_validation_predictions"],
            arrays["raw_validation_predictions"],
        )
        and np.array_equal(
            arrays["selected_test_predictions"],
            arrays["raw_test_predictions"],
        )
    )


def result_is_complete(
    path: Path,
    plan: Dict[str, Any],
    expected_attempt_id: str | None = None,
) -> bool:
    if not path.exists():
        return False
    try:
        data = read_json(path)
        configuration = configuration_by_id(plan["training_id"])
        checks = {
            "schema_version": 1,
            "experiment_key": plan["experiment_key"],
            "config_key": plan["config_key"],
            "run_name": plan["run_name"],
            "training_id": plan["training_id"],
            "family": plan["family"],
            "seed": int(plan["seed"]),
            "no_ctgan": True,
            "synthetic_rows": 0,
            "train_indices_sha256": plan["train_indices_sha256"],
            "validation_indices_sha256": plan["validation_indices_sha256"],
        }
        for key, expected in checks.items():
            if data.get(key) != expected:
                return False
        if (
            expected_attempt_id is not None
            and data.get("attempt_id") != expected_attempt_id
        ):
            return False
        if not metrics_are_complete(data.get("raw_test_metrics")):
            return False
        if not metrics_are_complete(data.get("raw_validation_metrics")):
            return False
        if configuration["supports_scaled_row"]:
            if not metrics_are_complete(data.get("scaled_test_metrics")):
                return False
            if not metrics_are_complete(data.get("selected_validation_metrics")):
                return False
            for key in ("r2l_score_coefficient", "u2r_score_coefficient"):
                coefficient = float(data.get(key, 0.0))
                if not np.isfinite(coefficient) or coefficient <= 0.0:
                    return False
            score_search_path = Path(str(data.get("score_search_path", "")))
            if not score_search_path.is_file() or sha256_file(
                score_search_path
            ) != data.get("score_search_sha256"):
                return False
        elif data.get("scaled_test_metrics") is not None:
            return False
        prediction_path = Path(str(data.get("prediction_path", "")))
        if not prediction_artifact_is_complete(prediction_path, data):
            return False
        return True
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
        "--worker-seed",
        str(plan["seed"]),
        "--worker-run-name",
        str(plan["run_name"]),
        "--worker-result-path",
        str(plan["result_path"]),
        "--worker-attempt-id",
        attempt_id,
        "--experiment-key",
        str(plan["experiment_key"]),
        "--config-key",
        str(plan["config_key"]),
        "--val-split",
        str(args.val_split),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--cb-beta",
        str(args.cb_beta),
        "--focal-gamma",
        str(args.focal_gamma),
        "--minority-per-batch",
        str(args.minority_per_batch),
        "--coefficient-min",
        str(args.coefficient_min),
        "--coefficient-max",
        str(args.coefficient_max),
        "--coefficient-step",
        str(args.coefficient_step),
        "--min-validation-macro-f1-retention",
        str(args.min_validation_macro_f1_retention),
        "--cnn-groups",
        str(args.cnn_groups),
        "--cnn-base-filters",
        str(args.cnn_base_filters),
        "--cnn-dense-units",
        str(args.cnn_dense_units),
        "--cnn-dropout1",
        str(args.cnn_dropout1),
        "--cnn-dropout2",
        str(args.cnn_dropout2),
        "--transformer-d-model",
        str(args.transformer_d_model),
        "--transformer-num-heads",
        str(args.transformer_num_heads),
        "--transformer-blocks",
        str(args.transformer_blocks),
        "--transformer-ff-dim",
        str(args.transformer_ff_dim),
        "--transformer-dense-units",
        str(args.transformer_dense_units),
        "--transformer-dropout",
        str(args.transformer_dropout),
        "--transformer-head-dropout",
        str(args.transformer_head_dropout),
        "--expected-transformer-parameters",
        str(args.expected_transformer_parameters),
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
    return command


def build_all_run_rows(
    plans: Sequence[Dict[str, Any]],
    seeds: Sequence[int],
) -> pd.DataFrame:
    results_by_training_seed: Dict[tuple[str, int], tuple[Dict[str, Any], Path]] = {}
    for plan in plans:
        result_path = Path(plan["result_path"])
        if not result_is_complete(result_path, plan):
            raise ValueError(f"Incomplete result during aggregation: {result_path}")
        results_by_training_seed[(plan["training_id"], int(plan["seed"]))] = (
            read_json(result_path),
            result_path,
        )

    rows: List[Dict[str, Any]] = []
    for report_index, report in enumerate(REPORT_ROWS):
        for seed in seeds:
            result, result_path = results_by_training_seed[
                (report["training_id"], int(seed))
            ]
            metrics = result[report["metrics_key"]]
            scaled = report["metrics_key"] == "scaled_test_metrics"
            row: Dict[str, Any] = {
                "condition_order": report_index,
                "condition": report["condition"],
                "display_name": report["display_name"],
                "seed": int(seed),
                "source_training_id": report["training_id"],
                "source_run_name": result["run_name"],
                "decision_policy": (
                    "validation_selected_score_scaling" if scaled else "raw_argmax"
                ),
                "r2l_score_coefficient": (
                    result["r2l_score_coefficient"] if scaled else 1.0
                ),
                "u2r_score_coefficient": (
                    result["u2r_score_coefficient"] if scaled else 1.0
                ),
                "no_ctgan": True,
                "result_path": str(result_path),
                **{metric: float(metrics[metric]) for metric in METRICS},
            }
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["condition_order", "seed"])
        .reset_index(drop=True)
    )


def build_summary(raw: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    expected_seeds = {int(seed) for seed in seeds}
    rows: List[Dict[str, Any]] = []
    for order, report in enumerate(REPORT_ROWS):
        group = raw[raw["condition"] == report["condition"]]
        observed_seeds = set(group["seed"].astype(int).tolist())
        if observed_seeds != expected_seeds or len(group) != len(expected_seeds):
            raise ValueError(
                f"{report['condition']} does not have one result per seed."
            )
        row: Dict[str, Any] = {
            "condition_order": order,
            "condition": report["condition"],
            "display_name": report["display_name"],
            "runs": len(group),
            "seeds": ",".join(str(seed) for seed in sorted(observed_seeds)),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def build_paired_deltas(
    raw: pd.DataFrame,
    seeds: Sequence[int],
) -> pd.DataFrame:
    indexed = raw.set_index(["condition", "seed"])
    rows: List[Dict[str, Any]] = []
    for comparison, enhanced, reference in PAIRED_COMPARISONS:
        for seed in seeds:
            enhanced_row = indexed.loc[(enhanced, int(seed))]
            reference_row = indexed.loc[(reference, int(seed))]
            rows.append(
                {
                    "comparison": comparison,
                    "enhanced_condition": enhanced,
                    "reference_condition": reference,
                    "seed": int(seed),
                    **{
                        f"delta_{metric}": float(enhanced_row[metric])
                        - float(reference_row[metric])
                        for metric in METRICS
                    },
                }
            )
    return pd.DataFrame(rows)


def format_percent(mean: float, std: float) -> str:
    return f"{100.0 * mean:.2f}% +/- {100.0 * std:.2f}%"


def formatted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    labels = [
        ("accuracy", "Accuracy"),
        ("mcc", "MCC"),
        ("macro_f1", "Macro-F1"),
        ("macro_recall", "Macro Recall"),
        ("r2l_precision", "R2L Precision"),
        ("r2l_recall", "R2L Recall"),
        ("r2l_f1", "R2L F1"),
        ("u2r_precision", "U2R Precision"),
        ("u2r_recall", "U2R Recall"),
        ("u2r_f1", "U2R F1"),
    ]
    rows: List[Dict[str, Any]] = []
    for source in summary.to_dict(orient="records"):
        row: Dict[str, Any] = {
            "Condition": source["display_name"],
            "Runs": int(source["runs"]),
        }
        for metric, label in labels:
            row[label] = format_percent(
                float(source[f"{metric}_mean"]),
                float(source[f"{metric}_std"]),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--name-prefix", default="no_ctgan_model_ablation")
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--cb-beta", type=float, default=0.99)
    parser.add_argument("--focal-gamma", type=float, default=0.5)
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument("--coefficient-min", type=float, default=0.05)
    parser.add_argument("--coefficient-max", type=float, default=2.00)
    parser.add_argument("--coefficient-step", type=float, default=0.15)
    parser.add_argument(
        "--min-validation-macro-f1-retention",
        type=float,
        default=0.90,
    )

    parser.add_argument("--cnn-groups", type=int, default=1)
    parser.add_argument("--cnn-base-filters", type=int, default=64)
    parser.add_argument("--cnn-dense-units", type=int, default=256)
    parser.add_argument("--cnn-dropout1", type=float, default=0.25)
    parser.add_argument("--cnn-dropout2", type=float, default=0.30)

    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-num-heads", type=int, default=4)
    parser.add_argument("--transformer-blocks", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=128)
    parser.add_argument("--transformer-dense-units", type=int, default=512)
    parser.add_argument("--transformer-dropout", type=float, default=0.10)
    parser.add_argument("--transformer-head-dropout", type=float, default=0.30)
    parser.add_argument(
        "--expected-transformer-parameters",
        type=int,
        default=110_661,
    )

    parser.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--xgb-n-estimators", type=int, default=1000)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.80)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.80)
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    parser.add_argument("--xgb-gamma", type=float, default=0.0)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--xgb-verbose-eval", type=int, default=50)

    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-all-commands", action="store_true")

    parser.add_argument(
        "--worker-mode",
        choices=["none", "train"],
        default="none",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-training-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-run-name", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result-path", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-attempt-id", default="", help=argparse.SUPPRESS)
    parser.add_argument("--experiment-key", default="", help=argparse.SUPPRESS)
    parser.add_argument("--config-key", default="", help=argparse.SUPPRESS)


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("--seeds cannot contain negative values.")
    if not 0.0 < args.val_split < 1.0:
        parser.error("--val-split must be between zero and one.")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be greater than zero.")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative.")
    if not 0.0 < args.cb_beta < 1.0:
        parser.error("--cb-beta must be between zero and one.")
    if not np.isfinite(args.focal_gamma) or args.focal_gamma < 0.0:
        parser.error("--focal-gamma must be finite and nonnegative.")
    if args.minority_per_batch <= 0:
        parser.error("--minority-per-batch must be greater than zero.")
    if (
        not np.isfinite(args.coefficient_min)
        or not np.isfinite(args.coefficient_max)
        or not np.isfinite(args.coefficient_step)
        or args.coefficient_min <= 0.0
        or args.coefficient_max < args.coefficient_min
        or args.coefficient_step <= 0.0
    ):
        parser.error("Score grid needs 0 < min <= max and a positive step.")
    if not args.coefficient_min <= 1.0 <= args.coefficient_max:
        parser.error("The score grid must contain the raw-argmax point 1.0.")
    if not 0.0 <= args.min_validation_macro_f1_retention <= 1.0:
        parser.error("Macro-F1 retention must be between zero and one.")
    candidate_count = len(
        score_coefficient_values(
            args.coefficient_min,
            args.coefficient_max,
            args.coefficient_step,
        )
    )
    if candidate_count**2 > 100_000:
        parser.error("The score-coefficient grid exceeds 100,000 pairs.")
    if args.cnn_groups <= 0 or args.cnn_base_filters % args.cnn_groups != 0:
        parser.error("--cnn-groups must divide --cnn-base-filters.")
    if args.cnn_dense_units <= 0:
        parser.error("--cnn-dense-units must be greater than zero.")
    for name, value in (
        ("--cnn-dropout1", args.cnn_dropout1),
        ("--cnn-dropout2", args.cnn_dropout2),
        ("--transformer-dropout", args.transformer_dropout),
        ("--transformer-head-dropout", args.transformer_head_dropout),
    ):
        if not 0.0 <= value < 1.0:
            parser.error(f"{name} must be in [0, 1).")
    if args.transformer_d_model <= 0:
        parser.error("--transformer-d-model must be greater than zero.")
    if (
        args.transformer_num_heads <= 0
        or args.transformer_d_model % args.transformer_num_heads != 0
    ):
        parser.error("Transformer heads must divide d_model.")
    if (
        args.transformer_blocks <= 0
        or args.transformer_ff_dim <= 0
        or args.transformer_dense_units <= 0
        or args.expected_transformer_parameters <= 0
    ):
        parser.error(
            "Transformer widths, blocks, and parameter count must be positive."
        )
    if args.xgb_n_estimators <= 0 or args.xgb_early_stopping_rounds < 0:
        parser.error("XGBoost estimator count must be positive; patience nonnegative.")
    if args.xgb_max_depth <= 0 or args.xgb_learning_rate <= 0.0:
        parser.error("XGBoost depth and learning rate must be positive.")
    if not 0.0 < args.xgb_subsample <= 1.0:
        parser.error("--xgb-subsample must be in (0, 1].")
    if not 0.0 < args.xgb_colsample_bytree <= 1.0:
        parser.error("--xgb-colsample-bytree must be in (0, 1].")
    if args.xgb_min_child_weight < 0.0 or args.xgb_reg_lambda < 0.0:
        parser.error("XGBoost child weight and lambda cannot be negative.")
    if args.xgb_gamma < 0.0 or args.xgb_verbose_eval < 0:
        parser.error("XGBoost gamma and verbose interval cannot be negative.")


def experiment_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "library_versions": {
            "tensorflow": package_version("tensorflow"),
            "keras": package_version("keras"),
            "xgboost": package_version("xgboost"),
            "scikit_learn": package_version("scikit-learn"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "seeds": [int(seed) for seed in args.seeds],
        "val_split": float(args.val_split),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "early_stopping_patience": int(args.early_stopping_patience),
        "cb_beta": float(args.cb_beta),
        "focal_gamma": float(args.focal_gamma),
        "minority_per_batch": int(args.minority_per_batch),
        "score_grid": {
            "minimum": float(args.coefficient_min),
            "maximum": float(args.coefficient_max),
            "step": float(args.coefficient_step),
            "macro_f1_retention": float(args.min_validation_macro_f1_retention),
        },
        "cnn": {
            "groups": int(args.cnn_groups),
            "base_filters": int(args.cnn_base_filters),
            "dense_units": int(args.cnn_dense_units),
            "dropout1": float(args.cnn_dropout1),
            "dropout2": float(args.cnn_dropout2),
            "batch_norm": True,
            "residual": True,
        },
        "transformer": {
            "d_model": int(args.transformer_d_model),
            "num_heads": int(args.transformer_num_heads),
            "blocks": int(args.transformer_blocks),
            "ff_dim": int(args.transformer_ff_dim),
            "dense_units": int(args.transformer_dense_units),
            "dropout": float(args.transformer_dropout),
            "head_dropout": float(args.transformer_head_dropout),
            "expected_parameters": int(args.expected_transformer_parameters),
        },
        "xgboost": {
            "device": args.xgb_device,
            "n_estimators": int(args.xgb_n_estimators),
            "early_stopping_rounds": int(args.xgb_early_stopping_rounds),
            "max_depth": int(args.xgb_max_depth),
            "learning_rate": float(args.xgb_learning_rate),
            "subsample": float(args.xgb_subsample),
            "colsample_bytree": float(args.xgb_colsample_bytree),
            "min_child_weight": float(args.xgb_min_child_weight),
            "reg_lambda": float(args.xgb_reg_lambda),
            "gamma": float(args.xgb_gamma),
            "n_jobs": int(args.xgb_n_jobs),
        },
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = Path(__file__).resolve()
    parser = argparse.ArgumentParser(
        description=("No-CTGAN paired-seed ablation for CNN, Transformer, and XGBoost.")
    )
    add_arguments(parser)
    args = parser.parse_args()
    validate_arguments(parser, args)

    if args.worker_mode == "train":
        required = {
            "--worker-training-id": args.worker_training_id,
            "--worker-run-name": args.worker_run_name,
            "--worker-result-path": args.worker_result_path,
            "--worker-attempt-id": args.worker_attempt_id,
            "--experiment-key": args.experiment_key,
            "--config-key": args.config_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(f"Worker invocation is missing: {missing}")
        configuration_by_id(args.worker_training_id)
        run_training_worker(args, repo_root)
        return

    try:
        gpus = parse_gpus(args.gpus)
    except ValueError as error:
        parser.error(str(error))
    prefix = args.name_prefix.strip() or "no_ctgan_model_ablation"
    if Path(prefix).name != prefix:
        parser.error("--name-prefix must be a filename-safe name, not a path.")

    train_path = repo_root / "data" / "KDDTrain+.txt"
    test_path = repo_root / "data" / "KDDTest+.txt"
    required_paths = [
        train_path,
        test_path,
        repo_root / "src" / "cnn_opt.py",
        repo_root / "src" / "cnn_opt_1d_4gpu.py",
        repo_root / "src" / "cnn_gan_foc.py",
        script_path,
    ]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise SystemExit(f"Required files are missing: {missing_paths}")

    source_fingerprint = fingerprint_files(required_paths)
    settings = experiment_settings(args)
    identity = {
        "settings": settings,
        "source_and_data_fingerprint": source_fingerprint,
        "training_configurations": TRAINING_CONFIGS,
        "report_rows": REPORT_ROWS,
        "paired_comparisons": PAIRED_COMPARISONS,
    }
    experiment_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    raw_train = load_collapsed_nsl_kdd(train_path, is_train=True)
    labels = raw_train["class"].to_numpy(dtype=np.int64)
    fold_protocol: Dict[str, Dict[str, Any]] = {}
    for seed in args.seeds:
        train_indices, val_indices = split_raw_indices(
            labels,
            seed,
            args.val_split,
        )
        fold_protocol[str(seed)] = {
            "train_indices_sha256": sha256_indices(train_indices),
            "validation_indices_sha256": sha256_indices(val_indices),
            "train_counts": np.bincount(labels[train_indices], minlength=5)
            .astype(int)
            .tolist(),
            "validation_counts": np.bincount(labels[val_indices], minlength=5)
            .astype(int)
            .tolist(),
        }

    results_dir = repo_root / "results"
    stem = f"{prefix}_{experiment_key}"
    run_dir = results_dir / f"{stem}_runs"
    log_dir = results_dir / f"{stem}_logs"
    plan_path = results_dir / f"{stem}_plan.csv"
    protocol_path = results_dir / f"{stem}_protocol.json"

    plans: List[Dict[str, Any]] = []
    for configuration in TRAINING_CONFIGS:
        config_key = hashlib.sha256(
            json.dumps(
                {
                    "experiment_key": experiment_key,
                    "configuration": configuration,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:10]
        for seed in args.seeds:
            run_name = f"{stem}_{configuration['training_id']}_{config_key}_s{seed}"
            fold = fold_protocol[str(seed)]
            plans.append(
                {
                    **configuration,
                    "experiment_key": experiment_key,
                    "config_key": config_key,
                    "seed": int(seed),
                    "run_name": run_name,
                    "train_indices_sha256": fold["train_indices_sha256"],
                    "validation_indices_sha256": fold["validation_indices_sha256"],
                    "result_path": str(run_dir / f"{run_name}.json"),
                    "log_path": str(log_dir / f"{run_name}.log"),
                }
            )
    expected_training_count = len(TRAINING_CONFIGS) * len(args.seeds)
    if len(TRAINING_CONFIGS) != 8 or len(REPORT_ROWS) != 11:
        raise RuntimeError("The fixed design must contain 8 fits and 11 report rows.")
    if len(plans) != expected_training_count:
        raise RuntimeError("Plan construction produced the wrong number of fits.")

    print("No-CTGAN model ablation")
    print(f"Experiment key: {experiment_key}")
    print(f"Seeds: {args.seeds}")
    print(f"GPUs: {gpus}")
    print(f"Unique trainings per seed: {len(TRAINING_CONFIGS)}")
    print(f"Total unique trainings: {len(plans)}")
    print(f"Reported rows: {len(REPORT_ROWS)}")
    print("Split: raw stratified KDDTrain+ 80/20 before preprocessing")
    print("Synthetic data: none")
    print("KDDTest+: final post-training evaluation only")

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
            print(f"... {len(plans) - len(shown)} more commands")
        print("Dry run complete; no files were written and nothing was trained.")
        return

    validate_runtime_dependencies(parser, args)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "experiment_key": experiment_key,
        "title": "No-CTGAN CNN/Transformer/XGBoost ablation",
        "no_ctgan": True,
        "synthetic_files_accessed": False,
        "synthetic_rows": 0,
        "source_and_data_fingerprint": source_fingerprint,
        "kddtrain_sha256": sha256_file(train_path),
        "kddtest_sha256": sha256_file(test_path),
        "split_protocol": (
            "raw KDDTrain+ stratified split before encoder/scaler fitting"
        ),
        "preprocessing_fit_data": "80 percent real training fold only",
        "validation_data": "20 percent held-out real KDDTrain+ fold",
        "test_policy": (
            "KDDTest+ loaded/transformed only after each model is trained and "
            "validation score coefficients are frozen; its file hash is "
            "recorded beforehand solely to identify the dataset version"
        ),
        "score_scaling": (
            "validation grid search; same trained model reused; test never used "
            "for coefficient selection"
        ),
        "family_specific_controls": (
            "CNN and Transformer add class-balanced focal loss, then "
            "minority-guaranteed batches. XGBoost instead adds balanced "
            "per-row class weights; focal loss and mini-batches are not "
            "applied to boosted trees. Score scaling is model-agnostic."
        ),
        "minority_batch_sampling": (
            "each batch guarantees the requested R2L and U2R rows when "
            "available, then samples the remainder from the complete training "
            "fold with replacement; this is not an all-class equal sampler or "
            "an epoch-wide permutation"
        ),
        "prediction_artifacts": (
            "compressed validation/test probabilities, labels, raw/selected "
            "predictions, split indices, and score coefficients saved per fit"
        ),
        "model_selection_metrics": (
            "neural checkpoints use validation Macro-F1; XGBoost early stopping "
            "uses unweighted validation multiclass log-loss. Incremental rows "
            "within each family are controlled; cross-family comparisons use "
            "different native selection criteria."
        ),
        "settings": settings,
        "folds": fold_protocol,
        "training_configurations": TRAINING_CONFIGS,
        "report_rows": REPORT_ROWS,
        "paired_comparisons": PAIRED_COMPARISONS,
        "unique_trainings_per_seed": 8,
        "total_unique_trainings": len(plans),
        "reported_conditions": 11,
    }
    atomic_json(protocol_path, protocol)
    pd.DataFrame(plans).to_csv(plan_path, index=False)
    print(f"Protocol: {protocol_path}")
    print(f"Plan: {plan_path}")

    task_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
    for plan in plans:
        task_queue.put(plan)
    print_lock = threading.Lock()
    data_lock = threading.Lock()
    statuses: Dict[str, str] = {}
    assigned_gpus: Dict[str, str] = {}
    runtimes: Dict[str, float] = {}
    failures: List[str] = []

    def gpu_worker(gpu: str) -> None:
        while True:
            try:
                plan = task_queue.get_nowait()
            except queue.Empty:
                return
            run_name = plan["run_name"]
            result_path = Path(plan["result_path"])
            if not args.rerun and result_is_complete(result_path, plan):
                with data_lock:
                    statuses[run_name] = "skipped_complete"
                    assigned_gpus[run_name] = gpu
                with print_lock:
                    print(f"[GPU {gpu}] SKIP {run_name}", flush=True)
                task_queue.task_done()
                continue

            attempt_id = hashlib.sha256(
                f"{run_name}:{time.time_ns()}:{os.getpid()}".encode("utf-8")
            ).hexdigest()[:16]
            command = build_worker_command(
                script_path,
                plan,
                args,
                attempt_id,
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["EXPERIMENT_GPU_ID"] = gpu
            environment["PYTHONHASHSEED"] = str(plan["seed"])
            environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            log_path = Path(plan["log_path"])
            with print_lock:
                print(f"[GPU {gpu}] START {run_name}", flush=True)
            start_time = time.perf_counter()
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
            runtime = time.perf_counter() - start_time
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
            "The ablation is incomplete. Fix the logged error and rerun; "
            "valid completed jobs will be resumed."
        )

    raw = build_all_run_rows(plans, args.seeds)
    all_runs_path = results_dir / f"{stem}_all_runs.csv"
    raw.to_csv(all_runs_path, index=False)
    paired_deltas = build_paired_deltas(raw, args.seeds)
    paired_deltas_path = results_dir / f"{stem}_paired_deltas.csv"
    paired_deltas.to_csv(paired_deltas_path, index=False)
    summary = build_summary(raw, args.seeds)
    summary_path = results_dir / f"{stem}_summary.csv"
    summary.to_csv(summary_path, index=False)
    pretty = formatted_summary(summary)
    formatted_path = results_dir / f"{stem}_summary_formatted.csv"
    pretty.to_csv(formatted_path, index=False)
    text_path = results_dir / f"{stem}_summary.txt"
    text_path.write_text(
        "No-CTGAN CNN/Transformer/XGBoost ablation\n"
        f"Experiment key: {experiment_key}\n"
        f"Seeds: {args.seeds}\n"
        "Training/validation: raw KDDTrain+ stratified 80/20 split\n"
        "Preprocessing fit: 80% real training fold only\n"
        "KDDTest+: final reporting only; never used for selection\n"
        "Score scaling: selected on real validation fold and frozen\n"
        "Standard deviation: sample SD across paired seeds\n\n"
        + pretty.to_string(index=False)
        + "\n",
        encoding="utf-8",
    )

    print("\n=== No-CTGAN paired-seed KDDTest+ comparison ===")
    print(pretty.to_string(index=False))
    print(f"\nCompleted unique trainings: {len(plans)}/{len(plans)}")
    print(f"Per-seed report rows: {all_runs_path}")
    print(f"Paired incremental deltas: {paired_deltas_path}")
    print(f"Mean/sample-SD summary: {summary_path}")
    print(f"Formatted summary: {formatted_path}")
    print(f"Readable table: {text_path}")
    print(f"Plan/status: {plan_path}")
    print(f"Protocol: {protocol_path}")


if __name__ == "__main__":
    main()
