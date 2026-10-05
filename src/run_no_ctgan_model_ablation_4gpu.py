"""Shared preprocessing, metrics, and artifact helpers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


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
