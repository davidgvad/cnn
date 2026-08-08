"""Run the Robust Calibrated Super-Stack search on saved KDDTrain+ OOF data.

This is a CPU-only meta-learning experiment.  It never opens KDDTest+, never
fits a backbone model, and never uses final-test labels.  General, Focal, and
Minority-Batching OOF probability artifacts are treated as frozen inputs.

For every requested architecture the runner performs:

* four-fold nested *meta-level* validation (three-fold tuning inside each
  held-out outer fold), with a shared configuration selected across seeds;
* raw versus per-expert temperature calibration fitted inside each meta-
  training split;
* F0/F1/F2 confidence and disagreement feature families;
* L2 multinomial logistic stacking over five class-weight exponents and seven
  regularization strengths;
* shrinkage to the original raw three-expert probability average;
* independent R2L/U2R log-score offsets;
* natural and rare-subtype-balanced validation views;
* a separate four-fold all-OOF selection followed by one deployable refit per
  seed.

The base-model OOF predictions were created before this script and are not
nested again inside the meta outer folds.  The resulting estimate is therefore
"nested meta-level CV conditional on frozen cross-fitted expert predictions",
not fully nested end-to-end validation.  A fully nested claim would require
retraining the neural experts inside every meta outer split.

Example:

    python -u src/tune_robust_calibrated_super_stack_all.py --dry-run
    python -u src/tune_robust_calibrated_super_stack_all.py --workers 32

All numerical-library threads are limited inside workers.  Independent
calibration/feature tasks are distributed over CPU processes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Sequence
import warnings
import zipfile

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import run_no_ctgan_model_ablation_4gpu as core
import tune_conv2d_safe_stack_fusion as safe


SCHEMA_VERSION = 1
ARCHITECTURES = ("conv2d", "conv1d", "transformer", "mlp")
ARCHITECTURE_LABELS = dict(safe.ARCHITECTURE_LABELS)
EXPERTS = tuple(safe.EXPERTS)
EXPERT_LABELS = dict(safe.EXPERT_LABELS)
SEEDS_DEFAULT = (0, 1, 2)
FOLD_IDS = (0, 1, 2, 3)
CLASS_COUNT = 5
R2L_CLASS = 2
U2R_CLASS = 3
RARE_CLASSES = (R2L_CLASS, U2R_CLASS)
MAJORITY_CLASSES = (0, 1, 4)

CALIBRATIONS = ("raw", "temperature")
FEATURE_SETS = ("F0", "F1", "F2")
Q_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
RHO_VALUES = (0.0, 0.25, 0.50, 0.75, 1.0)
NONZERO_RHO_VALUES = RHO_VALUES[1:]
DELTA_VALUES = (-1.0, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0)
OFFSET_PAIRS = tuple((dr, du) for dr in DELTA_VALUES for du in DELTA_VALUES)
NOMINAL_LIBRARY_COUNT = 85_054
CANONICAL_LIBRARY_COUNT = 68_124

METRICS = tuple(core.METRICS)
METRIC_INDEX = {name: index for index, name in enumerate(METRICS)}
EPSILON_DEFAULT = 1e-12
MACRO_GUARD_DEFAULT = 0.005
MCC_GUARD_DEFAULT = 0.005

R2L_SUBTYPES = (
    "warezclient",
    "guess_passwd",
    "warezmaster",
    "imap",
    "ftp_write",
    "multihop",
    "phf",
    "spy",
)
U2R_SUBTYPES = ("buffer_overflow", "rootkit", "loadmodule", "perl")
DOS_SUBTYPES = ("neptune", "smurf", "back", "teardrop", "pod", "land")
PROBE_SUBTYPES = ("satan", "ipsweep", "portsweep", "nmap")

# A worker receives this state through fork copy-on-write.  The arrays are
# read-only by convention; each worker allocates only split-local features.
_WORKER_CONTEXT: Dict[str, Any] | None = None
_WORKER_THREAD_LIMITER: Any = None


@dataclass(frozen=True)
class SearchSettings:
    epsilon: float
    max_iter: int
    tolerance: float
    temperature_min: float
    temperature_max: float
    temperature_xatol: float
    temperature_maxiter: int
    candidate_chunk_size: int
    macro_guard: float
    mcc_guard: float


@dataclass(frozen=True)
class MetaTask:
    architecture: str
    stage: str
    outer_fold: int
    seed: int
    calibration: str
    feature_set: str
    fingerprint: str
    cache_path: str


@dataclass
class ArchitectureInput:
    architecture: str
    seeds: tuple[int, ...]
    probabilities: np.ndarray  # [seed, expert, row, class], float32
    labels: np.ndarray
    fold_ids: np.ndarray
    subtypes: np.ndarray
    source_paths: Dict[str, Dict[int, str]]
    source_hashes: Dict[str, Dict[int, str]]
    pointer_paths: Dict[str, str]
    pointer_hashes: Dict[str, str]
    protocol_paths: Dict[str, str]
    protocol_hashes: Dict[str, str]
    focal_best: Dict[str, Any]


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    core.atomic_npz(path, **arrays)


def available_logical_cpus() -> int:
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def detect_physical_cpus() -> int:
    """Return physical cores within the current Linux CPU affinity mask."""
    allowed = (
        set(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else set(range(available_logical_cpus()))
    )
    topology_root = Path("/sys/devices/system/cpu")
    physical: set[tuple[str, str]] = set()
    for cpu in sorted(allowed):
        topology = topology_root / f"cpu{cpu}" / "topology"
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core_id = (topology / "core_id").read_text().strip()
        except OSError:
            continue
        physical.add((package, core_id))
    if physical:
        return len(physical)
    # Conservative fallback for the common SMT=2 server layout.
    return max(1, available_logical_cpus() // 2)


def resolve_template(repo_root: Path, template: str, architecture: str) -> Path:
    return safe.resolve_cli_path(repo_root, template.format(architecture=architecture))


def read_protocol_from_pointer(repo_root: Path, pointer_path: Path) -> tuple[Path, Dict[str, Any]]:
    pointer = safe.read_json(pointer_path)
    if "protocol" not in pointer:
        raise KeyError(f"Missing protocol in {pointer_path}.")
    protocol_path = safe.resolve_recorded_path(repo_root, pointer_path, pointer["protocol"])
    return protocol_path, safe.read_json(protocol_path)


def validate_protocol_lineage(
    repo_root: Path,
    pointer_path: Path,
    train_path: Path,
    artifacts: Mapping[int, safe.OOFArtifact],
    expert: str,
    selected_config_id: str | None = None,
) -> tuple[Path, Dict[str, Any]]:
    """Validate recorded data/fold/OOF hashes when the source exposes them."""
    pointer = safe.read_json(pointer_path)
    protocol_path, protocol = read_protocol_from_pointer(repo_root, pointer_path)
    train_hash = core.sha256_file(train_path)
    if "kddtrain_sha256" not in protocol:
        raise KeyError(f"Missing kddtrain_sha256 in {protocol_path}.")
    recorded_train_hash = protocol["kddtrain_sha256"]
    if recorded_train_hash != train_hash:
        raise ValueError(
            f"KDDTrain+ hash mismatch for {pointer_path}: protocol has "
            f"{recorded_train_hash}, current file is {train_hash}."
        )
    if protocol.get("kddtest_accessed") is not False:
        raise ValueError(f"{protocol_path} reports KDDTest+ access.")
    if "synthetic_rows" not in protocol or int(protocol["synthetic_rows"]) != 0:
        raise ValueError(f"{protocol_path} reports synthetic rows.")

    reference = artifacts[sorted(artifacts)[0]]
    fold_records = protocol.get("folds", [])
    by_id = {int(row["fold_id"]): row for row in fold_records}
    if len(fold_records) != len(FOLD_IDS) or sorted(by_id) != list(FOLD_IDS):
        raise ValueError(f"Unexpected or missing fold records in {protocol_path}.")
    all_indices = np.arange(len(reference.labels), dtype=np.int64)
    for fold_id in FOLD_IDS:
        validation = all_indices[reference.fold_ids == fold_id]
        training = all_indices[reference.fold_ids != fold_id]
        row = by_id[fold_id]
        for required in ("validation_indices_sha256", "train_indices_sha256"):
            if required not in row:
                raise KeyError(f"Missing {required} in {protocol_path}, fold {fold_id}.")
        if core.sha256_indices(validation) != row["validation_indices_sha256"]:
            raise ValueError(f"Validation fold hash mismatch in {protocol_path}, fold {fold_id}.")
        if core.sha256_indices(training) != row["train_indices_sha256"]:
            raise ValueError(f"Training fold hash mismatch in {protocol_path}, fold {fold_id}.")

    assignment_value = protocol.get("fold_assignment_path")
    assignment_hash = protocol.get("fold_assignment_sha256")
    if not assignment_value or not assignment_hash:
        raise KeyError(f"Missing fold-assignment path/hash in {protocol_path}.")
    assignment_path = safe.resolve_recorded_path(repo_root, protocol_path, assignment_value)
    if core.sha256_file(assignment_path) != assignment_hash:
        raise ValueError(f"Fold-assignment file hash mismatch: {assignment_path}")

    metrics_key = "seed_metrics" if expert == "focal" else "raw_seed_metrics"
    metrics_value = pointer.get(metrics_key)
    if not metrics_value:
        raise KeyError(f"Missing {metrics_key} in {pointer_path}.")
    metrics_path = safe.resolve_recorded_path(repo_root, pointer_path, metrics_value)
    metrics = pd.read_csv(metrics_path)
    required_columns = {"seed", "oof_sha256"}
    if selected_config_id is not None:
        required_columns.add("config_id")
    missing_columns = required_columns - set(metrics.columns)
    if missing_columns:
        raise KeyError(f"{metrics_path} is missing columns: {sorted(missing_columns)}")
    if selected_config_id is not None:
        metrics = metrics[metrics["config_id"].astype(str) == selected_config_id]
    expected_seeds = sorted(int(seed) for seed in artifacts)
    observed_seeds = sorted(metrics["seed"].astype(int).tolist())
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Expected exactly one OOF hash row for seeds {expected_seeds} in "
            f"{metrics_path}, got {observed_seeds}."
        )
    expected = {
        int(row.seed): str(row.oof_sha256)
        for row in metrics[["seed", "oof_sha256"]].itertuples(index=False)
    }
    for seed, artifact in artifacts.items():
        if artifact.sha256 != expected[seed]:
            raise ValueError(f"OOF hash mismatch for {artifact.path}.")
    return protocol_path, protocol


def collapse_subtypes(subtypes: np.ndarray) -> np.ndarray:
    mapping: Dict[str, int] = {"normal": 4}
    mapping.update({name: 0 for name in DOS_SUBTYPES})
    mapping.update({name: 1 for name in PROBE_SUBTYPES})
    mapping.update({name: 2 for name in R2L_SUBTYPES})
    mapping.update({name: 3 for name in U2R_SUBTYPES})
    normalized = np.char.lower(np.char.strip(np.asarray(subtypes, dtype=str)))
    unknown = sorted(set(normalized.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown KDDTrain+ attack subtypes: {unknown}")
    return np.fromiter((mapping[value] for value in normalized), dtype=np.int64)


def load_train_subtypes(train_path: Path, expected_labels: np.ndarray) -> np.ndarray:
    frame = pd.read_csv(
        train_path,
        header=None,
        usecols=[41],
        dtype={41: str},
    )
    subtypes = frame.iloc[:, 0].astype(str).str.strip().str.lower().to_numpy(copy=True)
    if len(subtypes) != len(expected_labels):
        raise ValueError(
            f"KDDTrain+ has {len(subtypes)} rows, OOF artifacts have {len(expected_labels)}."
        )
    collapsed = collapse_subtypes(subtypes)
    if not np.array_equal(collapsed, np.asarray(expected_labels, dtype=np.int64)):
        raise ValueError("Raw KDDTrain+ subtypes do not collapse to the OOF labels.")
    return subtypes


def subtype_balanced_weights(labels: np.ndarray, subtypes: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    labels = np.asarray(labels, dtype=np.int64)
    subtypes = np.asarray(subtypes, dtype=str)
    if labels.shape != subtypes.shape:
        raise ValueError("Subtype labels and class labels must have identical shape.")
    weights = np.ones(len(labels), dtype=np.float64)
    details: Dict[str, Any] = {}
    canonical = {R2L_CLASS: R2L_SUBTYPES, U2R_CLASS: U2R_SUBTYPES}
    for class_id, label in ((R2L_CLASS, "R2L"), (U2R_CLASS, "U2R")):
        mask = labels == class_id
        if not np.any(mask):
            raise ValueError(f"Scored partition contains no {label} rows.")
        names, inverse, counts = np.unique(
            subtypes[mask], return_inverse=True, return_counts=True
        )
        total = int(mask.sum())
        subtype_count = len(names)
        weights[mask] = (total / subtype_count) / counts[inverse]
        details[label] = {
            "rows": total,
            "observed_subtypes": names.tolist(),
            "missing_subtypes": sorted(set(canonical[class_id]) - set(names.tolist())),
            "counts": {name: int(count) for name, count in zip(names, counts)},
            "row_weights": {
                name: float((total / subtype_count) / count)
                for name, count in zip(names, counts)
            },
        }
        if not np.isclose(weights[mask].sum(), total, atol=1e-10, rtol=0.0):
            raise RuntimeError(f"Subtype weights do not preserve {label} mass.")
    return weights, details


def load_architecture_input(
    repo_root: Path,
    train_path: Path,
    architecture: str,
    seeds: Sequence[int],
    general_template: str,
    focal_template: str,
    batching_template: str,
) -> ArchitectureInput:
    pointer_paths = {
        "general": resolve_template(repo_root, general_template, architecture),
        "focal": resolve_template(repo_root, focal_template, architecture),
        "batching": resolve_template(repo_root, batching_template, architecture),
    }
    standard_paths = {
        "general": safe.discover_standard_oof_paths(
            repo_root,
            pointer_paths["general"],
            seeds,
            expected_training_mode="baseline_ce",
            expected_architecture=architecture,
        ),
        "batching": safe.discover_standard_oof_paths(
            repo_root,
            pointer_paths["batching"],
            seeds,
            expected_training_mode="baseline_batch",
            expected_architecture=architecture,
        ),
    }
    focal_paths, focal_best = safe.discover_focal_oof_paths(
        repo_root,
        pointer_paths["focal"],
        seeds,
        expected_architecture=architecture,
    )
    source_paths: Dict[str, Dict[int, Path]] = {
        **standard_paths,
        "focal": focal_paths,
    }
    artifacts: Dict[str, Dict[int, safe.OOFArtifact]] = {
        expert: {
            int(seed): safe.load_oof_artifact(expert, int(seed), source_paths[expert][int(seed)])
            for seed in seeds
        }
        for expert in EXPERTS
    }
    safe.validate_alignment(artifacts, seeds)

    protocol_paths: Dict[str, str] = {}
    protocol_hashes: Dict[str, str] = {}
    selected_config_id = str(focal_best.get("config_id", ""))
    for expert in EXPERTS:
        protocol_path, _ = validate_protocol_lineage(
            repo_root,
            pointer_paths[expert],
            train_path,
            artifacts[expert],
            expert,
            selected_config_id if expert == "focal" else None,
        )
        protocol_paths[expert] = str(protocol_path)
        protocol_hashes[expert] = core.sha256_file(protocol_path)

    ordered_seeds = tuple(sorted(int(seed) for seed in seeds))
    probabilities = np.stack(
        [
            np.stack(
                [artifacts[expert][seed].probabilities for expert in EXPERTS],
                axis=0,
            )
            for seed in ordered_seeds
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    reference = artifacts[EXPERTS[0]][ordered_seeds[0]]
    subtypes = load_train_subtypes(train_path, reference.labels)
    return ArchitectureInput(
        architecture=architecture,
        seeds=ordered_seeds,
        probabilities=probabilities,
        labels=np.asarray(reference.labels, dtype=np.int64),
        fold_ids=np.asarray(reference.fold_ids, dtype=np.int64),
        subtypes=subtypes,
        source_paths={
            expert: {seed: str(source_paths[expert][seed]) for seed in ordered_seeds}
            for expert in EXPERTS
        },
        source_hashes={
            expert: {seed: artifacts[expert][seed].sha256 for seed in ordered_seeds}
            for expert in EXPERTS
        },
        pointer_paths={key: str(value) for key, value in pointer_paths.items()},
        pointer_hashes={key: core.sha256_file(value) for key, value in pointer_paths.items()},
        protocol_paths=protocol_paths,
        protocol_hashes=protocol_hashes,
        focal_best=focal_best,
    )


def validate_cross_architecture_alignment(inputs: Sequence[ArchitectureInput]) -> None:
    if not inputs:
        raise ValueError("No architecture inputs were loaded.")
    reference = inputs[0]
    for current in inputs[1:]:
        for name in ("labels", "fold_ids", "subtypes"):
            if not np.array_equal(getattr(current, name), getattr(reference, name)):
                raise ValueError(
                    f"Cross-architecture alignment failure: {current.architecture} {name} "
                    f"differs from {reference.architecture}."
                )


def temperature_scale(probabilities: np.ndarray, temperature: float, epsilon: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Temperature must be positive, got {temperature}.")
    log_values = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0))
    scaled = log_values / float(temperature)
    scaled -= logsumexp(scaled, axis=1, keepdims=True)
    result = np.exp(scaled)
    if not np.isfinite(result).all():
        raise RuntimeError("Temperature scaling produced non-finite probabilities.")
    return result


def fit_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    settings: SearchSettings,
) -> tuple[float, bool]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    log_values = np.log(np.clip(probabilities, settings.epsilon, 1.0))
    rows = np.arange(len(labels), dtype=np.int64)

    def objective(log_temperature: float) -> float:
        scaled = log_values / math.exp(float(log_temperature))
        value = -np.mean(scaled[rows, labels] - logsumexp(scaled, axis=1))
        return float(value)

    low = math.log(settings.temperature_min)
    high = math.log(settings.temperature_max)
    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=(low, high),
        options={
            "xatol": settings.temperature_xatol,
            "maxiter": settings.temperature_maxiter,
        },
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x):
        raise RuntimeError(f"Temperature optimization failed: {result}")
    temperature = math.exp(float(result.x))
    boundary = bool(
        np.isclose(result.x, low, atol=10.0 * settings.temperature_xatol, rtol=0.0)
        or np.isclose(result.x, high, atol=10.0 * settings.temperature_xatol, rtol=0.0)
    )
    return temperature, boundary


def feature_names(feature_set: str) -> List[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    names = [f"{expert}_log_p_class_{class_id}" for expert in EXPERTS for class_id in range(5)]
    if feature_set in {"F1", "F2"}:
        names.extend(f"{expert}_entropy" for expert in EXPERTS)
        names.extend(f"{expert}_top1_top2_probability_margin" for expert in EXPERTS)
        names.extend(
            f"{expert}_{rare}_vs_max_majority_log_margin"
            for expert in EXPERTS
            for rare in ("r2l", "u2r")
        )
        names.extend(("js_general_focal", "js_general_batching", "js_focal_batching"))
    if feature_set == "F2":
        for rare in ("r2l", "u2r"):
            names.extend(
                (
                    f"{rare}_margin_general_x_focal",
                    f"{rare}_margin_general_x_batching",
                    f"{rare}_margin_focal_x_batching",
                )
            )
    return names


def build_features(probabilities: np.ndarray, feature_set: str, epsilon: float) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[1:] != (3, 5):
        raise ValueError(f"Expected probability shape (N,3,5), got {probabilities.shape}.")
    clipped = np.clip(probabilities, epsilon, 1.0)
    clipped /= clipped.sum(axis=2, keepdims=True)
    logs = np.log(clipped)
    parts: List[np.ndarray] = [logs.reshape(len(clipped), 15)]
    rare_margins: np.ndarray | None = None
    if feature_set in {"F1", "F2"}:
        entropy = -np.sum(clipped * logs, axis=2)
        partitioned = np.partition(clipped, kth=3, axis=2)
        top_margin = partitioned[:, :, 4] - partitioned[:, :, 3]
        majority_log = np.max(logs[:, :, MAJORITY_CLASSES], axis=2)
        rare_margins = logs[:, :, RARE_CLASSES] - majority_log[:, :, None]
        js_columns = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            midpoint = 0.5 * (clipped[:, first] + clipped[:, second])
            midpoint_log = np.log(np.clip(midpoint, epsilon, 1.0))
            js_columns.append(
                0.5
                * np.sum(
                    clipped[:, first] * (logs[:, first] - midpoint_log)
                    + clipped[:, second] * (logs[:, second] - midpoint_log),
                    axis=1,
                )
            )
        parts.extend(
            [
                entropy,
                top_margin,
                rare_margins.reshape(len(clipped), 6),
                np.column_stack(js_columns),
            ]
        )
    if feature_set == "F2":
        assert rare_margins is not None
        interactions = []
        for rare_index in range(2):
            interactions.extend(
                (
                    rare_margins[:, 0, rare_index] * rare_margins[:, 1, rare_index],
                    rare_margins[:, 0, rare_index] * rare_margins[:, 2, rare_index],
                    rare_margins[:, 1, rare_index] * rare_margins[:, 2, rare_index],
                )
            )
        parts.append(np.column_stack(interactions))
    features = np.column_stack(parts)
    expected = {"F0": 15, "F1": 30, "F2": 36}[feature_set]
    if features.shape != (len(probabilities), expected):
        raise RuntimeError(f"{feature_set} produced shape {features.shape}, expected (*,{expected}).")
    if not np.isfinite(features).all():
        raise RuntimeError(f"{feature_set} produced non-finite features.")
    return features


def fit_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, ddof=0, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def apply_standardizer(features: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    result = (np.asarray(features, dtype=np.float64) - mean) / scale
    if not np.isfinite(result).all():
        raise RuntimeError("Feature standardization produced non-finite values.")
    return result


def normalized_sample_weights(labels: np.ndarray, q: float) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=CLASS_COUNT).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Meta-training split is missing a class: {counts.astype(int).tolist()}")
    class_weights = (len(labels) / counts) ** float(q)
    sample_weights = class_weights[labels]
    sample_weights /= sample_weights.mean()
    return sample_weights


def metrics_from_confusions(confusions: np.ndarray) -> np.ndarray:
    matrices = np.asarray(confusions, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (5, 5):
        raise ValueError(f"Expected confusion shape (P,5,5), got {matrices.shape}.")
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
    accuracy = np.divide(correct, totals, out=np.zeros_like(correct), where=totals != 0)
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
    columns = {
        "accuracy": accuracy,
        "mcc": mcc,
        "macro_f1": f1.mean(axis=1),
        "macro_recall": recall.mean(axis=1),
        "rare_f1": f1[:, [R2L_CLASS, U2R_CLASS]].mean(axis=1),
        "minimum_minority_recall": recall[:, [R2L_CLASS, U2R_CLASS]].min(axis=1),
        "r2l_precision": precision[:, R2L_CLASS],
        "r2l_recall": recall[:, R2L_CLASS],
        "r2l_f1": f1[:, R2L_CLASS],
        "u2r_precision": precision[:, U2R_CLASS],
        "u2r_recall": recall[:, U2R_CLASS],
        "u2r_f1": f1[:, U2R_CLASS],
    }
    return np.column_stack([columns[name] for name in METRICS])


def confusion_batch(
    labels: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return one 5x5 confusion per prediction column."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    if predictions.shape[0] != len(labels):
        raise ValueError("Prediction rows and labels do not align.")
    candidate_count = predictions.shape[1]
    codes = (
        25 * np.arange(candidate_count, dtype=np.int64)[None, :]
        + 5 * labels[:, None]
        + predictions
    )
    repeated_weights = None
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != labels.shape:
            raise ValueError("Confusion weights do not align with labels.")
        repeated_weights = np.repeat(weights, candidate_count)
    counts = np.bincount(
        codes.ravel(),
        weights=repeated_weights,
        minlength=candidate_count * 25,
    )
    return counts.reshape(candidate_count, 5, 5)


def apply_offsets(
    probabilities: np.ndarray,
    delta_r2l: float,
    delta_u2r: float,
    epsilon: float,
) -> np.ndarray:
    return np.argmax(
        offset_decision_probabilities(
            probabilities, delta_r2l, delta_u2r, epsilon
        ),
        axis=1,
    ).astype(np.int64)


def offset_decision_probabilities(
    probabilities: np.ndarray,
    delta_r2l: float,
    delta_u2r: float,
    epsilon: float,
) -> np.ndarray:
    scores = np.log(np.clip(np.asarray(probabilities, dtype=np.float64), epsilon, 1.0))
    scores[:, R2L_CLASS] += float(delta_r2l)
    scores[:, U2R_CLASS] += float(delta_u2r)
    scores -= logsumexp(scores, axis=1, keepdims=True)
    result = np.exp(scores)
    if not np.isfinite(result).all():
        raise RuntimeError("Rare-offset adjustment produced non-finite probabilities.")
    return result


def score_probability_family(
    labels: np.ndarray,
    balanced_weights: np.ndarray,
    average_probabilities: np.ndarray,
    stack_probabilities: np.ndarray,
    settings: SearchSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Score 4 rhos x 81 offsets for one stack probability matrix."""
    natural = np.empty((len(NONZERO_RHO_VALUES), len(OFFSET_PAIRS), len(METRICS)), dtype=np.float64)
    balanced_rare = np.empty((len(NONZERO_RHO_VALUES), len(OFFSET_PAIRS)), dtype=np.float64)
    balanced_r2l = np.empty_like(balanced_rare)
    balanced_u2r = np.empty_like(balanced_rare)
    offset_array = np.asarray(OFFSET_PAIRS, dtype=np.float64)
    for rho_index, rho in enumerate(NONZERO_RHO_VALUES):
        blended = (1.0 - rho) * average_probabilities + rho * stack_probabilities
        base_scores = np.log(np.clip(blended, settings.epsilon, 1.0))
        for start in range(0, len(OFFSET_PAIRS), settings.candidate_chunk_size):
            stop = min(start + settings.candidate_chunk_size, len(OFFSET_PAIRS))
            offsets = offset_array[start:stop]
            scores = np.broadcast_to(
                base_scores[:, None, :],
                (len(labels), len(offsets), CLASS_COUNT),
            ).copy()
            scores[:, :, R2L_CLASS] += offsets[None, :, 0]
            scores[:, :, U2R_CLASS] += offsets[None, :, 1]
            predictions = np.argmax(scores, axis=2).astype(np.int64)
            natural_confusions = confusion_batch(labels, predictions)
            balanced_confusions = confusion_batch(labels, predictions, balanced_weights)
            natural[rho_index, start:stop] = metrics_from_confusions(natural_confusions)
            balanced_metrics = metrics_from_confusions(balanced_confusions)
            balanced_rare[rho_index, start:stop] = balanced_metrics[:, METRIC_INDEX["rare_f1"]]
            balanced_r2l[rho_index, start:stop] = balanced_metrics[:, METRIC_INDEX["r2l_f1"]]
            balanced_u2r[rho_index, start:stop] = balanced_metrics[:, METRIC_INDEX["u2r_f1"]]
    return natural, balanced_rare, balanced_r2l, balanced_u2r


def _worker_initializer(threads_per_worker: int) -> None:
    global _WORKER_THREAD_LIMITER
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads_per_worker)
    _WORKER_THREAD_LIMITER = threadpool_limits(limits=threads_per_worker)


def _task_cache_valid(task: MetaTask) -> bool:
    path = Path(task.cache_path)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["fingerprint"].item()) != task.fingerprint:
                return False
            required = {
                "fingerprint",
                "natural_metrics",
                "balanced_rare_f1",
                "balanced_r2l_f1",
                "balanced_u2r_f1",
                "valid_q_c",
                "iterations",
                "converged",
                "temperatures",
                "temperature_boundary",
                "runtime_seconds",
            }
            if not required.issubset(data.files):
                return False
            expected_metrics = (
                len(Q_VALUES),
                len(C_VALUES),
                len(NONZERO_RHO_VALUES),
                len(OFFSET_PAIRS),
                len(METRICS),
            )
            if data["natural_metrics"].shape != expected_metrics:
                return False
            expected_balanced = expected_metrics[:-1]
            for key in ("balanced_rare_f1", "balanced_r2l_f1", "balanced_u2r_f1"):
                if data[key].shape != expected_balanced:
                    return False
            if data["valid_q_c"].shape != (len(Q_VALUES), len(C_VALUES)):
                return False
            expected_fold_count = 4 if task.outer_fold < 0 else 3
            expected_fit_shape = (expected_fold_count, len(Q_VALUES), len(C_VALUES))
            if data["iterations"].shape != expected_fit_shape:
                return False
            if data["converged"].shape != expected_fit_shape:
                return False
            expected_temperature_shape = (expected_fold_count, len(EXPERTS))
            if data["temperatures"].shape != expected_temperature_shape:
                return False
            if data["temperature_boundary"].shape != expected_temperature_shape:
                return False
            if data["runtime_seconds"].shape != ():
                return False
            for key in ("valid_q_c", "converged", "temperature_boundary"):
                if data[key].dtype.kind != "b":
                    return False
            if not np.array_equal(
                data["valid_q_c"], np.all(data["converged"], axis=0)
            ):
                return False
            if data["iterations"].dtype.kind not in {"i", "u"}:
                return False
            if np.any(data["iterations"] < 0):
                return False
            for key in (
                "natural_metrics",
                "balanced_rare_f1",
                "balanced_r2l_f1",
                "balanced_u2r_f1",
                "temperatures",
                "runtime_seconds",
            ):
                if not np.isfinite(data[key]).all():
                    return False
            if np.any(data["temperatures"] <= 0.0) or float(data["runtime_seconds"]) < 0.0:
                return False
    except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile):
        return False
    return True


def _run_meta_task(task: MetaTask) -> Dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context was not initialized.")
    started = time.perf_counter()
    context = _WORKER_CONTEXT
    settings: SearchSettings = context["settings"]
    seeds: tuple[int, ...] = context["seeds"]
    try:
        seed_index = seeds.index(task.seed)
    except ValueError as error:
        raise RuntimeError(f"Unknown task seed {task.seed}.") from error
    probabilities = np.asarray(context["probabilities"][seed_index], dtype=np.float64)
    labels = np.asarray(context["labels"], dtype=np.int64)
    fold_ids = np.asarray(context["fold_ids"], dtype=np.int64)
    subtypes = np.asarray(context["subtypes"], dtype=str)

    scope_mask = np.ones(len(labels), dtype=bool) if task.outer_fold < 0 else fold_ids != task.outer_fold
    scope_indices = np.flatnonzero(scope_mask)
    validation_folds = [fold for fold in FOLD_IDS if task.outer_fold < 0 or fold != task.outer_fold]
    scope_position = np.full(len(labels), -1, dtype=np.int64)
    scope_position[scope_indices] = np.arange(len(scope_indices), dtype=np.int64)
    crossfit = np.full(
        (len(Q_VALUES), len(C_VALUES), len(scope_indices), CLASS_COUNT),
        np.nan,
        dtype=np.float64,
    )
    iterations = np.zeros((len(validation_folds), len(Q_VALUES), len(C_VALUES)), dtype=np.int64)
    converged = np.ones_like(iterations, dtype=bool)
    temperatures = np.ones((len(validation_folds), len(EXPERTS)), dtype=np.float64)
    temperature_boundary = np.zeros_like(temperatures, dtype=bool)

    for fold_position, validation_fold in enumerate(validation_folds):
        train_mask = scope_mask & (fold_ids != validation_fold)
        validation_mask = fold_ids == validation_fold
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise RuntimeError(f"Empty meta split for {task.stage}, fold {validation_fold}.")
        active_train = probabilities[:, train_indices, :].transpose(1, 0, 2)
        active_validation = probabilities[:, validation_indices, :].transpose(1, 0, 2)
        if task.calibration == "temperature":
            active_train = active_train.copy()
            active_validation = active_validation.copy()
            for expert_index in range(len(EXPERTS)):
                temperature, boundary = fit_temperature(
                    active_train[:, expert_index, :], labels[train_indices], settings
                )
                temperatures[fold_position, expert_index] = temperature
                temperature_boundary[fold_position, expert_index] = boundary
                active_train[:, expert_index, :] = temperature_scale(
                    active_train[:, expert_index, :], temperature, settings.epsilon
                )
                active_validation[:, expert_index, :] = temperature_scale(
                    active_validation[:, expert_index, :], temperature, settings.epsilon
                )
        elif task.calibration != "raw":
            raise ValueError(f"Unknown calibration: {task.calibration}")

        train_features = build_features(active_train, task.feature_set, settings.epsilon)
        validation_features = build_features(active_validation, task.feature_set, settings.epsilon)
        feature_mean, feature_scale = fit_standardizer(train_features)
        train_features = apply_standardizer(train_features, feature_mean, feature_scale)
        validation_features = apply_standardizer(validation_features, feature_mean, feature_scale)
        validation_positions = scope_position[validation_indices]
        for q_index, q in enumerate(Q_VALUES):
            sample_weights = normalized_sample_weights(labels[train_indices], q)
            for c_index, c_value in enumerate(C_VALUES):
                model = LogisticRegression(
                    C=float(c_value),
                    penalty="l2",
                    solver="lbfgs",
                    fit_intercept=True,
                    tol=settings.tolerance,
                    max_iter=settings.max_iter,
                    random_state=0,
                )
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(train_features, labels[train_indices], sample_weight=sample_weights)
                did_converge = not any(
                    issubclass(item.category, ConvergenceWarning) for item in caught
                )
                converged[fold_position, q_index, c_index] = did_converge
                iterations[fold_position, q_index, c_index] = int(np.max(model.n_iter_))
                if not np.array_equal(model.classes_, np.arange(CLASS_COUNT)):
                    raise RuntimeError(f"Unexpected logistic classes: {model.classes_.tolist()}")
                predicted = model.predict_proba(validation_features)
                if not np.isfinite(predicted).all():
                    raise RuntimeError("Logistic regression produced non-finite probabilities.")
                crossfit[q_index, c_index, validation_positions, :] = predicted

    if not np.isfinite(crossfit).all():
        raise RuntimeError(f"Cross-fitted stack probabilities are incomplete for {task}.")
    valid_q_c = np.all(converged, axis=0)
    scope_labels = labels[scope_indices]
    scope_subtypes = subtypes[scope_indices]
    balanced_weights, _ = subtype_balanced_weights(scope_labels, scope_subtypes)
    raw_probabilities = probabilities[:, scope_indices, :].transpose(1, 0, 2)
    raw_average = np.mean(raw_probabilities, axis=1, dtype=np.float64)

    natural_metrics = np.empty(
        (
            len(Q_VALUES),
            len(C_VALUES),
            len(NONZERO_RHO_VALUES),
            len(OFFSET_PAIRS),
            len(METRICS),
        ),
        dtype=np.float64,
    )
    balanced_rare = np.empty(natural_metrics.shape[:-1], dtype=np.float64)
    balanced_r2l = np.empty_like(balanced_rare)
    balanced_u2r = np.empty_like(balanced_rare)
    for q_index in range(len(Q_VALUES)):
        for c_index in range(len(C_VALUES)):
            values = score_probability_family(
                scope_labels,
                balanced_weights,
                raw_average,
                crossfit[q_index, c_index],
                settings,
            )
            natural_metrics[q_index, c_index] = values[0]
            balanced_rare[q_index, c_index] = values[1]
            balanced_r2l[q_index, c_index] = values[2]
            balanced_u2r[q_index, c_index] = values[3]

    cache_path = Path(task.cache_path)
    atomic_npz(
        cache_path,
        fingerprint=np.asarray(task.fingerprint),
        natural_metrics=natural_metrics,
        balanced_rare_f1=balanced_rare,
        balanced_r2l_f1=balanced_r2l,
        balanced_u2r_f1=balanced_u2r,
        valid_q_c=valid_q_c,
        iterations=iterations,
        converged=converged,
        temperatures=temperatures,
        temperature_boundary=temperature_boundary,
        runtime_seconds=np.asarray(time.perf_counter() - started, dtype=np.float64),
    )
    return {
        "task": asdict(task),
        "runtime_seconds": float(time.perf_counter() - started),
        "valid_configurations": int(valid_q_c.sum()),
        "total_configurations": int(valid_q_c.size),
        "temperature_boundaries": int(temperature_boundary.sum()),
    }


def task_stage(outer_fold: int) -> str:
    return "final_cv" if outer_fold < 0 else f"outer_{outer_fold}"


def make_tasks(
    architecture: str,
    seeds: Sequence[int],
    experiment_key: str,
    cache_dir: Path,
) -> List[MetaTask]:
    tasks: List[MetaTask] = []
    for outer_fold in (*FOLD_IDS, -1):
        stage = task_stage(outer_fold)
        for seed in seeds:
            for calibration in CALIBRATIONS:
                for feature_set in FEATURE_SETS:
                    identity = {
                        "experiment_key": experiment_key,
                        "architecture": architecture,
                        "stage": stage,
                        "outer_fold": outer_fold,
                        "seed": int(seed),
                        "calibration": calibration,
                        "feature_set": feature_set,
                    }
                    fingerprint = stable_hash(identity, length=24)
                    filename = (
                        f"{stage}_s{seed}_{calibration}_{feature_set.lower()}_"
                        f"{fingerprint}.npz"
                    )
                    tasks.append(
                        MetaTask(
                            architecture=architecture,
                            stage=stage,
                            outer_fold=outer_fold,
                            seed=int(seed),
                            calibration=calibration,
                            feature_set=feature_set,
                            fingerprint=fingerprint,
                            cache_path=str(cache_dir / filename),
                        )
                    )
    return tasks


def load_task_arrays(task: MetaTask) -> Dict[str, np.ndarray]:
    if not _task_cache_valid(task):
        raise ValueError(f"Invalid or incomplete task cache: {task.cache_path}")
    with np.load(task.cache_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def metric_values_for_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
    balanced_weights: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    natural = metrics_from_confusions(confusion_batch(labels, predictions))[0]
    balanced = metrics_from_confusions(
        confusion_batch(labels, predictions, balanced_weights)
    )[0]
    return (
        natural,
        float(balanced[METRIC_INDEX["rare_f1"]]),
        float(balanced[METRIC_INDEX["r2l_f1"]]),
        float(balanced[METRIC_INDEX["u2r_f1"]]),
    )


def fixed_and_average_offset_seed_metrics(
    architecture_input: ArchitectureInput,
    outer_fold: int,
    settings: SearchSettings,
) -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
]:
    scope_mask = (
        np.ones(len(architecture_input.labels), dtype=bool)
        if outer_fold < 0
        else architecture_input.fold_ids != outer_fold
    )
    indices = np.flatnonzero(scope_mask)
    labels = architecture_input.labels[indices]
    subtypes = architecture_input.subtypes[indices]
    balanced_weights, subtype_details = subtype_balanced_weights(labels, subtypes)

    descriptors: List[Dict[str, Any]] = []
    for expert in EXPERTS:
        descriptors.append(
            {
                "candidate_id": f"fixed_{expert}",
                "family": "fixed",
                "method_label": EXPERT_LABELS[expert],
                "fixed_expert": expert,
                "calibration": "none",
                "feature_set": "none",
                "q": np.nan,
                "C": np.nan,
                "rho": 0.0,
                "delta_r2l": 0.0,
                "delta_u2r": 0.0,
                "family_order": 0,
                "calibration_order": 0,
                "feature_order": 0,
                "q_order": 0,
            }
        )
    descriptors.append(
        {
            "candidate_id": "fixed_average",
            "family": "fixed",
            "method_label": "Simple probability average",
            "fixed_expert": "average",
            "calibration": "none",
            "feature_set": "none",
            "q": np.nan,
            "C": np.nan,
            "rho": 0.0,
            "delta_r2l": 0.0,
            "delta_u2r": 0.0,
            "family_order": 0,
            "calibration_order": 0,
            "feature_order": 0,
            "q_order": 0,
        }
    )
    nonzero_average_offsets = [pair for pair in OFFSET_PAIRS if pair != (0.0, 0.0)]
    for offset_index, (delta_r2l, delta_u2r) in enumerate(nonzero_average_offsets):
        descriptors.append(
            {
                "candidate_id": f"average_offset_{offset_index:02d}",
                "family": "average_offset",
                "method_label": "Simple average + rare offsets",
                "fixed_expert": "average",
                "calibration": "none",
                "feature_set": "none",
                "q": np.nan,
                "C": np.nan,
                "rho": 0.0,
                "delta_r2l": float(delta_r2l),
                "delta_u2r": float(delta_u2r),
                "family_order": 1,
                "calibration_order": 0,
                "feature_order": 0,
                "q_order": 0,
            }
        )
    descriptor_frame = pd.DataFrame(descriptors)
    candidate_count = len(descriptor_frame)
    seed_count = len(architecture_input.seeds)
    natural = np.empty((seed_count, candidate_count, len(METRICS)), dtype=np.float64)
    balanced_rare = np.empty((seed_count, candidate_count), dtype=np.float64)
    balanced_r2l = np.empty_like(balanced_rare)
    balanced_u2r = np.empty_like(balanced_rare)

    offset_start = len(EXPERTS) + 1
    offset_values = np.asarray(nonzero_average_offsets, dtype=np.float64)
    for seed_index, _seed in enumerate(architecture_input.seeds):
        raw = architecture_input.probabilities[seed_index][:, indices, :].transpose(1, 0, 2)
        average = np.mean(raw, axis=1, dtype=np.float64)
        predictions: List[np.ndarray] = [
            np.argmax(raw[:, expert_index, :], axis=1).astype(np.int64)
            for expert_index in range(len(EXPERTS))
        ]
        predictions.append(np.argmax(average, axis=1).astype(np.int64))
        for candidate_index, values in enumerate(predictions):
            metrics = metric_values_for_predictions(labels, values, balanced_weights)
            natural[seed_index, candidate_index] = metrics[0]
            balanced_rare[seed_index, candidate_index] = metrics[1]
            balanced_r2l[seed_index, candidate_index] = metrics[2]
            balanced_u2r[seed_index, candidate_index] = metrics[3]

        base_scores = np.log(np.clip(average, settings.epsilon, 1.0))
        for start in range(0, len(offset_values), settings.candidate_chunk_size):
            stop = min(start + settings.candidate_chunk_size, len(offset_values))
            offsets = offset_values[start:stop]
            scores = np.broadcast_to(
                base_scores[:, None, :],
                (len(labels), len(offsets), CLASS_COUNT),
            ).copy()
            scores[:, :, R2L_CLASS] += offsets[None, :, 0]
            scores[:, :, U2R_CLASS] += offsets[None, :, 1]
            predicted = np.argmax(scores, axis=2).astype(np.int64)
            natural_metrics = metrics_from_confusions(confusion_batch(labels, predicted))
            balanced_metrics = metrics_from_confusions(
                confusion_batch(labels, predicted, balanced_weights)
            )
            destination = slice(offset_start + start, offset_start + stop)
            natural[seed_index, destination] = natural_metrics
            balanced_rare[seed_index, destination] = balanced_metrics[:, METRIC_INDEX["rare_f1"]]
            balanced_r2l[seed_index, destination] = balanced_metrics[:, METRIC_INDEX["r2l_f1"]]
            balanced_u2r[seed_index, destination] = balanced_metrics[:, METRIC_INDEX["u2r_f1"]]
    return (
        descriptor_frame,
        natural,
        balanced_rare,
        balanced_r2l,
        balanced_u2r,
        subtype_details,
    )


def stack_descriptor_frame(calibration: str, feature_set: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    calibration_index = CALIBRATIONS.index(calibration)
    feature_index = FEATURE_SETS.index(feature_set)
    for q_index, q in enumerate(Q_VALUES):
        for c_index, c_value in enumerate(C_VALUES):
            for rho_index, rho in enumerate(NONZERO_RHO_VALUES):
                for offset_index, (delta_r2l, delta_u2r) in enumerate(OFFSET_PAIRS):
                    candidate_id = (
                        f"stack_c{calibration_index}_f{feature_index}_q{q_index}_"
                        f"C{c_index}_r{rho_index + 1}_o{offset_index:02d}"
                    )
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "family": "stack",
                            "method_label": "Robust Calibrated Super-Stack",
                            "fixed_expert": "none",
                            "calibration": calibration,
                            "feature_set": feature_set,
                            "q": float(q),
                            "C": float(c_value),
                            "rho": float(rho),
                            "delta_r2l": float(delta_r2l),
                            "delta_u2r": float(delta_u2r),
                            "family_order": 2,
                            "calibration_order": calibration_index,
                            "feature_order": feature_index,
                            "q_order": q_index,
                        }
                    )
    return pd.DataFrame(rows)


def aggregate_seed_metrics(
    descriptors: pd.DataFrame,
    natural: np.ndarray,
    balanced_rare: np.ndarray,
    balanced_r2l: np.ndarray,
    balanced_u2r: np.ndarray,
    valid: np.ndarray,
) -> pd.DataFrame:
    if natural.ndim != 3 or natural.shape[2] != len(METRICS):
        raise ValueError(f"Unexpected natural metric shape: {natural.shape}")
    if natural.shape[:2] != balanced_rare.shape or balanced_rare.shape != valid.shape:
        raise ValueError("Candidate metric arrays are not aligned.")
    if natural.shape[1] != len(descriptors):
        raise ValueError("Descriptor and metric candidate counts differ.")
    output = descriptors.copy()
    for metric_index, metric in enumerate(METRICS):
        values = natural[:, :, metric_index]
        output[f"{metric}_mean"] = values.mean(axis=0)
        output[f"{metric}_std"] = values.std(axis=0, ddof=1) if len(values) > 1 else 0.0
    output["balanced_rare_f1_mean"] = balanced_rare.mean(axis=0)
    output["balanced_rare_f1_std"] = (
        balanced_rare.std(axis=0, ddof=1) if len(balanced_rare) > 1 else 0.0
    )
    output["balanced_r2l_f1_mean"] = balanced_r2l.mean(axis=0)
    output["balanced_u2r_f1_mean"] = balanced_u2r.mean(axis=0)
    robust = np.minimum(natural[:, :, METRIC_INDEX["rare_f1"]], balanced_rare)
    rare_floor = np.minimum(
        natural[:, :, METRIC_INDEX["r2l_f1"]],
        natural[:, :, METRIC_INDEX["u2r_f1"]],
    )
    output["robust_rare_f1_mean"] = robust.mean(axis=0)
    output["robust_rare_f1_std"] = robust.std(axis=0, ddof=1) if len(robust) > 1 else 0.0
    output["minimum_rare_f1_mean"] = rare_floor.mean(axis=0)
    output["minimum_rare_f1_std"] = (
        rare_floor.std(axis=0, ddof=1) if len(rare_floor) > 1 else 0.0
    )
    output["valid_all_seeds"] = np.all(valid, axis=0)
    output["valid_seed_count"] = valid.sum(axis=0)
    return output


def build_stage_ranking(
    architecture_input: ArchitectureInput,
    tasks: Sequence[MetaTask],
    outer_fold: int,
    settings: SearchSettings,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    stage = task_stage(outer_fold)
    base = fixed_and_average_offset_seed_metrics(architecture_input, outer_fold, settings)
    descriptor_groups = [base[0]]
    natural_groups = [base[1]]
    balanced_rare_groups = [base[2]]
    balanced_r2l_groups = [base[3]]
    balanced_u2r_groups = [base[4]]
    valid_groups = [np.ones(base[2].shape, dtype=bool)]
    task_lookup = {
        (task.stage, task.seed, task.calibration, task.feature_set): task for task in tasks
    }
    for calibration in CALIBRATIONS:
        for feature_set in FEATURE_SETS:
            descriptors = stack_descriptor_frame(calibration, feature_set)
            seed_natural = []
            seed_balanced_rare = []
            seed_balanced_r2l = []
            seed_balanced_u2r = []
            seed_valid = []
            for seed in architecture_input.seeds:
                task = task_lookup[(stage, seed, calibration, feature_set)]
                arrays = load_task_arrays(task)
                seed_natural.append(arrays["natural_metrics"].reshape(-1, len(METRICS)))
                seed_balanced_rare.append(arrays["balanced_rare_f1"].reshape(-1))
                seed_balanced_r2l.append(arrays["balanced_r2l_f1"].reshape(-1))
                seed_balanced_u2r.append(arrays["balanced_u2r_f1"].reshape(-1))
                valid_q_c = arrays["valid_q_c"].astype(bool)
                expanded = np.broadcast_to(
                    valid_q_c[:, :, None, None],
                    (
                        len(Q_VALUES),
                        len(C_VALUES),
                        len(NONZERO_RHO_VALUES),
                        len(OFFSET_PAIRS),
                    ),
                )
                seed_valid.append(expanded.reshape(-1))
            descriptor_groups.append(descriptors)
            natural_groups.append(np.stack(seed_natural, axis=0))
            balanced_rare_groups.append(np.stack(seed_balanced_rare, axis=0))
            balanced_r2l_groups.append(np.stack(seed_balanced_r2l, axis=0))
            balanced_u2r_groups.append(np.stack(seed_balanced_u2r, axis=0))
            valid_groups.append(np.stack(seed_valid, axis=0))

    descriptors = pd.concat(descriptor_groups, ignore_index=True)
    natural = np.concatenate(natural_groups, axis=1)
    balanced_rare = np.concatenate(balanced_rare_groups, axis=1)
    balanced_r2l = np.concatenate(balanced_r2l_groups, axis=1)
    balanced_u2r = np.concatenate(balanced_u2r_groups, axis=1)
    valid = np.concatenate(valid_groups, axis=1)
    ranking = aggregate_seed_metrics(
        descriptors,
        natural,
        balanced_rare,
        balanced_r2l,
        balanced_u2r,
        valid,
    )
    average_row = ranking[ranking["candidate_id"] == "fixed_average"]
    if len(average_row) != 1:
        raise RuntimeError("Stage ranking does not contain exactly one fixed average.")
    reference = average_row.iloc[0]
    macro_floor = float(reference["macro_f1_mean"]) - settings.macro_guard
    mcc_floor = float(reference["mcc_mean"]) - settings.mcc_guard
    tolerance = 1e-12
    ranking["meets_macro_guard"] = ranking["macro_f1_mean"] >= macro_floor - tolerance
    ranking["meets_mcc_guard"] = ranking["mcc_mean"] >= mcc_floor - tolerance
    ranking["eligible"] = (
        ranking["valid_all_seeds"]
        & ranking["meets_macro_guard"]
        & ranking["meets_mcc_guard"]
    )
    ranking["offset_l1"] = ranking["delta_r2l"].abs() + ranking["delta_u2r"].abs()
    ranking["regularization_tiebreak"] = ranking["C"].fillna(0.0)
    ranking["rho_tiebreak"] = ranking["rho"].fillna(0.0)
    sort_metrics = [
        "robust_rare_f1_mean",
        "rare_f1_mean",
        "minimum_rare_f1_mean",
        "macro_f1_mean",
        "mcc_mean",
        "robust_rare_f1_std",
    ]
    for name in sort_metrics:
        ranking[f"_sort_{name}"] = ranking[name].round(12)
    ranking = ranking.sort_values(
        [
            "eligible",
            "_sort_robust_rare_f1_mean",
            "_sort_rare_f1_mean",
            "_sort_minimum_rare_f1_mean",
            "_sort_macro_f1_mean",
            "_sort_mcc_mean",
            "_sort_robust_rare_f1_std",
            "rho_tiebreak",
            "offset_l1",
            "regularization_tiebreak",
            "family_order",
            "calibration_order",
            "feature_order",
            "q_order",
            "candidate_id",
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
            True,
            True,
            True,
            True,
            True,
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1, dtype=np.int64))
    ranking.insert(1, "stage", stage)
    ranking.insert(2, "outer_fold", outer_fold)
    ranking.insert(3, "architecture", architecture_input.architecture)
    ranking = ranking.drop(columns=[column for column in ranking if column.startswith("_sort_")])
    if not bool(ranking.iloc[0]["eligible"]):
        raise RuntimeError(f"No eligible candidate for {architecture_input.architecture} {stage}.")
    details = {
        "stage": stage,
        "outer_fold": outer_fold,
        "rows": int(np.sum(np.ones(len(architecture_input.labels), dtype=bool) if outer_fold < 0 else architecture_input.fold_ids != outer_fold)),
        "macro_f1_floor": macro_floor,
        "mcc_floor": mcc_floor,
        "simple_average_macro_f1_mean": float(reference["macro_f1_mean"]),
        "simple_average_mcc_mean": float(reference["mcc_mean"]),
        "subtype_balance": base[5],
        "eligible_candidates": int(ranking["eligible"].sum()),
        "invalid_candidates": int((~ranking["valid_all_seeds"]).sum()),
    }
    return ranking, details


def candidate_config(row: Mapping[str, Any]) -> Dict[str, Any]:
    def optional_float(name: str) -> float | None:
        value = row[name]
        return None if pd.isna(value) else float(value)

    return {
        "candidate_id": str(row["candidate_id"]),
        "family": str(row["family"]),
        "method_label": str(row["method_label"]),
        "fixed_expert": str(row["fixed_expert"]),
        "calibration": str(row["calibration"]),
        "feature_set": str(row["feature_set"]),
        "q": optional_float("q"),
        "C": optional_float("C"),
        "rho": float(row["rho"]),
        "delta_r2l": float(row["delta_r2l"]),
        "delta_u2r": float(row["delta_u2r"]),
        "selection_metrics": {
            key: float(row[key])
            for key in (
                "robust_rare_f1_mean",
                "robust_rare_f1_std",
                "rare_f1_mean",
                "rare_f1_std",
                "balanced_rare_f1_mean",
                "minimum_rare_f1_mean",
                "macro_f1_mean",
                "mcc_mean",
            )
        },
    }


def fit_selected_candidate(
    architecture_input: ArchitectureInput,
    seed: int,
    config: Mapping[str, Any],
    train_indices: np.ndarray,
    prediction_indices: np.ndarray,
    settings: SearchSettings,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    seed_index = architecture_input.seeds.index(int(seed))
    raw = np.asarray(architecture_input.probabilities[seed_index], dtype=np.float64)
    family = str(config["family"])
    prediction_indices = np.asarray(prediction_indices, dtype=np.int64)
    raw_prediction = raw[:, prediction_indices, :].transpose(1, 0, 2)
    average_prediction = np.mean(raw_prediction, axis=1, dtype=np.float64)
    empty_state = {
        "temperatures": np.empty((0,), dtype=np.float64),
        "temperature_boundary": np.empty((0,), dtype=bool),
        "feature_mean": np.empty((0,), dtype=np.float64),
        "feature_scale": np.empty((0,), dtype=np.float64),
        "classes": np.empty((0,), dtype=np.int64),
        "coef": np.empty((0, 0), dtype=np.float64),
        "intercept": np.empty((0,), dtype=np.float64),
        "n_iter": np.empty((0,), dtype=np.int64),
    }
    if family == "fixed":
        fixed_expert = str(config["fixed_expert"])
        if fixed_expert == "average":
            probabilities = average_prediction
        else:
            probabilities = raw_prediction[:, EXPERTS.index(fixed_expert), :]
        predictions = np.argmax(probabilities, axis=1).astype(np.int64)
        return predictions, probabilities, empty_state
    if family == "average_offset":
        decision_probabilities = offset_decision_probabilities(
            average_prediction,
            float(config["delta_r2l"]),
            float(config["delta_u2r"]),
            settings.epsilon,
        )
        predictions = np.argmax(decision_probabilities, axis=1).astype(np.int64)
        return predictions, decision_probabilities, empty_state
    if family != "stack":
        raise ValueError(f"Unknown selected family: {family}")

    train_indices = np.asarray(train_indices, dtype=np.int64)
    active_train = raw[:, train_indices, :].transpose(1, 0, 2)
    active_prediction = raw_prediction.copy()
    temperatures = np.ones(len(EXPERTS), dtype=np.float64)
    temperature_boundary = np.zeros(len(EXPERTS), dtype=bool)
    if config["calibration"] == "temperature":
        active_train = active_train.copy()
        for expert_index in range(len(EXPERTS)):
            temperature, boundary = fit_temperature(
                active_train[:, expert_index, :],
                architecture_input.labels[train_indices],
                settings,
            )
            temperatures[expert_index] = temperature
            temperature_boundary[expert_index] = boundary
            active_train[:, expert_index, :] = temperature_scale(
                active_train[:, expert_index, :], temperature, settings.epsilon
            )
            if len(prediction_indices):
                active_prediction[:, expert_index, :] = temperature_scale(
                    active_prediction[:, expert_index, :], temperature, settings.epsilon
                )
    elif config["calibration"] != "raw":
        raise ValueError(f"Unknown selected calibration: {config['calibration']}")

    train_features = build_features(active_train, str(config["feature_set"]), settings.epsilon)
    feature_mean, feature_scale = fit_standardizer(train_features)
    train_features = apply_standardizer(train_features, feature_mean, feature_scale)
    sample_weights = normalized_sample_weights(
        architecture_input.labels[train_indices], float(config["q"])
    )
    model = LogisticRegression(
        C=float(config["C"]),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=True,
        tol=settings.tolerance,
        max_iter=settings.max_iter,
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_features, architecture_input.labels[train_indices], sample_weight=sample_weights)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(f"Selected candidate failed to converge during refit: {config['candidate_id']}")
    if not np.array_equal(model.classes_, np.arange(CLASS_COUNT)):
        raise RuntimeError(f"Unexpected refit classes: {model.classes_.tolist()}")
    if len(prediction_indices):
        prediction_features = build_features(
            active_prediction, str(config["feature_set"]), settings.epsilon
        )
        prediction_features = apply_standardizer(
            prediction_features, feature_mean, feature_scale
        )
        stack_probabilities = model.predict_proba(prediction_features)
        blended = (
            (1.0 - float(config["rho"])) * average_prediction
            + float(config["rho"]) * stack_probabilities
        )
        decision_probabilities = offset_decision_probabilities(
            blended,
            float(config["delta_r2l"]),
            float(config["delta_u2r"]),
            settings.epsilon,
        )
        predictions = np.argmax(decision_probabilities, axis=1).astype(np.int64)
    else:
        decision_probabilities = np.empty((0, CLASS_COUNT), dtype=np.float64)
        predictions = np.empty((0,), dtype=np.int64)
    state = {
        "temperatures": temperatures,
        "temperature_boundary": temperature_boundary,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "classes": np.asarray(model.classes_, dtype=np.int64),
        "coef": np.asarray(model.coef_, dtype=np.float64),
        "intercept": np.asarray(model.intercept_, dtype=np.float64),
        "n_iter": np.asarray(model.n_iter_, dtype=np.int64),
    }
    for key in ("temperatures", "feature_mean", "feature_scale", "coef", "intercept"):
        if not np.isfinite(state[key]).all():
            raise RuntimeError(f"Selected refit produced non-finite {key}.")
    expected_features = len(feature_names(str(config["feature_set"])))
    if state["feature_mean"].shape != (expected_features,):
        raise RuntimeError("Selected refit feature-mean shape is invalid.")
    if state["feature_scale"].shape != (expected_features,):
        raise RuntimeError("Selected refit feature-scale shape is invalid.")
    if state["coef"].shape != (CLASS_COUNT, expected_features):
        raise RuntimeError("Selected refit coefficient shape is invalid.")
    if state["intercept"].shape != (CLASS_COUNT,):
        raise RuntimeError("Selected refit intercept shape is invalid.")
    if len(prediction_indices) and not np.array_equal(
        predictions, np.argmax(decision_probabilities, axis=1)
    ):
        raise RuntimeError("Saved decision probabilities do not reproduce predictions.")
    return predictions, decision_probabilities, state


def run_search_tasks(
    architecture_input: ArchitectureInput,
    tasks: Sequence[MetaTask],
    settings: SearchSettings,
    workers: int,
    threads_per_worker: int,
    resume: bool,
) -> pd.DataFrame:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "architecture": architecture_input.architecture,
        "seeds": architecture_input.seeds,
        "probabilities": architecture_input.probabilities,
        "labels": architecture_input.labels,
        "fold_ids": architecture_input.fold_ids,
        "subtypes": architecture_input.subtypes,
        "settings": settings,
    }
    for key in ("probabilities", "labels", "fold_ids", "subtypes"):
        _WORKER_CONTEXT[key].setflags(write=False)
    completed_rows: List[Dict[str, Any]] = []
    pending: List[MetaTask] = []
    for task in tasks:
        if resume and _task_cache_valid(task):
            arrays = load_task_arrays(task)
            completed_rows.append(
                {
                    **asdict(task),
                    "status": "resumed",
                    "runtime_seconds": float(arrays["runtime_seconds"].item()),
                    "valid_configurations": int(arrays["valid_q_c"].sum()),
                    "total_configurations": int(arrays["valid_q_c"].size),
                    "temperature_boundaries": int(arrays["temperature_boundary"].sum()),
                }
            )
        else:
            pending.append(task)
    print(
        f"{ARCHITECTURE_LABELS[architecture_input.architecture]} meta tasks: "
        f"{len(tasks)} total, {len(completed_rows)} resumed, {len(pending)} pending",
        flush=True,
    )
    if pending:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "This CPU runner requires the POSIX fork start method so the validated "
                "OOF arrays can be shared read-only without copying them into every worker."
            )
        process_context = mp.get_context("fork")
        worker_count = min(max(1, workers), len(pending))
        started = time.perf_counter()
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=process_context,
            initializer=_worker_initializer,
            initargs=(threads_per_worker,),
        )
        pending_iterator = iter(pending)
        in_flight: Dict[Any, MetaTask] = {}
        submission_limit = min(len(pending), 2 * worker_count)

        def submit_until_full() -> None:
            while len(in_flight) < submission_limit:
                try:
                    next_task = next(pending_iterator)
                except StopIteration:
                    break
                in_flight[executor.submit(_run_meta_task, next_task)] = next_task

        completed_index = 0
        try:
            submit_until_full()
            while in_flight:
                finished, _unfinished = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in finished:
                    task = in_flight.pop(future)
                    result = future.result()
                    completed_index += 1
                    completed_rows.append(
                        {
                            **result["task"],
                            "status": "computed",
                            "runtime_seconds": result["runtime_seconds"],
                            "valid_configurations": result["valid_configurations"],
                            "total_configurations": result["total_configurations"],
                            "temperature_boundaries": result["temperature_boundaries"],
                        }
                    )
                    elapsed = time.perf_counter() - started
                    rate = completed_index / elapsed if elapsed > 0 else 0.0
                    remaining = (
                        (len(pending) - completed_index) / rate if rate > 0 else float("nan")
                    )
                    print(
                        f"  [{completed_index:>3}/{len(pending)}] {task.stage} seed={task.seed} "
                        f"{task.calibration}/{task.feature_set}; task={result['runtime_seconds']:.1f}s; "
                        f"ETA={remaining / 60.0:.1f} min",
                        flush=True,
                    )
                submit_until_full()
        except BaseException:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    for task in tasks:
        if not _task_cache_valid(task):
            raise RuntimeError(f"Task did not produce a valid cache: {task.cache_path}")
    frame = pd.DataFrame(completed_rows)
    return frame.sort_values(
        ["stage", "seed", "calibration", "feature_set"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_method_runs(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (architecture, method, method_label), group in per_seed.groupby(
        ["architecture", "method", "method_label"], sort=False
    ):
        row: Dict[str, Any] = {
            "architecture": architecture,
            "model": ARCHITECTURE_LABELS[str(architecture)],
            "method": method,
            "method_label": method_label,
            "runs": int(len(group)),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"])),
        }
        metric_names = [*METRICS, "balanced_rare_f1", "balanced_r2l_f1", "balanced_u2r_f1", "robust_rare_f1"]
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_architecture(
    architecture_input: ArchitectureInput,
    tasks: Sequence[MetaTask],
    settings: SearchSettings,
    stem: str,
    results_dir: Path,
    top_n: int,
    experiment_key: str,
    protocol_path: Path,
    train_path: Path,
) -> Dict[str, Any]:
    architecture = architecture_input.architecture
    architecture_stem = f"{stem}_{architecture}"
    rankings: Dict[str, pd.DataFrame] = {}
    stage_details: Dict[str, Any] = {}
    selections: Dict[str, Dict[str, Any]] = {}
    ranking_paths: Dict[str, str] = {}
    top_paths: Dict[str, str] = {}
    for outer_fold in (*FOLD_IDS, -1):
        stage = task_stage(outer_fold)
        print(f"Building {ARCHITECTURE_LABELS[architecture]} {stage} ranking...", flush=True)
        ranking, details = build_stage_ranking(
            architecture_input, tasks, outer_fold, settings
        )
        if len(ranking) != CANONICAL_LIBRARY_COUNT:
            raise RuntimeError(f"Expected 68,124 canonical candidates, got {len(ranking)}.")
        rankings[stage] = ranking
        stage_details[stage] = details
        selections[stage] = candidate_config(ranking.iloc[0])
        ranking_path = results_dir / f"{architecture_stem}_{stage}_ranking.csv.gz"
        top_path = results_dir / f"{architecture_stem}_{stage}_top_{top_n}.csv"
        atomic_csv_gzip(ranking_path, ranking)
        core.atomic_csv(top_path, ranking.head(top_n))
        ranking_paths[stage] = str(ranking_path)
        top_paths[stage] = str(top_path)
        winner = selections[stage]
        print(
            f"  selected {winner['candidate_id']}: robust Rare F1="
            f"{winner['selection_metrics']['robust_rare_f1_mean']:.4f}, "
            f"natural Rare F1={winner['selection_metrics']['rare_f1_mean']:.4f}",
            flush=True,
        )

    row_count = len(architecture_input.labels)
    seed_count = len(architecture_input.seeds)
    nested_predictions = np.full((seed_count, row_count), -1, dtype=np.int64)
    nested_decision_probabilities = np.full(
        (seed_count, row_count, CLASS_COUNT), np.nan, dtype=np.float64
    )
    outer_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    for outer_fold in FOLD_IDS:
        stage = task_stage(outer_fold)
        config = selections[stage]
        selection_rows.append(
            {
                "architecture": architecture,
                "outer_fold": outer_fold,
                **{key: value for key, value in config.items() if key != "selection_metrics"},
                **config["selection_metrics"],
                "tuning_rows": stage_details[stage]["rows"],
                "eligible_candidates": stage_details[stage]["eligible_candidates"],
                "invalid_candidates": stage_details[stage]["invalid_candidates"],
            }
        )
        train_indices = np.flatnonzero(architecture_input.fold_ids != outer_fold)
        validation_indices = np.flatnonzero(architecture_input.fold_ids == outer_fold)
        validation_labels = architecture_input.labels[validation_indices]
        validation_subtypes = architecture_input.subtypes[validation_indices]
        validation_weights, validation_subtype_details = subtype_balanced_weights(
            validation_labels, validation_subtypes
        )
        for seed_index, seed in enumerate(architecture_input.seeds):
            predicted, probabilities, _state = fit_selected_candidate(
                architecture_input,
                seed,
                config,
                train_indices,
                validation_indices,
                settings,
            )
            nested_predictions[seed_index, validation_indices] = predicted
            if not np.array_equal(predicted, np.argmax(probabilities, axis=1)):
                raise RuntimeError("Outer decision probabilities do not reproduce predictions.")
            nested_decision_probabilities[seed_index, validation_indices] = probabilities
            metrics = metric_values_for_predictions(
                validation_labels, predicted, validation_weights
            )
            row = {
                "architecture": architecture,
                "outer_fold": outer_fold,
                "seed": int(seed),
                "candidate_id": config["candidate_id"],
                "family": config["family"],
                **{metric: float(metrics[0][index]) for index, metric in enumerate(METRICS)},
                "balanced_rare_f1": metrics[1],
                "balanced_r2l_f1": metrics[2],
                "balanced_u2r_f1": metrics[3],
                "robust_rare_f1": min(float(metrics[0][METRIC_INDEX["rare_f1"]]), metrics[1]),
                "subtype_balance": json.dumps(validation_subtype_details, sort_keys=True),
            }
            outer_rows.append(row)
    if np.any(nested_predictions < 0) or not np.isfinite(
        nested_decision_probabilities
    ).all():
        raise RuntimeError(f"Nested outer predictions are incomplete for {architecture}.")

    global_weights, global_subtype_details = subtype_balanced_weights(
        architecture_input.labels, architecture_input.subtypes
    )
    per_seed_rows: List[Dict[str, Any]] = []
    method_specs = [
        ("general", "General (cross-entropy)"),
        ("focal", "Focal"),
        ("batching", "Minority batching"),
        ("simple_average", "Simple probability average"),
        ("nested_selected_fusion", "Nested selected fusion procedure"),
    ]
    for seed_index, seed in enumerate(architecture_input.seeds):
        raw = architecture_input.probabilities[seed_index].transpose(1, 0, 2)
        raw_predictions = {
            "general": np.argmax(raw[:, 0, :], axis=1),
            "focal": np.argmax(raw[:, 1, :], axis=1),
            "batching": np.argmax(raw[:, 2, :], axis=1),
            "simple_average": np.argmax(np.mean(raw, axis=1), axis=1),
            "nested_selected_fusion": nested_predictions[seed_index],
        }
        for method, method_label in method_specs:
            metrics = metric_values_for_predictions(
                architecture_input.labels,
                raw_predictions[method],
                global_weights,
            )
            per_seed_rows.append(
                {
                    "architecture": architecture,
                    "method": method,
                    "method_label": method_label,
                    "seed": int(seed),
                    **{metric: float(metrics[0][index]) for index, metric in enumerate(METRICS)},
                    "balanced_rare_f1": metrics[1],
                    "balanced_r2l_f1": metrics[2],
                    "balanced_u2r_f1": metrics[3],
                    "robust_rare_f1": min(
                        float(metrics[0][METRIC_INDEX["rare_f1"]]), metrics[1]
                    ),
                }
            )
    per_seed_frame = pd.DataFrame(per_seed_rows)
    summary = summarize_method_runs(per_seed_frame)
    method_order = {method: index for index, (method, _label) in enumerate(method_specs)}
    summary["_order"] = summary["method"].map(method_order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    selected_by_fold = np.asarray(
        [selections[task_stage(fold)]["candidate_id"] for fold in FOLD_IDS], dtype="U96"
    )
    nested_predictions_path = results_dir / f"{architecture_stem}_nested_predictions.npz"
    atomic_npz(
        nested_predictions_path,
        row_indices=np.arange(row_count, dtype=np.int64),
        fold_ids=architecture_input.fold_ids,
        labels=architecture_input.labels,
        seeds=np.asarray(architecture_input.seeds, dtype=np.int64),
        predictions=nested_predictions,
        decision_probabilities=nested_decision_probabilities,
        selected_candidate_by_fold=selected_by_fold,
    )
    outer_runs_path = results_dir / f"{architecture_stem}_outer_fold_per_seed.csv"
    selections_path = results_dir / f"{architecture_stem}_outer_selected.csv"
    per_seed_path = results_dir / f"{architecture_stem}_nested_per_seed.csv"
    summary_path = results_dir / f"{architecture_stem}_nested_summary.csv"
    core.atomic_csv(outer_runs_path, pd.DataFrame(outer_rows))
    core.atomic_csv(selections_path, pd.DataFrame(selection_rows))
    core.atomic_csv(per_seed_path, per_seed_frame)
    core.atomic_csv(summary_path, summary)

    final_config = selections["final_cv"]
    all_indices = np.arange(row_count, dtype=np.int64)
    final_cv_predictions = np.full((seed_count, row_count), -1, dtype=np.int64)
    final_cv_probabilities = np.full(
        (seed_count, row_count, CLASS_COUNT), np.nan, dtype=np.float64
    )
    final_cv_rows: List[Dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for seed_index, seed in enumerate(architecture_input.seeds):
            for validation_fold in FOLD_IDS:
                train_indices = np.flatnonzero(
                    architecture_input.fold_ids != validation_fold
                )
                validation_indices = np.flatnonzero(
                    architecture_input.fold_ids == validation_fold
                )
                predicted, probabilities, _state = fit_selected_candidate(
                    architecture_input,
                    seed,
                    final_config,
                    train_indices,
                    validation_indices,
                    settings,
                )
                final_cv_predictions[seed_index, validation_indices] = predicted
                final_cv_probabilities[seed_index, validation_indices] = probabilities
            if np.any(final_cv_predictions[seed_index] < 0) or not np.isfinite(
                final_cv_probabilities[seed_index]
            ).all():
                raise RuntimeError("Final selected cross-fitted predictions are incomplete.")
            if not np.array_equal(
                final_cv_predictions[seed_index],
                np.argmax(final_cv_probabilities[seed_index], axis=1),
            ):
                raise RuntimeError("Final CV decision probabilities do not reproduce predictions.")
            metrics = metric_values_for_predictions(
                architecture_input.labels,
                final_cv_predictions[seed_index],
                global_weights,
            )
            final_cv_rows.append(
                {
                    "architecture": architecture,
                    "seed": int(seed),
                    "candidate_id": final_config["candidate_id"],
                    **{
                        metric: float(metrics[0][metric_index])
                        for metric_index, metric in enumerate(METRICS)
                    },
                    "balanced_rare_f1": metrics[1],
                    "balanced_r2l_f1": metrics[2],
                    "balanced_u2r_f1": metrics[3],
                    "robust_rare_f1": min(
                        float(metrics[0][METRIC_INDEX["rare_f1"]]), metrics[1]
                    ),
                }
            )
    final_cv_frame = pd.DataFrame(final_cv_rows)
    audit_means = {
        "rare_f1_mean": float(final_cv_frame["rare_f1"].mean()),
        "balanced_rare_f1_mean": float(final_cv_frame["balanced_rare_f1"].mean()),
        "robust_rare_f1_mean": float(final_cv_frame["robust_rare_f1"].mean()),
        "macro_f1_mean": float(final_cv_frame["macro_f1"].mean()),
        "mcc_mean": float(final_cv_frame["mcc"].mean()),
    }
    for metric, observed in audit_means.items():
        expected = float(final_config["selection_metrics"][metric])
        if not np.isclose(observed, expected, atol=1e-12, rtol=0.0):
            raise RuntimeError(
                f"Final selected prediction audit disagrees for {metric}: "
                f"ranking={expected}, reconstructed={observed}."
            )
    final_cv_predictions_path = (
        results_dir / f"{architecture_stem}_final_cv_selected_predictions.npz"
    )
    atomic_npz(
        final_cv_predictions_path,
        row_indices=np.arange(row_count, dtype=np.int64),
        fold_ids=architecture_input.fold_ids,
        labels=architecture_input.labels,
        seeds=np.asarray(architecture_input.seeds, dtype=np.int64),
        predictions=final_cv_predictions,
        decision_probabilities=final_cv_probabilities,
        candidate_id=np.asarray(final_config["candidate_id"]),
    )
    final_cv_per_seed_path = (
        results_dir / f"{architecture_stem}_final_cv_selected_per_seed.csv"
    )
    core.atomic_csv(final_cv_per_seed_path, final_cv_frame)

    model_payload: Dict[str, np.ndarray] = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "seeds": np.asarray(architecture_input.seeds, dtype=np.int64),
        "candidate_id": np.asarray(final_config["candidate_id"]),
        "family": np.asarray(final_config["family"]),
        "calibration": np.asarray(final_config["calibration"]),
        "feature_set": np.asarray(final_config["feature_set"]),
        "q": np.asarray(
            np.nan if final_config["q"] is None else final_config["q"], dtype=np.float64
        ),
        "C": np.asarray(
            np.nan if final_config["C"] is None else final_config["C"], dtype=np.float64
        ),
        "rho": np.asarray(final_config["rho"], dtype=np.float64),
        "delta_r2l": np.asarray(final_config["delta_r2l"], dtype=np.float64),
        "delta_u2r": np.asarray(final_config["delta_u2r"], dtype=np.float64),
        "epsilon": np.asarray(settings.epsilon, dtype=np.float64),
        "feature_names": np.asarray(
            feature_names(final_config["feature_set"])
            if final_config["family"] == "stack"
            else [],
            dtype="U96",
        ),
        "experiment_key": np.asarray(experiment_key),
        "protocol_sha256": np.asarray(core.sha256_file(protocol_path)),
    }
    refit_rows: List[Dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for seed in architecture_input.seeds:
            _predicted, _probabilities, state = fit_selected_candidate(
                architecture_input,
                seed,
                final_config,
                all_indices,
                np.empty((0,), dtype=np.int64),
                settings,
            )
            prefix = f"seed_{seed}"
            for key, values in state.items():
                model_payload[f"{prefix}_{key}"] = np.asarray(values)
            refit_rows.append(
                {
                    "architecture": architecture,
                    "seed": int(seed),
                    "candidate_id": final_config["candidate_id"],
                    "family": final_config["family"],
                    "temperature_general": (
                        float(state["temperatures"][0]) if len(state["temperatures"]) else np.nan
                    ),
                    "temperature_focal": (
                        float(state["temperatures"][1]) if len(state["temperatures"]) else np.nan
                    ),
                    "temperature_batching": (
                        float(state["temperatures"][2]) if len(state["temperatures"]) else np.nan
                    ),
                    "feature_count": int(len(state["feature_mean"])),
                    "temperature_boundary_general": (
                        bool(state["temperature_boundary"][0])
                        if len(state["temperature_boundary"])
                        else False
                    ),
                    "temperature_boundary_focal": (
                        bool(state["temperature_boundary"][1])
                        if len(state["temperature_boundary"])
                        else False
                    ),
                    "temperature_boundary_batching": (
                        bool(state["temperature_boundary"][2])
                        if len(state["temperature_boundary"])
                        else False
                    ),
                    "max_iterations": (
                        int(np.max(state["n_iter"])) if len(state["n_iter"]) else 0
                    ),
                }
            )
    model_path = results_dir / f"{architecture_stem}_final_seed_models.npz"
    atomic_npz(model_path, **model_payload)
    refit_path = results_dir / f"{architecture_stem}_final_refits.csv"
    core.atomic_csv(refit_path, pd.DataFrame(refit_rows))

    final_config_path = results_dir / f"{architecture_stem}_best_config.json"
    final_record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "architecture": architecture,
        "model": ARCHITECTURE_LABELS[architecture],
        "selection_stage": "four-fold all-OOF meta-CV after nested procedure evaluation",
        "config": final_config,
        "feature_names": (
            feature_names(final_config["feature_set"])
            if final_config["family"] == "stack"
            else []
        ),
        "model_state_npz": str(model_path),
        "model_state_sha256": core.sha256_file(model_path),
        "protocol": str(protocol_path),
        "protocol_sha256": core.sha256_file(protocol_path),
        "kddtrain_sha256": core.sha256_file(train_path),
        "settings": asdict(settings),
        "final_cv_ranking": ranking_paths["final_cv"],
        "final_cv_ranking_sha256": core.sha256_file(Path(ranking_paths["final_cv"])),
        "final_cv_selected_predictions": str(final_cv_predictions_path),
        "final_cv_selected_predictions_sha256": core.sha256_file(
            final_cv_predictions_path
        ),
        "final_cv_selected_per_seed": str(final_cv_per_seed_path),
        "source_hashes": architecture_input.source_hashes,
        "source_pointer_hashes": architecture_input.pointer_hashes,
        "source_protocol_hashes": architecture_input.protocol_hashes,
        "kddtest_accessed": False,
        "test_status": "not accessed; configuration frozen from KDDTrain+ OOF only",
    }
    core.atomic_json(final_config_path, final_record)

    stage_details_path = results_dir / f"{architecture_stem}_stage_details.json"
    core.atomic_json(
        stage_details_path,
        {
            "architecture": architecture,
            "stages": stage_details,
            "global_subtype_balance": global_subtype_details,
            "outer_selections": selections,
        },
    )
    return {
        "architecture": architecture,
        "nested_summary": str(summary_path),
        "nested_per_seed": str(per_seed_path),
        "outer_fold_per_seed": str(outer_runs_path),
        "outer_selected": str(selections_path),
        "nested_predictions": str(nested_predictions_path),
        "nested_predictions_sha256": core.sha256_file(nested_predictions_path),
        "rankings": ranking_paths,
        "top_rankings": top_paths,
        "stage_details": str(stage_details_path),
        "best_config": str(final_config_path),
        "final_seed_models": str(model_path),
        "final_refits": str(refit_path),
        "final_cv_selected_predictions": str(final_cv_predictions_path),
        "final_cv_selected_per_seed": str(final_cv_per_seed_path),
        "summary_frame": summary,
        "per_seed_frame": per_seed_frame,
        "final_config": final_config,
    }


def experiment_definition(
    script_path: Path,
    train_path: Path,
    architecture_inputs: Sequence[ArchitectureInput],
    settings: SearchSettings,
    threads_per_worker: int,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "script_sha256": core.sha256_file(script_path),
        "numerical_runtime_identity": {
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "scikit_learn": package_version("scikit-learn"),
            "threadpoolctl": package_version("threadpoolctl"),
        },
        "kddtrain_sha256": core.sha256_file(train_path),
        "architectures": [item.architecture for item in architecture_inputs],
        "seeds": list(architecture_inputs[0].seeds),
        "fold_ids": list(FOLD_IDS),
        "experts": list(EXPERTS),
        "source_hashes": {
            item.architecture: item.source_hashes for item in architecture_inputs
        },
        "pointer_hashes": {
            item.architecture: item.pointer_hashes for item in architecture_inputs
        },
        "protocol_hashes": {
            item.architecture: item.protocol_hashes for item in architecture_inputs
        },
        "settings": asdict(settings),
        "threads_per_worker": int(threads_per_worker),
        "grid": {
            "calibrations": list(CALIBRATIONS),
            "feature_sets": list(FEATURE_SETS),
            "q_values": list(Q_VALUES),
            "C_values": list(C_VALUES),
            "rho_values": list(RHO_VALUES),
            "delta_values": list(DELTA_VALUES),
        },
        "semantic_choices": {
            "average_anchor": "original raw arithmetic mean of G/F/B probabilities",
            "seed_handling": "separate model per seed; one shared candidate selected by seed mean",
            "robust_seed_score": "min(natural Rare F1, subtype-balanced Rare F1) per seed",
            "seed_stability_tiebreak": "sample SD of per-seed robust Rare F1",
            "class_weight_normalization": "sample weights normalized to arithmetic mean one",
            "standardization": "training-split mean and population SD; zero scales replaced by one",
            "temperature_parameterization": "one positive scalar per expert via bounded log-temperature NLL minimization",
        },
    }


def build_protocol(
    definition: Mapping[str, Any],
    experiment_key: str,
    architecture_inputs: Sequence[ArchitectureInput],
) -> Dict[str, Any]:
    feature_schema = {feature_set: feature_names(feature_set) for feature_set in FEATURE_SETS}
    return {
        "schema_version": SCHEMA_VERSION,
        "title": "Robust Calibrated Super-Stack all-architecture OOF search",
        "experiment_key": experiment_key,
        "definition": definition,
        "feature_schema": feature_schema,
        "feature_definitions": {
            "F0": "15 expert-major natural-log probabilities",
            "F1_additions": (
                "3 natural-log entropies, 3 probability top1-top2 margins, "
                "6 rare-vs-maximum-majority log margins, 3 pairwise Jensen-Shannon divergences"
            ),
            "F2_additions": "six within-rare-class pairwise products of expert rare margins",
        },
        "subtype_balanced_view": {
            "definition": (
                "For each scored partition and rare class r, an example of observed subtype t "
                "gets n_r/(K_r*n_r_t); majority true examples get weight one. Weighted one-vs-rest "
                "R2L and U2R F1 are averaged."
            ),
            "natural_metric_guards_only": True,
        },
        "selection": {
            "guard_macro_f1": (
                "candidate natural mean >= raw AVG mean - "
                f"{float(definition['settings']['macro_guard']):g} absolute"
            ),
            "guard_mcc": (
                "candidate natural mean >= raw AVG mean - "
                f"{float(definition['settings']['mcc_guard']):g} absolute"
            ),
            "ranking": [
                "robust Rare F1 mean descending",
                "natural Rare F1 mean descending",
                "mean per-seed min(R2L F1,U2R F1) descending",
                "natural Macro-F1 mean descending",
                "natural MCC mean descending",
                "robust Rare F1 sample SD ascending",
                "rho ascending",
                "absolute rare-offset sum ascending",
                "C ascending (stronger L2 regularization)",
                "fixed/lower-complexity family, raw calibration, smaller feature set, smaller q, candidate ID",
            ],
            "comparison_rounding_decimals": 12,
        },
        "nested_meta_protocol": {
            "outer_folds": 4,
            "inner_folds_per_outer": 3,
            "inner_training_original_folds": 2,
            "pool_inner_predictions_before_metrics": True,
            "outer_winner_refit": "separately per seed on all three outer-training folds",
            "final_selection": "fresh four-fold all-OOF meta-CV, then one all-OOF refit per seed",
            "claim_scope": (
                "Nested meta-level CV conditional on frozen cross-fitted expert probabilities; "
                "not fully nested end-to-end because base experts and focal settings are not retrained "
                "inside meta outer folds."
            ),
            "safe_stack_comparison_policy": (
                "Existing SAFE-Stack validation winners were selected non-nested on all OOF "
                "rows and are not a direct nested comparator. A superiority claim requires "
                "retuning SAFE-Stack inside the same inner folds."
            ),
        },
        "candidate_counts_per_architecture": {
            "trained_hyperparameter_combinations": 210,
            "postprocessing_per_trained_combination_nominal": 405,
            "stack_family_nominal": 85050,
            "fixed_candidates_nominal": 4,
            "total_nominal_library_entries": 85054,
            "canonical_stack_and_average_offset_rules": 68121,
            "canonical_unique_library_rules_including_G_F_B": 68124,
            "reason_for_deduplication": (
                "rho=0 ignores calibration/features/q/C; exact raw AVG also duplicates the fixed AVG"
            ),
        },
        "fit_counts_per_architecture": {
            "nested_inner_logistic_fits": 7560,
            "final_selection_logistic_fits": 2520,
            "outer_selected_refits_maximum": 12,
            "final_selected_cv_audit_refits_maximum": 12,
            "final_seed_refits_maximum": 3,
            "total_maximum_including_prediction_audit": 10107,
        },
        "runtime": {
            "numerical_environment_hashed_in_definition": definition[
                "numerical_runtime_identity"
            ],
            "operational_worker_count_excluded_from_scientific_identity": True,
        },
        "sources": {
            item.architecture: {
                "pointer_paths": item.pointer_paths,
                "pointer_hashes": item.pointer_hashes,
                "protocol_paths": item.protocol_paths,
                "protocol_hashes": item.protocol_hashes,
                "oof_paths": item.source_paths,
                "oof_hashes": item.source_hashes,
                "focal_config_id": str(item.focal_best.get("config_id", "")),
            }
            for item in architecture_inputs
        },
        "data_access": {
            "KDDTrain+": "read for subtype identity and verified against OOF labels",
            "KDDTest+": "NEVER ACCESSED",
            "synthetic_data": "not accessed",
        },
        "kddtest_accessed": False,
    }


def print_preflight(
    architecture_inputs: Sequence[ArchitectureInput],
    workers: int,
    threads_per_worker: int,
) -> None:
    print("Robust Calibrated Super-Stack all-architecture search")
    print(f"Architectures: {[item.architecture for item in architecture_inputs]}")
    print(f"Seeds: {list(architecture_inputs[0].seeds)}; folds: {list(FOLD_IDS)}")
    print(
        f"CPU: {available_logical_cpus()} logical available, "
        f"{detect_physical_cpus()} physical detected; workers={workers}; "
        f"threads/worker={threads_per_worker}"
    )
    print("GPU use: NO (saved-probability CPU meta-search)")
    print("Trained stacker hyperparameters: 210")
    print("Nominal library entries/architecture: 85,054")
    print("Canonical unique rules evaluated/architecture: 68,124")
    print(
        "Logistic fits/architecture: 7,560 nested + 2,520 final-CV + "
        "<=27 selected/audit/final refits"
    )
    for item in architecture_inputs:
        counts = np.bincount(item.labels, minlength=CLASS_COUNT).tolist()
        folds = {fold: int(np.sum(item.fold_ids == fold)) for fold in FOLD_IDS}
        _weights, details = subtype_balanced_weights(item.labels, item.subtypes)
        print(
            f"  {ARCHITECTURE_LABELS[item.architecture]}: rows={len(item.labels):,}; "
            f"classes={counts}; folds={folds}; "
            f"R2L subtypes={len(details['R2L']['observed_subtypes'])}; "
            f"U2R subtypes={len(details['U2R']['observed_subtypes'])}"
        )
    print("Validation claim: nested meta-level CV; base experts remain frozen")
    print("KDDTest+ accessed: NO")


def run_benchmark(
    architecture_input: ArchitectureInput,
    settings: SearchSettings,
    workers: int,
    threads_per_worker: int,
) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "architecture": architecture_input.architecture,
        "seeds": architecture_input.seeds,
        "probabilities": architecture_input.probabilities,
        "labels": architecture_input.labels,
        "fold_ids": architecture_input.fold_ids,
        "subtypes": architecture_input.subtypes,
        "settings": settings,
    }
    print(
        "Benchmarking one conservative task: outer 0, seed 0, temperature calibration, F2; "
        "105 logistic fits plus 11,340 canonical post-processing rules."
    )
    with tempfile.TemporaryDirectory(prefix="robust_super_stack_benchmark_") as directory:
        identity = {
            "benchmark": True,
            "architecture": architecture_input.architecture,
            "settings": asdict(settings),
        }
        task = MetaTask(
            architecture=architecture_input.architecture,
            stage="outer_0",
            outer_fold=0,
            seed=architecture_input.seeds[0],
            calibration="temperature",
            feature_set="F2",
            fingerprint=stable_hash(identity, 24),
            cache_path=str(Path(directory) / "benchmark.npz"),
        )
        _worker_initializer(threads_per_worker)
        result = _run_meta_task(task)
        task_seconds = float(result["runtime_seconds"])
    nested_serial = 72.0 * task_seconds
    full_serial = nested_serial + 18.0 * task_seconds * (4.0 / 3.0)
    efficiency = 0.65
    parallel_nested = nested_serial / (max(1, workers) * efficiency)
    parallel_full = full_serial / (max(1, workers) * efficiency)
    print(f"Benchmark task wall time: {task_seconds:.1f} seconds")
    print(f"Conservative serial projection: nested={nested_serial / 60:.1f} min; full={full_serial / 60:.1f} min")
    print(
        f"Approximate {workers}-worker projection at 65% efficiency: "
        f"nested={parallel_nested / 60:.1f} min; nested+final={parallel_full / 60:.1f} min"
    )
    print("Benchmark wrote no persistent files and did not access KDDTest+.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
        help="Architectures processed by this one invocation (default: all four).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS_DEFAULT))
    parser.add_argument(
        "--general-template",
        default="results/{architecture}_baseline_cv_latest.json",
    )
    parser.add_argument(
        "--focal-template",
        default="results/{architecture}_focal_stage1_latest.json",
    )
    parser.add_argument(
        "--batching-template",
        default="results/{architecture}_batch_baseline_cv_latest.json",
    )
    parser.add_argument("--train-data", default="data/KDDTrain+.txt")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-prefix", default="robust_calibrated_super_stack")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="CPU worker processes; 0 auto-detects allowed physical cores.",
    )
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--candidate-chunk-size", type=int, default=27)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--epsilon", type=float, default=EPSILON_DEFAULT)
    parser.add_argument("--temperature-min", type=float, default=0.01)
    parser.add_argument("--temperature-max", type=float, default=100.0)
    parser.add_argument("--temperature-xatol", type=float, default=1e-8)
    parser.add_argument("--temperature-maxiter", type=int, default=500)
    parser.add_argument("--macro-guard", type=float, default=MACRO_GUARD_DEFAULT)
    parser.add_argument("--mcc-guard", type=float, default=MCC_GUARD_DEFAULT)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--benchmark",
        choices=ARCHITECTURES,
        help="Run one real-OOF timing task for this architecture and write nothing.",
    )
    args = parser.parse_args(argv)
    if len(args.architectures) != len(set(args.architectures)):
        parser.error("--architectures must be unique.")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if sorted(args.seeds) != list(SEEDS_DEFAULT):
        parser.error("This predeclared experiment requires exactly seeds 0 1 2.")
    if args.workers < 0:
        parser.error("--workers must be zero or positive.")
    if args.threads_per_worker <= 0:
        parser.error("--threads-per-worker must be positive.")
    if args.candidate_chunk_size <= 0:
        parser.error("--candidate-chunk-size must be positive.")
    if args.max_iter <= 0 or args.tolerance <= 0:
        parser.error("--max-iter and --tolerance must be positive.")
    if args.epsilon <= 0 or args.epsilon >= 1:
        parser.error("--epsilon must be in (0,1).")
    if args.temperature_min <= 0 or args.temperature_max <= args.temperature_min:
        parser.error("Temperature bounds must satisfy 0 < min < max.")
    if args.temperature_xatol <= 0 or args.temperature_maxiter <= 0:
        parser.error("Temperature optimizer tolerance/iterations must be positive.")
    if args.macro_guard < 0 or args.mcc_guard < 0:
        parser.error("Metric guards cannot be negative.")
    if args.top_n <= 0:
        parser.error("--top-n must be positive.")
    if not args.output_prefix.strip():
        parser.error("--output-prefix cannot be empty.")
    if Path(args.output_prefix).name != args.output_prefix or args.output_prefix in {".", ".."}:
        parser.error("--output-prefix must be a filename-safe prefix without path separators.")
    if args.dry_run and args.benchmark:
        parser.error("--dry-run and --benchmark are mutually exclusive.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    train_path = safe.resolve_cli_path(repo_root, args.train_data)
    results_dir = safe.resolve_cli_path(repo_root, args.results_dir)
    if not train_path.is_file():
        raise SystemExit(f"KDDTrain+ not found: {train_path}")
    architectures = [args.benchmark] if args.benchmark else list(args.architectures)
    workers = args.workers or detect_physical_cpus()
    workers = min(workers, available_logical_cpus())
    if workers * args.threads_per_worker > available_logical_cpus():
        raise SystemExit(
            f"workers ({workers}) x threads-per-worker ({args.threads_per_worker}) exceeds "
            f"the {available_logical_cpus()} logical CPUs available to this process."
        )
    settings = SearchSettings(
        epsilon=float(args.epsilon),
        max_iter=int(args.max_iter),
        tolerance=float(args.tolerance),
        temperature_min=float(args.temperature_min),
        temperature_max=float(args.temperature_max),
        temperature_xatol=float(args.temperature_xatol),
        temperature_maxiter=int(args.temperature_maxiter),
        candidate_chunk_size=int(args.candidate_chunk_size),
        macro_guard=float(args.macro_guard),
        mcc_guard=float(args.mcc_guard),
    )
    print("Loading and validating saved OOF inputs...", flush=True)
    inputs = [
        load_architecture_input(
            repo_root,
            train_path,
            architecture,
            sorted(args.seeds),
            args.general_template,
            args.focal_template,
            args.batching_template,
        )
        for architecture in architectures
    ]
    validate_cross_architecture_alignment(inputs)
    print_preflight(inputs, workers, args.threads_per_worker)
    if args.dry_run:
        print("Dry run complete; all inputs align, no models were fitted, and no files were written.")
        return
    if args.benchmark:
        run_benchmark(inputs[0], settings, workers, args.threads_per_worker)
        return

    definition = experiment_definition(
        script_path,
        train_path,
        inputs,
        settings,
        args.threads_per_worker,
    )
    experiment_key = stable_hash(definition, length=12)
    stem = f"{args.output_prefix.strip()}_{experiment_key}"
    results_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_protocol(
        definition,
        experiment_key,
        inputs,
    )
    protocol_path = results_dir / f"{stem}_protocol.json"
    if protocol_path.exists():
        existing_protocol = safe.read_json(protocol_path)
        if existing_protocol != protocol:
            raise RuntimeError(
                f"Immutable protocol collision at {protocol_path}; refuse to overwrite."
            )
    else:
        core.atomic_json(protocol_path, protocol)
    print(f"Experiment key: {experiment_key}")
    print(f"Protocol: {protocol_path}")

    architecture_results: Dict[str, Dict[str, Any]] = {}
    combined_summaries: List[pd.DataFrame] = []
    combined_per_seed: List[pd.DataFrame] = []
    task_run_paths: Dict[str, str] = {}
    overall_started = time.perf_counter()
    for architecture_input in inputs:
        architecture = architecture_input.architecture
        print(f"\n=== {ARCHITECTURE_LABELS[architecture]} ===", flush=True)
        cache_dir = results_dir / f"{stem}_tasks" / architecture
        cache_dir.mkdir(parents=True, exist_ok=True)
        tasks = make_tasks(architecture, architecture_input.seeds, experiment_key, cache_dir)
        task_runs = run_search_tasks(
            architecture_input,
            tasks,
            settings,
            workers,
            args.threads_per_worker,
            resume=not args.no_resume,
        )
        task_runs_path = results_dir / f"{stem}_{architecture}_task_runs.csv"
        core.atomic_csv(task_runs_path, task_runs)
        task_run_paths[architecture] = str(task_runs_path)
        result = evaluate_architecture(
            architecture_input,
            tasks,
            settings,
            stem,
            results_dir,
            args.top_n,
            experiment_key,
            protocol_path,
            train_path,
        )
        architecture_results[architecture] = {
            key: value
            for key, value in result.items()
            if key not in {"summary_frame", "per_seed_frame"}
        }
        combined_summaries.append(result["summary_frame"])
        combined_per_seed.append(result["per_seed_frame"])

    combined_summary = pd.concat(combined_summaries, ignore_index=True)
    combined_seed = pd.concat(combined_per_seed, ignore_index=True)
    combined_summary_path = results_dir / f"{stem}_all_architectures_nested_summary.csv"
    combined_seed_path = results_dir / f"{stem}_all_architectures_nested_per_seed.csv"
    core.atomic_csv(combined_summary_path, combined_summary)
    core.atomic_csv(combined_seed_path, combined_seed)

    is_default_all_architecture_run = architectures == list(ARCHITECTURES)
    latest_suffix = "" if is_default_all_architecture_run else "_" + "_".join(architectures)
    latest_path = results_dir / f"{args.output_prefix.strip()}{latest_suffix}_latest.json"
    latest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "architectures": architectures,
        "protocol": str(protocol_path),
        "protocol_sha256": core.sha256_file(protocol_path),
        "all_architectures_nested_summary": str(combined_summary_path),
        "all_architectures_nested_per_seed": str(combined_seed_path),
        "task_runs": task_run_paths,
        "architecture_results": architecture_results,
        "runtime_seconds": float(time.perf_counter() - overall_started),
        "workers": workers,
        "threads_per_worker": args.threads_per_worker,
        "kddtest_accessed": False,
        "status": "complete",
    }
    core.atomic_json(latest_path, latest)

    display_columns = [
        "model",
        "method_label",
        "runs",
        "macro_f1_mean",
        "mcc_mean",
        "rare_f1_mean",
        "balanced_rare_f1_mean",
        "robust_rare_f1_mean",
        "r2l_f1_mean",
        "u2r_f1_mean",
    ]
    print("\n=== Nested meta-level OOF summary ===")
    print(combined_summary[display_columns].to_string(index=False))
    print("\nSaved results:")
    print(f"  Summary: {combined_summary_path}")
    print(f"  Per-seed: {combined_seed_path}")
    print(f"  Protocol: {protocol_path}")
    print(f"  Latest pointer: {latest_path}")
    print("KDDTest+ accessed: NO")


if __name__ == "__main__":
    main()
