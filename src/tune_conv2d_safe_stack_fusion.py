"""Tune architecture-specific SAFE-Stack fusion from saved OOF probabilities.

This script performs no neural-network training and never opens KDDTest+.
It combines the matching four-fold OOF predictions produced by:

* General expert: cross-entropy with ordinary shuffled batches.
* Focal expert: class-balanced focal loss with ordinary shuffled batches.
* Batching expert: cross-entropy with minority-guaranteed batches.

For each training seed, the three inputs must describe the same KDDTrain+
rows and the same held-out fold assignment.  The base expert is selected once
from the three expert summaries by mean OOF Macro-F1, then mean MCC, then mean
Rare F1.  The fixed-base fusion grid contains 5,625 settings:

    15 R2L weight triplets x 15 U2R weight triplets
    x 5 R2L margins x 5 U2R margins.

Every setting is evaluated separately on each seed's complete pooled OOF
vector.  The ranking is based on mean Rare F1 across seeds, followed by mean
Macro-F1 and mean MCC.  All per-seed and all-configuration results are saved,
so a different predeclared selection rule can be applied without rerunning
models or the fusion search.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

import run_no_ctgan_model_ablation_4gpu as core


SCHEMA_VERSION = 1
ARCHITECTURE = "conv2d"
ARCHITECTURE_LABELS = {
    "conv1d": "Conv1D",
    "conv2d": "Conv2D",
}
EXPERTS = ("general", "focal", "batching")
EXPERT_LABELS = {
    "general": "General (cross-entropy)",
    "focal": "Focal",
    "batching": "Minority batching",
}
EXPERT_TIE_ORDER = {expert: index for index, expert in enumerate(EXPERTS)}

# Actual integer order used throughout this repository:
# 0=DoS, 1=Probe, 2=R2L, 3=U2R, 4=Normal.
R2L_CLASS = 2
U2R_CLASS = 3
RARE_CLASSES = (R2L_CLASS, U2R_CLASS)
MAJORITY_CLASSES = np.asarray([0, 1, 4], dtype=np.int64)

DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_MARGINS = (0.0, 0.025, 0.05, 0.10, 0.15)
DEFAULT_WEIGHT_STEP = 0.25
DEFAULT_SUPPORT = 2
METRICS = tuple(core.METRICS)

REQUIRED_OOF_KEYS = {
    "row_indices",
    "fold_ids",
    "labels",
    "probabilities",
    "raw_predictions",
}


@dataclass(frozen=True)
class OOFArtifact:
    expert: str
    seed: int
    path: Path
    sha256: str
    row_indices: np.ndarray
    fold_ids: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    raw_predictions: np.ndarray


@dataclass(frozen=True)
class CandidateGrid:
    simplex_weights: np.ndarray
    r2l_weight_index: np.ndarray
    u2r_weight_index: np.ndarray
    r2l_margin: np.ndarray
    u2r_margin: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.r2l_margin))


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def resolve_cli_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def resolve_recorded_path(
    repo_root: Path,
    pointer_path: Path,
    value: str | Path,
) -> Path:
    """Resolve an artifact path even after a repository was moved."""
    recorded = Path(value).expanduser()
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.extend(
            [
                pointer_path.parent / recorded,
                repo_root / recorded,
            ]
        )
    candidates.extend(
        [
            pointer_path.parent / recorded.name,
            repo_root / "results" / recorded.name,
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve recorded path {value!s} from {pointer_path}. "
        "If the results were moved, pass the matching --*-latest path."
    )


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def discover_standard_oof_paths(
    repo_root: Path,
    pointer_path: Path,
    seeds: Sequence[int],
    expected_training_mode: str,
    expected_architecture: str = ARCHITECTURE,
) -> Dict[int, Path]:
    pointer = read_json(pointer_path)
    if pointer.get("architecture") not in {None, expected_architecture}:
        raise ValueError(
            f"{pointer_path} belongs to architecture={pointer.get('architecture')!r}."
        )
    if "oof_directory" not in pointer:
        raise KeyError(f"Missing oof_directory in {pointer_path}.")
    if "protocol" not in pointer:
        raise KeyError(f"Missing protocol in {pointer_path}.")
    protocol_path = resolve_recorded_path(
        repo_root, pointer_path, str(pointer["protocol"])
    )
    protocol = read_json(protocol_path)
    training_settings = protocol.get("training_settings", {})
    observed_mode = training_settings.get("training_mode")
    if observed_mode != expected_training_mode:
        raise ValueError(
            f"{pointer_path} uses training_mode={observed_mode!r}; expected "
            f"{expected_training_mode!r}."
        )
    if training_settings.get("architecture") not in {None, expected_architecture}:
        raise ValueError(
            f"{protocol_path} is not a "
            f"{ARCHITECTURE_LABELS[expected_architecture]} protocol."
        )
    if protocol.get("kddtest_accessed") not in {None, False}:
        raise ValueError(f"{protocol_path} reports KDDTest+ access.")
    directory = resolve_recorded_path(
        repo_root, pointer_path, str(pointer["oof_directory"])
    )
    paths = {
        int(seed): directory / f"seed_{int(seed)}_oof_probabilities.npz"
        for seed in seeds
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing pooled OOF artifacts referenced by "
            f"{pointer_path}: {missing}"
        )
    return paths


def discover_focal_oof_paths(
    repo_root: Path,
    pointer_path: Path,
    seeds: Sequence[int],
    expected_architecture: str = ARCHITECTURE,
) -> tuple[Dict[int, Path], Dict[str, Any]]:
    pointer = read_json(pointer_path)
    if pointer.get("architecture") not in {None, expected_architecture}:
        raise ValueError(
            f"{pointer_path} belongs to architecture={pointer.get('architecture')!r}."
        )
    for key in ("best_config", "oof_directory"):
        if key not in pointer:
            raise KeyError(f"Missing {key} in {pointer_path}.")
    best_path = resolve_recorded_path(
        repo_root, pointer_path, str(pointer["best_config"])
    )
    best = read_json(best_path)
    if "protocol" not in pointer:
        raise KeyError(f"Missing protocol in {pointer_path}.")
    protocol_path = resolve_recorded_path(
        repo_root, pointer_path, str(pointer["protocol"])
    )
    protocol = read_json(protocol_path)
    settings = protocol.get("settings", {})
    expected_model = ARCHITECTURE_LABELS[expected_architecture]
    if settings.get("model") not in {None, expected_model}:
        raise ValueError(
            f"{protocol_path} is not a {expected_model} focal protocol."
        )
    if settings.get("batching") not in {None, "ordinary_shuffled"}:
        raise ValueError(
            f"{protocol_path} is not the ordinary-batch focal experiment."
        )
    if protocol.get("kddtest_accessed") not in {None, False}:
        raise ValueError(f"{protocol_path} reports KDDTest+ access.")
    config_id = str(best.get("config_id", "")).strip()
    if not config_id:
        raise ValueError(f"Missing selected config_id in {best_path}.")
    recorded_seeds = sorted(int(seed) for seed in best.get("training_seeds", []))
    if recorded_seeds and recorded_seeds != sorted(int(seed) for seed in seeds):
        raise ValueError(
            f"Focal best configuration contains seeds {recorded_seeds}, "
            f"but the requested seeds are {sorted(int(seed) for seed in seeds)}."
        )
    directory = resolve_recorded_path(
        repo_root, pointer_path, str(pointer["oof_directory"])
    )
    paths = {
        int(seed): directory / f"{config_id}_s{int(seed)}_oof_predictions.npz"
        for seed in seeds
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing selected focal OOF artifacts referenced by "
            f"{pointer_path}: {missing}"
        )
    best = {**best, "best_config_path": str(best_path)}
    return paths, best


def load_oof_artifact(expert: str, seed: int, path: Path) -> OOFArtifact:
    with np.load(path, allow_pickle=False) as artifact:
        missing = REQUIRED_OOF_KEYS - set(artifact.files)
        if missing:
            raise KeyError(f"{path} is missing NPZ arrays: {sorted(missing)}")
        row_indices = np.asarray(artifact["row_indices"], dtype=np.int64)
        fold_ids = np.asarray(artifact["fold_ids"], dtype=np.int64)
        labels = np.asarray(artifact["labels"], dtype=np.int64)
        probabilities = np.asarray(artifact["probabilities"], dtype=np.float32)
        raw_predictions = np.asarray(artifact["raw_predictions"], dtype=np.int64)

    row_count = len(labels)
    expected_rows = np.arange(row_count, dtype=np.int64)
    if not np.array_equal(row_indices, expected_rows):
        raise ValueError(
            f"{path} row_indices must be the ordered range 0..N-1."
        )
    if fold_ids.shape != (row_count,):
        raise ValueError(f"{path} has invalid fold_ids shape {fold_ids.shape}.")
    if labels.shape != (row_count,):
        raise ValueError(f"{path} has invalid labels shape {labels.shape}.")
    if raw_predictions.shape != (row_count,):
        raise ValueError(
            f"{path} has invalid raw_predictions shape {raw_predictions.shape}."
        )
    if probabilities.shape != (row_count, 5):
        raise ValueError(
            f"{path} has probability shape {probabilities.shape}; expected (N, 5)."
        )
    if row_count == 0:
        raise ValueError(f"{path} contains no OOF rows.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{path} contains non-finite probabilities.")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError(f"{path} contains values outside probability range [0, 1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError(f"{path} probability rows do not sum to one.")
    if not np.array_equal(raw_predictions, np.argmax(probabilities, axis=1)):
        raise ValueError(f"{path} raw_predictions do not match probability argmax.")
    if np.any((labels < 0) | (labels > 4)):
        raise ValueError(f"{path} contains a class label outside 0..4.")
    unique_folds = sorted(int(value) for value in np.unique(fold_ids))
    if unique_folds != [0, 1, 2, 3]:
        raise ValueError(
            f"{path} must contain fold IDs [0,1,2,3], got {unique_folds}."
        )

    return OOFArtifact(
        expert=expert,
        seed=int(seed),
        path=path.resolve(),
        sha256=core.sha256_file(path),
        row_indices=row_indices,
        fold_ids=fold_ids,
        labels=labels,
        probabilities=probabilities,
        raw_predictions=raw_predictions,
    )


def validate_alignment(
    artifacts: Mapping[str, Mapping[int, OOFArtifact]],
    seeds: Sequence[int],
) -> None:
    reference = artifacts[EXPERTS[0]][int(seeds[0])]
    for expert in EXPERTS:
        if sorted(artifacts[expert]) != sorted(int(seed) for seed in seeds):
            raise ValueError(f"Expert {expert} does not contain every requested seed.")
        for seed in seeds:
            current = artifacts[expert][int(seed)]
            for field in ("row_indices", "fold_ids", "labels"):
                if not np.array_equal(getattr(current, field), getattr(reference, field)):
                    raise ValueError(
                        f"OOF alignment failure for expert={expert}, seed={seed}: "
                        f"{field} differs from the shared reference."
                    )


def generate_simplex_weights(step: float = DEFAULT_WEIGHT_STEP) -> np.ndarray:
    if not np.isfinite(step) or step <= 0.0 or step > 1.0:
        raise ValueError("Weight step must be finite and in (0, 1].")
    units_float = 1.0 / float(step)
    units = int(round(units_float))
    if not np.isclose(units_float, units, atol=1e-12, rtol=0.0):
        raise ValueError("Weight step must divide one exactly, such as 0.25.")
    weights: List[List[float]] = []
    for general_units in range(units, -1, -1):
        for focal_units in range(units - general_units, -1, -1):
            batching_units = units - general_units - focal_units
            weights.append(
                [
                    general_units / units,
                    focal_units / units,
                    batching_units / units,
                ]
            )
    result = np.asarray(weights, dtype=np.float64)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("Generated expert weights do not sum to one.")
    if np.any(result < 0.0):
        raise RuntimeError("Generated a negative expert weight.")
    return result


def make_candidate_grid(
    simplex_weights: np.ndarray,
    margins: Sequence[float],
) -> CandidateGrid:
    margin_values = np.asarray(sorted(float(value) for value in margins), dtype=np.float64)
    if (
        len(margin_values) == 0
        or len(np.unique(margin_values)) != len(margin_values)
        or np.any(~np.isfinite(margin_values))
        or np.any(margin_values < 0.0)
    ):
        raise ValueError("Margins must be unique, finite, nonnegative values.")

    rare_options = [
        (weight_index, float(margin))
        for weight_index in range(len(simplex_weights))
        for margin in margin_values
    ]
    r_weight: List[int] = []
    u_weight: List[int] = []
    r_margin: List[float] = []
    u_margin: List[float] = []
    for r_weight_index, r_margin_value in rare_options:
        for u_weight_index, u_margin_value in rare_options:
            r_weight.append(r_weight_index)
            u_weight.append(u_weight_index)
            r_margin.append(r_margin_value)
            u_margin.append(u_margin_value)
    return CandidateGrid(
        simplex_weights=np.asarray(simplex_weights, dtype=np.float64),
        r2l_weight_index=np.asarray(r_weight, dtype=np.int64),
        u2r_weight_index=np.asarray(u_weight, dtype=np.int64),
        r2l_margin=np.asarray(r_margin, dtype=np.float64),
        u2r_margin=np.asarray(u_margin, dtype=np.float64),
    )


def metrics_from_confusions(confusions: np.ndarray) -> pd.DataFrame:
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
    accuracy = np.divide(
        correct, totals, out=np.zeros_like(correct), where=totals != 0
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
            "rare_f1": f1[:, [R2L_CLASS, U2R_CLASS]].mean(axis=1),
            "minimum_minority_recall": recall[:, [R2L_CLASS, U2R_CLASS]].min(axis=1),
            "r2l_precision": precision[:, R2L_CLASS],
            "r2l_recall": recall[:, R2L_CLASS],
            "r2l_f1": f1[:, R2L_CLASS],
            "u2r_precision": precision[:, U2R_CLASS],
            "u2r_recall": recall[:, U2R_CLASS],
            "u2r_f1": f1[:, U2R_CLASS],
        }
    )


def stable_top_two_support(probabilities: np.ndarray, class_id: int) -> np.ndarray:
    """Count experts that place class_id in their deterministic top two."""
    # Input is N x 3 experts x 5 classes. Stable sorting resolves exact ties by
    # lower class ID because the original class axis is ordered 0..4.
    top_two = np.argsort(-probabilities, axis=2, kind="stable")[:, :, :2]
    return np.any(top_two == int(class_id), axis=2).sum(axis=1).astype(np.int8)


def score_fusion_confusions(
    labels: np.ndarray,
    expert_probabilities: np.ndarray,
    base_index: int,
    grid: CandidateGrid,
    minimum_support: int,
    sample_chunk_size: int,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Evaluate all fusion candidates using memory-bounded sample chunks."""
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(expert_probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), len(EXPERTS), 5):
        raise ValueError(
            "expert_probabilities must have shape (N, 3, 5); got "
            f"{probabilities.shape}."
        )
    if not 0 <= base_index < len(EXPERTS):
        raise ValueError("base_index is outside the expert axis.")
    if minimum_support < 1 or minimum_support > len(EXPERTS):
        raise ValueError("minimum_support must be between one and three.")
    if sample_chunk_size <= 0:
        raise ValueError("sample_chunk_size must be positive.")

    candidate_count = grid.size
    pair_offsets = (25 * np.arange(candidate_count, dtype=np.int64))[None, :]
    counts = np.zeros((candidate_count, 25), dtype=np.int64)
    diagnostic_names = (
        "r2l_overrides",
        "u2r_overrides",
        "both_rare_pass",
        "changed_predictions",
    )
    diagnostics = {
        name: np.zeros(candidate_count, dtype=np.int64) for name in diagnostic_names
    }

    for start in range(0, len(labels), sample_chunk_size):
        stop = min(len(labels), start + sample_chunk_size)
        chunk = probabilities[start:stop]
        chunk_labels = labels[start:stop]
        base_probabilities = chunk[:, base_index, :]
        base_predictions = np.argmax(base_probabilities, axis=1).astype(np.int8)
        base_is_majority = ~np.isin(base_predictions, RARE_CLASSES)
        majority_anchor = base_probabilities[:, MAJORITY_CLASSES].max(axis=1)

        r2l_support = stable_top_two_support(chunk, R2L_CLASS)
        u2r_support = stable_top_two_support(chunk, U2R_CLASS)
        r2l_weight_scores = chunk[:, :, R2L_CLASS] @ grid.simplex_weights.T
        u2r_weight_scores = chunk[:, :, U2R_CLASS] @ grid.simplex_weights.T
        r2l_scores = r2l_weight_scores[:, grid.r2l_weight_index]
        u2r_scores = u2r_weight_scores[:, grid.u2r_weight_index]
        r2l_excess = (
            r2l_scores
            - majority_anchor[:, None]
            - grid.r2l_margin[None, :]
        )
        u2r_excess = (
            u2r_scores
            - majority_anchor[:, None]
            - grid.u2r_margin[None, :]
        )
        r2l_pass = (
            base_is_majority[:, None]
            & (r2l_support[:, None] >= minimum_support)
            & (r2l_excess >= -1e-12)
        )
        u2r_pass = (
            base_is_majority[:, None]
            & (u2r_support[:, None] >= minimum_support)
            & (u2r_excess >= -1e-12)
        )
        both_pass = r2l_pass & u2r_pass

        # If both pass, use the class with the larger margin excess. An exact
        # tie keeps the base prediction, which is the conservative outcome.
        choose_r2l = r2l_pass & (
            ~u2r_pass | (r2l_excess > u2r_excess + 1e-12)
        )
        choose_u2r = u2r_pass & (
            ~r2l_pass | (u2r_excess > r2l_excess + 1e-12)
        )

        predictions = np.broadcast_to(
            base_predictions[:, None], (len(chunk), candidate_count)
        ).copy()
        predictions[choose_r2l] = R2L_CLASS
        predictions[choose_u2r] = U2R_CLASS

        codes = chunk_labels[:, None] * 5 + predictions.astype(np.int64)
        linear_codes = codes + pair_offsets
        counts += np.bincount(
            linear_codes.ravel(), minlength=candidate_count * 25
        ).reshape(candidate_count, 25)
        diagnostics["r2l_overrides"] += choose_r2l.sum(axis=0)
        diagnostics["u2r_overrides"] += choose_u2r.sum(axis=0)
        diagnostics["both_rare_pass"] += both_pass.sum(axis=0)
        diagnostics["changed_predictions"] += (
            predictions != base_predictions[:, None]
        ).sum(axis=0)

    return counts.reshape(candidate_count, 5, 5), diagnostics


def predict_one_fusion(
    expert_probabilities: np.ndarray,
    base_index: int,
    r2l_weights: np.ndarray,
    u2r_weights: np.ndarray,
    r2l_margin: float,
    u2r_margin: float,
    minimum_support: int,
) -> Dict[str, np.ndarray]:
    probabilities = np.asarray(expert_probabilities, dtype=np.float64)
    base_probabilities = probabilities[:, base_index, :]
    base_predictions = np.argmax(base_probabilities, axis=1).astype(np.int64)
    majority_anchor = base_probabilities[:, MAJORITY_CLASSES].max(axis=1)
    base_is_majority = ~np.isin(base_predictions, RARE_CLASSES)
    r2l_support = stable_top_two_support(probabilities, R2L_CLASS)
    u2r_support = stable_top_two_support(probabilities, U2R_CLASS)
    r2l_scores = probabilities[:, :, R2L_CLASS] @ np.asarray(r2l_weights)
    u2r_scores = probabilities[:, :, U2R_CLASS] @ np.asarray(u2r_weights)
    r2l_excess = r2l_scores - majority_anchor - float(r2l_margin)
    u2r_excess = u2r_scores - majority_anchor - float(u2r_margin)
    r2l_pass = (
        base_is_majority
        & (r2l_support >= minimum_support)
        & (r2l_excess >= -1e-12)
    )
    u2r_pass = (
        base_is_majority
        & (u2r_support >= minimum_support)
        & (u2r_excess >= -1e-12)
    )
    choose_r2l = r2l_pass & (
        ~u2r_pass | (r2l_excess > u2r_excess + 1e-12)
    )
    choose_u2r = u2r_pass & (
        ~r2l_pass | (u2r_excess > r2l_excess + 1e-12)
    )
    predictions = base_predictions.copy()
    predictions[choose_r2l] = R2L_CLASS
    predictions[choose_u2r] = U2R_CLASS
    return {
        "base_predictions": base_predictions,
        "final_predictions": predictions,
        "majority_anchor": majority_anchor.astype(np.float32),
        "r2l_fused_score": r2l_scores.astype(np.float32),
        "u2r_fused_score": u2r_scores.astype(np.float32),
        "r2l_support": r2l_support,
        "u2r_support": u2r_support,
        "r2l_override": choose_r2l,
        "u2r_override": choose_u2r,
    }


def expert_seed_metrics(
    artifacts: Mapping[str, Mapping[int, OOFArtifact]],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for expert in EXPERTS:
        for seed in seeds:
            artifact = artifacts[expert][int(seed)]
            metrics = core.calculate_metrics(artifact.labels, artifact.raw_predictions)
            rows.append(
                {
                    "expert": expert,
                    "expert_label": EXPERT_LABELS[expert],
                    "seed": int(seed),
                    "oof_path": str(artifact.path),
                    **metrics,
                }
            )
    per_seed = pd.DataFrame(rows).sort_values(["expert", "seed"]).reset_index(drop=True)
    summary_rows: List[Dict[str, Any]] = []
    for expert in EXPERTS:
        group = per_seed[per_seed["expert"] == expert]
        row: Dict[str, Any] = {
            "expert": expert,
            "expert_label": EXPERT_LABELS[expert],
            "runs": int(len(group)),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"])),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["_expert_tie_order"] = summary["expert"].map(EXPERT_TIE_ORDER)
    summary = summary.sort_values(
        ["macro_f1_mean", "mcc_mean", "rare_f1_mean", "_expert_tie_order"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).drop(columns="_expert_tie_order").reset_index(drop=True)
    summary.insert(0, "base_rank", np.arange(1, len(summary) + 1))
    summary["selected_base"] = summary["base_rank"] == 1
    return per_seed, summary


def candidate_columns(grid: CandidateGrid) -> pd.DataFrame:
    r_weights = grid.simplex_weights[grid.r2l_weight_index]
    u_weights = grid.simplex_weights[grid.u2r_weight_index]
    return pd.DataFrame(
        {
            "candidate_id": np.arange(grid.size, dtype=np.int64),
            "r2l_weight_general": r_weights[:, 0],
            "r2l_weight_focal": r_weights[:, 1],
            "r2l_weight_batching": r_weights[:, 2],
            "u2r_weight_general": u_weights[:, 0],
            "u2r_weight_focal": u_weights[:, 1],
            "u2r_weight_batching": u_weights[:, 2],
            "r2l_margin": grid.r2l_margin,
            "u2r_margin": grid.u2r_margin,
        }
    )


def aggregate_candidate_scores(
    per_seed: pd.DataFrame,
    candidates: pd.DataFrame,
    seeds: Sequence[int],
) -> pd.DataFrame:
    summary_rows: List[Dict[str, Any]] = []
    expected_seeds = sorted(int(seed) for seed in seeds)
    for candidate_id, group in per_seed.groupby("candidate_id", sort=True):
        observed = sorted(int(seed) for seed in group["seed"])
        if observed != expected_seeds:
            raise RuntimeError(
                f"Candidate {candidate_id} is missing one or more seed results."
            )
        row: Dict[str, Any] = {
            "candidate_id": int(candidate_id),
            "runs": int(len(group)),
            "seeds": ",".join(str(seed) for seed in observed),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        for diagnostic in (
            "r2l_overrides",
            "u2r_overrides",
            "both_rare_pass",
            "changed_predictions",
        ):
            values = pd.to_numeric(group[diagnostic], errors="raise")
            row[f"{diagnostic}_mean"] = float(values.mean())
            row[f"{diagnostic}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = candidates.merge(pd.DataFrame(summary_rows), on="candidate_id", how="inner")
    summary["minimum_rare_f1_mean"] = np.minimum(
        summary["r2l_f1_mean"], summary["u2r_f1_mean"]
    )
    summary["total_margin"] = summary["r2l_margin"] + summary["u2r_margin"]
    summary = summary.sort_values(
        [
            "rare_f1_mean",
            "macro_f1_mean",
            "mcc_mean",
            "minimum_rare_f1_mean",
            "rare_f1_std",
            "total_margin",
            "candidate_id",
        ],
        ascending=[False, False, False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    return summary


def format_summary(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    output = frame.head(top_n)[
        [
            "rank",
            "r2l_weight_general",
            "r2l_weight_focal",
            "r2l_weight_batching",
            "u2r_weight_general",
            "u2r_weight_focal",
            "u2r_weight_batching",
            "r2l_margin",
            "u2r_margin",
        ]
    ].copy()
    labels = {
        "rare_f1": "Rare F1",
        "macro_f1": "Macro-F1",
        "mcc": "MCC",
        "r2l_precision": "R2L Precision",
        "r2l_recall": "R2L Recall",
        "u2r_precision": "U2R Precision",
        "u2r_recall": "U2R Recall",
    }
    source = frame.head(top_n)
    for metric, label in labels.items():
        output[label] = [
            f"{100.0 * mean:.2f}% +/- {100.0 * std:.2f}%"
            for mean, std in zip(
                source[f"{metric}_mean"], source[f"{metric}_std"], strict=True
            )
        ]
    return output


def simple_average_metrics(
    artifacts: Mapping[str, Mapping[int, OOFArtifact]],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        reference = artifacts[EXPERTS[0]][int(seed)]
        probabilities = np.stack(
            [artifacts[expert][int(seed)].probabilities for expert in EXPERTS], axis=1
        )
        predictions = np.argmax(probabilities.mean(axis=1), axis=1)
        rows.append(
            {
                "method": "simple_average",
                "method_label": "Simple probability average",
                "seed": int(seed),
                **core.calculate_metrics(reference.labels, predictions),
            }
        )
    return rows


def build_reference_ablation(
    expert_per_seed: pd.DataFrame,
    average_rows: Sequence[Dict[str, Any]],
    selected_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    for _, row in expert_per_seed.iterrows():
        rows.append(
            {
                "method": str(row["expert"]),
                "method_label": str(row["expert_label"]),
                "seed": int(row["seed"]),
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    rows.extend(dict(row) for row in average_rows)
    for _, row in selected_rows.iterrows():
        rows.append(
            {
                "method": "safe_stack",
                "method_label": "SAFE-Stack selected fusion",
                "seed": int(row["seed"]),
                **{metric: float(row[metric]) for metric in METRICS},
            }
        )
    per_seed = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate([*EXPERTS, "simple_average", "safe_stack"])}
    summary_rows: List[Dict[str, Any]] = []
    for method, group in per_seed.groupby("method", sort=False):
        row: Dict[str, Any] = {
            "method": method,
            "method_label": str(group.iloc[0]["method_label"]),
            "runs": int(len(group)),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"])),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary["_order"] = summary["method"].map(order)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    return per_seed, summary


def parse_args(
    argv: Sequence[str] | None = None,
    default_architecture: str = ARCHITECTURE,
) -> argparse.Namespace:
    architecture = str(default_architecture).strip().lower()
    if architecture not in ARCHITECTURE_LABELS:
        raise ValueError(f"Unsupported SAFE-Stack architecture: {architecture!r}")
    architecture_label = ARCHITECTURE_LABELS[architecture]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--general-latest",
        default=f"results/{architecture}_baseline_cv_latest.json",
        help=(
            f"Latest-results JSON for {architecture_label} "
            "cross-entropy/ordinary batches."
        ),
    )
    parser.add_argument(
        "--focal-latest",
        default=f"results/{architecture}_focal_stage1_latest.json",
        help=(
            f"Latest-results JSON for the {architecture_label} "
            "focal-only Stage-1 sweep."
        ),
    )
    parser.add_argument(
        "--batching-latest",
        default=f"results/{architecture}_batch_baseline_cv_latest.json",
        help=(
            f"Latest-results JSON for {architecture_label} "
            "cross-entropy/minority batches."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--weight-step",
        type=float,
        default=DEFAULT_WEIGHT_STEP,
        help="Simplex weight increment; 0.25 produces 15 triplets.",
    )
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=list(DEFAULT_MARGINS),
    )
    parser.add_argument(
        "--minimum-support", type=int, default=DEFAULT_SUPPORT, choices=[1, 2, 3]
    )
    parser.add_argument(
        "--sample-chunk-size",
        type=int,
        default=256,
        help="Number of OOF rows scored against all candidates simultaneously.",
    )
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument(
        "--output-prefix", default=f"{architecture}_safe_stack_fusion"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and describe the inputs/grid without evaluating candidates.",
    )
    args = parser.parse_args(argv)
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds cannot be negative.")
    if args.sample_chunk_size <= 0:
        parser.error("--sample-chunk-size must be positive.")
    if args.top_n <= 0:
        parser.error("--top-n must be positive.")
    if not args.output_prefix.strip():
        parser.error("--output-prefix cannot be empty.")
    args.architecture = architecture
    return args


def main(
    argv: Sequence[str] | None = None,
    default_architecture: str = ARCHITECTURE,
) -> None:
    args = parse_args(argv, default_architecture=default_architecture)
    architecture = str(args.architecture)
    architecture_label = ARCHITECTURE_LABELS[architecture]
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    results_dir = repo_root / "results"
    seeds = sorted(int(seed) for seed in args.seeds)
    latest_paths = {
        "general": resolve_cli_path(repo_root, args.general_latest),
        "focal": resolve_cli_path(repo_root, args.focal_latest),
        "batching": resolve_cli_path(repo_root, args.batching_latest),
    }

    standard_paths = {
        "general": discover_standard_oof_paths(
            repo_root,
            latest_paths["general"],
            seeds,
            expected_training_mode="baseline_ce",
            expected_architecture=architecture,
        ),
        "batching": discover_standard_oof_paths(
            repo_root,
            latest_paths["batching"],
            seeds,
            expected_training_mode="baseline_batch",
            expected_architecture=architecture,
        ),
    }
    focal_paths, focal_best = discover_focal_oof_paths(
        repo_root,
        latest_paths["focal"],
        seeds,
        expected_architecture=architecture,
    )
    source_paths = {**standard_paths, "focal": focal_paths}
    artifacts: Dict[str, Dict[int, OOFArtifact]] = {
        expert: {
            seed: load_oof_artifact(expert, seed, source_paths[expert][seed])
            for seed in seeds
        }
        for expert in EXPERTS
    }
    validate_alignment(artifacts, seeds)

    weights = generate_simplex_weights(args.weight_step)
    grid = make_candidate_grid(weights, args.margins)
    expected_triplets = 15 if np.isclose(args.weight_step, 0.25) else len(weights)
    expected_candidates = len(weights) ** 2 * len(args.margins) ** 2
    if len(weights) != expected_triplets or grid.size != expected_candidates:
        raise RuntimeError("Generated fusion grid has an unexpected size.")

    expert_per_seed, base_selection = expert_seed_metrics(artifacts, seeds)
    base_expert = str(base_selection.iloc[0]["expert"])
    base_index = EXPERTS.index(base_expert)
    print(f"{architecture_label} SAFE-Stack OOF fusion search")
    print(f"Seeds: {seeds}; matched folds per seed: 4")
    print(f"Selected base: {EXPERT_LABELS[base_expert]}")
    print(
        "Base rule: highest mean OOF Macro-F1, then mean MCC, then mean Rare F1"
    )
    print(f"Simplex weights: {len(weights)}; candidate settings: {grid.size}")
    print(f"Minimum top-two support: {args.minimum_support} of 3 experts")
    print(f"Margins: {sorted(float(value) for value in args.margins)}")
    print("KDDTest+ accessed: NO")
    if args.dry_run:
        print("Dry run complete; inputs align and no result files were written.")
        return

    candidates = candidate_columns(grid)
    per_seed_parts: List[pd.DataFrame] = []
    started = time.perf_counter()
    for position, seed in enumerate(seeds, start=1):
        reference = artifacts[EXPERTS[0]][seed]
        probabilities = np.stack(
            [artifacts[expert][seed].probabilities for expert in EXPERTS], axis=1
        )
        seed_started = time.perf_counter()
        confusions, diagnostics = score_fusion_confusions(
            reference.labels,
            probabilities,
            base_index,
            grid,
            int(args.minimum_support),
            int(args.sample_chunk_size),
        )
        frame = metrics_from_confusions(confusions)
        frame.insert(0, "candidate_id", candidates["candidate_id"].to_numpy())
        frame.insert(0, "seed", int(seed))
        for name, values in diagnostics.items():
            frame[name] = values
        per_seed_parts.append(frame)
        elapsed = time.perf_counter() - seed_started
        print(
            f"Seed {seed} complete ({position}/{len(seeds)}): {elapsed:.1f} seconds",
            flush=True,
        )

    per_seed_scores = pd.concat(per_seed_parts, ignore_index=True)
    ranking = aggregate_candidate_scores(per_seed_scores, candidates, seeds)
    best = ranking.iloc[0]
    selected_id = int(best["candidate_id"])
    selected_seed_scores = per_seed_scores[
        per_seed_scores["candidate_id"] == selected_id
    ].sort_values("seed")

    source_identity = {
        expert: {
            str(seed): {
                "path": str(artifacts[expert][seed].path),
                "sha256": artifacts[expert][seed].sha256,
            }
            for seed in seeds
        }
        for expert in EXPERTS
    }
    settings = {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture,
        "seeds": seeds,
        "fold_count": 4,
        "base_selection_rule": (
            "mean OOF Macro-F1 descending; mean MCC descending; "
            "mean Rare F1 descending; fixed expert order"
        ),
        "selected_base": base_expert,
        "weight_step": float(args.weight_step),
        "simplex_weight_count": int(len(weights)),
        "margins": sorted(float(value) for value in args.margins),
        "minimum_support": int(args.minimum_support),
        "support_definition": "rare class is in an expert's stable top two",
        "base_rare_predictions_preserved": True,
        "majority_anchor": "maximum base probability over class IDs 0,1,4",
        "both_pass_rule": (
            "larger fused-score-minus-majority-minus-margin excess wins; "
            "an exact tie preserves the base"
        ),
        "ranking_rule": (
            "mean Rare F1 descending; mean Macro-F1 descending; mean MCC "
            "descending; mean minimum per-class rare F1 descending; Rare F1 "
            "sample SD ascending; total margin descending"
        ),
        "candidate_count": int(grid.size),
        "selection_data": "KDDTrain+ pooled OOF predictions only",
        "kddtest_accessed": False,
        "focal_best": focal_best,
        "source_artifacts": source_identity,
    }
    experiment_key = stable_hash(settings, 12)
    stem = f"{args.output_prefix.strip()}_{experiment_key}"
    protocol_path = results_dir / f"{stem}_protocol.json"
    expert_seed_path = results_dir / f"{stem}_expert_seed_metrics.csv"
    base_path = results_dir / f"{stem}_base_selection.csv"
    per_seed_path = results_dir / f"{stem}_per_seed_scores.csv"
    ranking_path = results_dir / f"{stem}_ranking.csv"
    formatted_path = results_dir / f"{stem}_top_formatted.csv"
    best_path = results_dir / f"{stem}_best_config.json"
    ablation_seed_path = results_dir / f"{stem}_validation_ablation_per_seed.csv"
    ablation_summary_path = results_dir / f"{stem}_validation_ablation_summary.csv"
    selected_dir = results_dir / f"{stem}_selected_oof"
    latest_path = results_dir / f"{args.output_prefix.strip()}_latest.json"

    selected_r_weights = np.asarray(
        [
            best["r2l_weight_general"],
            best["r2l_weight_focal"],
            best["r2l_weight_batching"],
        ],
        dtype=np.float64,
    )
    selected_u_weights = np.asarray(
        [
            best["u2r_weight_general"],
            best["u2r_weight_focal"],
            best["u2r_weight_batching"],
        ],
        dtype=np.float64,
    )
    selected_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        reference = artifacts[EXPERTS[0]][seed]
        probabilities = np.stack(
            [artifacts[expert][seed].probabilities for expert in EXPERTS], axis=1
        )
        selected_predictions = predict_one_fusion(
            probabilities,
            base_index,
            selected_r_weights,
            selected_u_weights,
            float(best["r2l_margin"]),
            float(best["u2r_margin"]),
            int(args.minimum_support),
        )
        core.atomic_npz(
            selected_dir / f"seed_{seed}_selected_fusion_oof.npz",
            row_indices=reference.row_indices,
            fold_ids=reference.fold_ids,
            labels=reference.labels,
            expert_probabilities=probabilities.astype(np.float32),
            **selected_predictions,
        )

    average_rows = simple_average_metrics(artifacts, seeds)
    ablation_per_seed, ablation_summary = build_reference_ablation(
        expert_per_seed, average_rows, selected_seed_scores
    )
    pretty = format_summary(ranking, min(int(args.top_n), len(ranking)))
    best_config = {
        **settings,
        "experiment_key": experiment_key,
        "rank": 1,
        "candidate_id": selected_id,
        "r2l_weights": {
            expert: float(selected_r_weights[index])
            for index, expert in enumerate(EXPERTS)
        },
        "u2r_weights": {
            expert: float(selected_u_weights[index])
            for index, expert in enumerate(EXPERTS)
        },
        "r2l_margin": float(best["r2l_margin"]),
        "u2r_margin": float(best["u2r_margin"]),
        "metrics": {
            metric: {
                "mean": float(best[f"{metric}_mean"]),
                "sample_std": float(best[f"{metric}_std"]),
            }
            for metric in METRICS
        },
        "development_score_warning": (
            "The same OOF predictions select and report this configuration; "
            "use untouched KDDTest+ only after freezing it for final evaluation."
        ),
    }
    protocol = {
        **settings,
        "experiment_key": experiment_key,
        "script_path": str(script_path),
        "script_sha256": core.sha256_file(script_path),
        "runtime_seconds": float(time.perf_counter() - started),
        "outputs": {
            "expert_seed_metrics": str(expert_seed_path),
            "base_selection": str(base_path),
            "per_seed_scores": str(per_seed_path),
            "ranking": str(ranking_path),
            "formatted_top": str(formatted_path),
            "best_config": str(best_path),
            "validation_ablation_per_seed": str(ablation_seed_path),
            "validation_ablation_summary": str(ablation_summary_path),
            "selected_oof_directory": str(selected_dir),
        },
    }

    core.atomic_json(protocol_path, protocol)
    core.atomic_csv(expert_seed_path, expert_per_seed)
    core.atomic_csv(base_path, base_selection)
    core.atomic_csv(per_seed_path, per_seed_scores)
    core.atomic_csv(ranking_path, ranking)
    core.atomic_csv(formatted_path, pretty)
    core.atomic_json(best_path, best_config)
    core.atomic_csv(ablation_seed_path, ablation_per_seed)
    core.atomic_csv(ablation_summary_path, ablation_summary)
    latest = {
        "schema_version": SCHEMA_VERSION,
        "architecture": architecture,
        "experiment_key": experiment_key,
        "protocol": str(protocol_path),
        "expert_seed_metrics": str(expert_seed_path),
        "base_selection": str(base_path),
        "per_seed_scores": str(per_seed_path),
        "ranking": str(ranking_path),
        "formatted_top": str(formatted_path),
        "best_config": str(best_path),
        "validation_ablation_per_seed": str(ablation_seed_path),
        "validation_ablation_summary": str(ablation_summary_path),
        "selected_oof_directory": str(selected_dir),
    }
    core.atomic_json(latest_path, latest)

    print(f"\n=== Ranked {architecture_label} SAFE-Stack configurations ===")
    print(pretty.to_string(index=False))
    print("\nSelected configuration:")
    print(f"  Base: {EXPERT_LABELS[base_expert]}")
    print(f"  R2L weights [G,F,B]: {selected_r_weights.tolist()}")
    print(f"  U2R weights [G,F,B]: {selected_u_weights.tolist()}")
    print(f"  R2L margin: {float(best['r2l_margin']):g}")
    print(f"  U2R margin: {float(best['u2r_margin']):g}")
    print(f"  Rare F1: {100.0 * float(best['rare_f1_mean']):.2f}% +/- "
          f"{100.0 * float(best['rare_f1_std']):.2f}%")
    print(f"  Macro-F1: {100.0 * float(best['macro_f1_mean']):.2f}% +/- "
          f"{100.0 * float(best['macro_f1_std']):.2f}%")
    print("\nSaved results:")
    print(f"  Full ranking: {ranking_path}")
    print(f"  Per-seed scores: {per_seed_path}")
    print(f"  Five-row validation ablation: {ablation_summary_path}")
    print(f"  Best configuration: {best_path}")
    print(f"  Latest pointer: {latest_path}")


if __name__ == "__main__":
    main()
