"""Select and audit fusion rules against four validation-only baselines.

This script reuses the completed Robust Calibrated Super-Stack rankings from
experiment ``6c7924f8d3a8``.  For each architecture it compares fusion against
four standalone KDDTrain+ OOF policies:

* General (cross-entropy baseline),
* focal-only,
* minority-batching-only, and
* validation-selected score scaling of the General probabilities.

A fusion passes when its mean Rare F1 is at least ``--minimum-improvement``
above every standalone policy.  Rare F1 is the only eligibility and ranking
metric.  Existing Macro-F1/MCC guard columns are deliberately ignored.

The script also audits the selection rule with four outer folds.  Within each
outer round, scaling and fusion are selected using only the other three folds;
the selected policies are then evaluated on the held-out fold.  No backbone is
retrained, no GPU is used, and KDDTest+ is never accessed.

Examples:

    python -u src/select_fusion_against_all_validation_baselines.py --dry-run
    python -u src/select_fusion_against_all_validation_baselines.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_no_ctgan_model_ablation_4gpu as core
import tune_conv2d_score_scaling_cv_4gpu as scaling
import tune_robust_calibrated_super_stack_all as stack


SCHEMA_VERSION = 1
SEARCH_EXPERIMENT_KEY = "6c7924f8d3a8"
SEARCH_PREFIX = "robust_calibrated_super_stack"
ARCHITECTURES = tuple(stack.ARCHITECTURES)
SEEDS = tuple(stack.SEEDS_DEFAULT)
FOLDS = tuple(stack.FOLD_IDS)
METRICS = tuple(core.METRICS)
STANDALONE_METHODS = ("general", "focal", "batching", "scaling")
METHOD_LABELS = {
    "general": "General (cross-entropy)",
    "focal": "Focal",
    "batching": "Minority batching",
    "scaling": "Baseline + selected score scaling",
    "selected_fusion": "Outer-fold best-fusion audit",
}
FUSION_FAMILIES = {"stack", "average_offset"}
COMPARISON_DECIMALS = 12
NUMERIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ScalingSource:
    pointer_path: Path
    pointer_sha256: str
    protocol_path: Path
    protocol_sha256: str
    ranking_path: Path
    ranking_sha256: str
    best_path: Path
    best_sha256: str
    coefficients: tuple[float, ...]
    stored_best: Dict[str, Any]


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def bool_series(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized) - {"true", "false"})
    if unexpected:
        raise ValueError(f"{name} contains non-boolean values: {unexpected}")
    return normalized.eq("true")


def resolve_recorded_path(
    recorded: str | Path,
    owner_path: Path,
    results_dir: Path,
) -> Path:
    value = Path(recorded).expanduser()
    candidates = [value]
    if not value.is_absolute():
        candidates.extend((owner_path.parent / value, results_dir / value))
    candidates.append(results_dir / value.name)
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not resolve recorded artifact {recorded!s} from {owner_path}."
    )


def load_search_protocol(
    results_dir: Path,
) -> tuple[Path, Dict[str, Any], stack.SearchSettings]:
    path = results_dir / f"{SEARCH_PREFIX}_{SEARCH_EXPERIMENT_KEY}_protocol.json"
    protocol = read_json(path)
    expected = {
        "schema_version": stack.SCHEMA_VERSION,
        "experiment_key": SEARCH_EXPERIMENT_KEY,
        "kddtest_accessed": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(
                f"Unexpected search protocol {key}={protocol.get(key)!r}; "
                f"expected {value!r}."
            )
    definition = protocol.get("definition")
    if not isinstance(definition, dict):
        raise KeyError(f"{path} is missing its experiment definition.")
    if definition.get("architectures") != list(ARCHITECTURES):
        raise ValueError(f"{path} is not the frozen all-architecture search.")
    if definition.get("seeds") != list(SEEDS):
        raise ValueError(f"{path} does not use frozen seeds {list(SEEDS)}.")
    settings_value = definition.get("settings")
    if not isinstance(settings_value, dict):
        raise KeyError(f"{path} is missing search settings.")
    try:
        settings = stack.SearchSettings(**settings_value)
    except TypeError as error:
        raise ValueError(f"Invalid search settings in {path}: {error}") from error
    return path.resolve(), protocol, settings


def normalized_seed_map(values: Mapping[str | int, Any]) -> Dict[str, str]:
    return {str(int(seed)): str(value) for seed, value in values.items()}


def validate_oof_lineage(
    search_protocol: Mapping[str, Any],
    architecture_input: stack.ArchitectureInput,
) -> None:
    architecture = architecture_input.architecture
    source = search_protocol.get("sources", {}).get(architecture)
    if not isinstance(source, dict):
        raise KeyError(f"Search protocol is missing {architecture} source lineage.")
    recorded_oof = source.get("oof_hashes")
    if not isinstance(recorded_oof, dict):
        raise KeyError(f"Search protocol is missing {architecture} OOF hashes.")
    for expert in stack.EXPERTS:
        observed = normalized_seed_map(architecture_input.source_hashes[expert])
        expected = normalized_seed_map(recorded_oof[expert])
        if observed != expected:
            raise ValueError(
                f"Current {architecture}:{expert} OOF artifacts differ from the "
                "ones used to construct the saved rankings."
            )
    if source.get("pointer_hashes") != architecture_input.pointer_hashes:
        raise ValueError(f"{architecture} source pointer hashes changed after search.")
    if source.get("protocol_hashes") != architecture_input.protocol_hashes:
        raise ValueError(f"{architecture} source protocol hashes changed after search.")


def ranking_path(results_dir: Path, architecture: str, stage: str) -> Path:
    return results_dir / (
        f"{SEARCH_PREFIX}_{SEARCH_EXPERIMENT_KEY}_{architecture}_{stage}_ranking.csv.gz"
    )


def validate_final_ranking_hash(
    results_dir: Path,
    architecture: str,
    final_path: Path,
    search_protocol_path: Path,
) -> None:
    record_path = results_dir / (
        f"{SEARCH_PREFIX}_{SEARCH_EXPERIMENT_KEY}_{architecture}_best_config.json"
    )
    record = read_json(record_path)
    expected = {
        "schema_version": stack.SCHEMA_VERSION,
        "experiment_key": SEARCH_EXPERIMENT_KEY,
        "architecture": architecture,
        "kddtest_accessed": False,
    }
    for name, value in expected.items():
        if record.get(name) != value:
            raise ValueError(
                f"{record_path} has {name}={record.get(name)!r}; expected {value!r}."
            )
    if record.get("protocol_sha256") != core.sha256_file(search_protocol_path):
        raise ValueError(f"{record_path} search-protocol hash does not match.")
    if record.get("final_cv_ranking_sha256") != core.sha256_file(final_path):
        raise ValueError(f"{final_path} hash does not match {record_path}.")


def validate_ranking(
    frame: pd.DataFrame,
    path: Path,
    architecture: str,
    stage: str,
) -> None:
    required = {
        "candidate_id",
        "family",
        "method_label",
        "fixed_expert",
        "calibration",
        "feature_set",
        "q",
        "C",
        "rho",
        "delta_r2l",
        "delta_u2r",
        "rare_f1_mean",
        "rare_f1_std",
        "robust_rare_f1_mean",
        "robust_rare_f1_std",
        "balanced_rare_f1_mean",
        "minimum_rare_f1_mean",
        "valid_all_seeds",
        *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing ranking columns: {missing}")
    if "stage" in frame and set(frame["stage"].astype(str)) != {stage}:
        raise ValueError(f"{path} contains a stage other than {stage}.")
    if "architecture" in frame and set(frame["architecture"].astype(str)) != {
        architecture
    }:
        raise ValueError(f"{path} contains another architecture.")
    if frame["candidate_id"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate candidate IDs.")
    if len(frame) != stack.CANONICAL_LIBRARY_COUNT:
        raise ValueError(
            f"{path} has {len(frame)} candidates; expected "
            f"{stack.CANONICAL_LIBRARY_COUNT}."
        )
    expected_family_counts = {
        "fixed": 4,
        "average_offset": len(stack.OFFSET_PAIRS) - 1,
        "stack": (
            len(stack.CALIBRATIONS)
            * len(stack.FEATURE_SETS)
            * len(stack.Q_VALUES)
            * len(stack.C_VALUES)
            * len(stack.NONZERO_RHO_VALUES)
            * len(stack.OFFSET_PAIRS)
        ),
    }
    observed_family_counts = frame["family"].astype(str).value_counts().to_dict()
    if observed_family_counts != expected_family_counts:
        raise ValueError(
            f"{path} family counts {observed_family_counts} do not match "
            f"{expected_family_counts}."
        )
    expected_fixed = {
        "fixed_general",
        "fixed_focal",
        "fixed_batching",
        "fixed_average",
    }
    observed_fixed = set(
        frame.loc[frame["family"].astype(str).eq("fixed"), "candidate_id"].astype(str)
    )
    if observed_fixed != expected_fixed:
        raise ValueError(f"{path} has unexpected fixed candidates: {observed_fixed}.")
    numeric_columns = [
        *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "std")],
        "robust_rare_f1_mean",
        "robust_rare_f1_std",
        "balanced_rare_f1_mean",
        "minimum_rare_f1_mean",
        "rho",
        "delta_r2l",
        "delta_u2r",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains a non-finite {column} value.")
        if column.endswith("_std") and np.any(values < 0.0):
            raise ValueError(f"{path} contains a negative {column} value.")
    stack_rows = frame[frame["family"].astype(str).eq("stack")]
    for column in ("q", "C"):
        values = pd.to_numeric(stack_rows[column], errors="raise").to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{path} contains a non-finite stack {column} value.")
    if np.any(pd.to_numeric(stack_rows["C"], errors="raise") <= 0.0):
        raise ValueError(f"{path} contains a nonpositive stack C value.")
    if np.any(pd.to_numeric(stack_rows["rho"], errors="raise") <= 0.0):
        raise ValueError(f"{path} contains a nonpositive stack rho value.")
    observed_calibrations = set(stack_rows["calibration"].astype(str))
    if observed_calibrations != set(stack.CALIBRATIONS):
        raise ValueError(f"{path} has unexpected stack calibrations.")
    observed_features = set(stack_rows["feature_set"].astype(str))
    if observed_features != set(stack.FEATURE_SETS):
        raise ValueError(f"{path} has unexpected stack feature sets.")
    for column, allowed in (
        ("q", stack.Q_VALUES),
        ("C", stack.C_VALUES),
        ("rho", stack.NONZERO_RHO_VALUES),
        ("delta_r2l", stack.DELTA_VALUES),
        ("delta_u2r", stack.DELTA_VALUES),
    ):
        observed = np.unique(
            pd.to_numeric(stack_rows[column], errors="raise").to_numpy(
                dtype=np.float64
            )
        )
        expected = np.asarray(allowed, dtype=np.float64)
        if observed.shape != expected.shape or not np.allclose(
            observed, np.sort(expected), atol=NUMERIC_TOLERANCE, rtol=0.0
        ):
            raise ValueError(f"{path} has unexpected stack {column} values.")


def true_fusion_mask(frame: pd.DataFrame) -> pd.Series:
    family = frame["family"].astype(str)
    candidate = frame["candidate_id"].astype(str)
    return family.isin(FUSION_FAMILIES) | candidate.eq("fixed_average")


def select_best_fusion(
    ranking: pd.DataFrame,
    standalone: Mapping[str, Mapping[str, float]],
    minimum_improvement: float,
) -> Dict[str, Any]:
    missing_methods = sorted(set(STANDALONE_METHODS) - set(standalone))
    if missing_methods:
        raise KeyError(f"Standalone metrics are missing: {missing_methods}")
    valid = bool_series(ranking["valid_all_seeds"], "valid_all_seeds")
    candidates = ranking.loc[true_fusion_mask(ranking) & valid].copy()
    if candidates.empty:
        raise RuntimeError("The ranking contains no technically valid fusion candidate.")
    candidates["_rare_sort"] = pd.to_numeric(
        candidates["rare_f1_mean"], errors="raise"
    )
    candidates = candidates.sort_values(
        ["_rare_sort", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns="_rare_sort")
    best = candidates.iloc[0]
    standalone_values = {
        method: float(standalone[method]["rare_f1_mean"])
        for method in STANDALONE_METHODS
    }
    if not all(np.isfinite(value) for value in standalone_values.values()):
        raise ValueError("A standalone Rare-F1 value is non-finite.")
    best_method = max(
        STANDALONE_METHODS,
        key=lambda name: (round(standalone_values[name], COMPARISON_DECIMALS), -STANDALONE_METHODS.index(name)),
    )
    best_standalone = standalone_values[best_method]
    threshold = best_standalone + float(minimum_improvement)
    fusion_value = float(best["rare_f1_mean"])
    passes = passes_minimum_gain(
        fusion_value,
        best_standalone,
        minimum_improvement,
    )
    return {
        "candidate": best,
        "candidate_count": int(len(candidates)),
        "best_standalone_method": best_method,
        "best_standalone_rare_f1": best_standalone,
        "required_rare_f1": threshold,
        "fusion_rare_f1": fusion_value,
        "improvement_over_best": fusion_value - best_standalone,
        "passes": bool(passes),
    }


def passes_minimum_gain(
    fusion_rare_f1: float,
    best_standalone_rare_f1: float,
    minimum_improvement: float,
) -> bool:
    values = np.asarray(
        [fusion_rare_f1, best_standalone_rare_f1, minimum_improvement],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or minimum_improvement < 0.0:
        raise ValueError("Rare-F1 values and minimum improvement must be finite.")
    return bool(
        float(fusion_rare_f1)
        >= float(best_standalone_rare_f1)
        + float(minimum_improvement)
        - NUMERIC_TOLERANCE
    )


def nested_acceptance(
    final_all_oof_pass: bool,
    outer_inner_pass_count: int,
    fold_count: int,
    nested_fusion_rare_f1: float,
    nested_best_standalone_rare_f1: float,
    minimum_improvement: float,
) -> Dict[str, bool]:
    if fold_count <= 0 or not 0 <= outer_inner_pass_count <= fold_count:
        raise ValueError("Outer-fold pass count is outside its valid range.")
    pooled_pass = passes_minimum_gain(
        nested_fusion_rare_f1,
        nested_best_standalone_rare_f1,
        minimum_improvement,
    )
    all_outer_pass = outer_inner_pass_count == fold_count
    nested_pass = all_outer_pass and pooled_pass
    return {
        "pooled_performance_pass": pooled_pass,
        "all_outer_inner_selections_pass": all_outer_pass,
        "nested_audit_pass": nested_pass,
        "advance_to_test": bool(final_all_oof_pass and nested_pass),
    }


def validate_fixed_ranking_rows(
    ranking: pd.DataFrame,
    standalone: Mapping[str, Mapping[str, float]],
    path: Path,
) -> None:
    references = {**{expert: expert for expert in stack.EXPERTS}, "average": "average"}
    for fixed_name, metric_name in references.items():
        candidate_id = f"fixed_{fixed_name}"
        matches = ranking[ranking["candidate_id"].astype(str).eq(candidate_id)]
        if len(matches) != 1:
            raise ValueError(f"{path} must contain exactly one {candidate_id} row.")
        row = matches.iloc[0]
        for metric in METRICS:
            for suffix in ("mean", "std"):
                name = f"{metric}_{suffix}"
                observed = float(row[name])
                expected = float(standalone[metric_name][name])
                if not np.isclose(observed, expected, atol=1e-12, rtol=0.0):
                    raise ValueError(
                        f"{path} {candidate_id} {name}={observed} does not match "
                        f"the loaded OOF value {expected}."
                    )


def aggregate_prediction_metrics(
    labels: np.ndarray,
    predictions: Mapping[int, np.ndarray],
    seeds: Sequence[int],
) -> Dict[str, float]:
    rows: list[Dict[str, float]] = []
    labels = np.asarray(labels, dtype=np.int64)
    for seed in seeds:
        values = np.asarray(predictions[int(seed)], dtype=np.int64)
        if values.shape != labels.shape:
            raise ValueError(f"Seed {seed} prediction shape {values.shape} != {labels.shape}.")
        if np.any(values < 0) or np.any(values >= stack.CLASS_COUNT):
            raise ValueError(f"Seed {seed} predictions contain an invalid class.")
        rows.append(core.calculate_metrics(labels, values))
    output: Dict[str, float] = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )
    return output


def standalone_metrics(
    architecture_input: stack.ArchitectureInput,
    indices: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    indices = np.asarray(indices, dtype=np.int64)
    labels = architecture_input.labels[indices]
    output: Dict[str, Dict[str, float]] = {}
    for expert_index, expert in enumerate(stack.EXPERTS):
        predictions = {
            seed: np.argmax(
                architecture_input.probabilities[seed_index, expert_index, indices],
                axis=1,
            ).astype(np.int64)
            for seed_index, seed in enumerate(architecture_input.seeds)
        }
        output[expert] = aggregate_prediction_metrics(
            labels, predictions, architecture_input.seeds
        )
    average_predictions = {
        seed: np.argmax(
            architecture_input.probabilities[seed_index][:, indices, :].mean(
                axis=0, dtype=np.float64
            ),
            axis=1,
        ).astype(np.int64)
        for seed_index, seed in enumerate(architecture_input.seeds)
    }
    output["average"] = aggregate_prediction_metrics(
        labels,
        average_predictions,
        architecture_input.seeds,
    )
    return output


def scaling_ranking(
    architecture_input: stack.ArchitectureInput,
    indices: np.ndarray,
    coefficients: Sequence[float],
    chunk_size: int,
) -> pd.DataFrame:
    indices = np.asarray(indices, dtype=np.int64)
    labels = architecture_input.labels[indices]
    values = np.asarray(coefficients, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) == 0
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or len(np.unique(values)) != len(values)
    ):
        raise ValueError("Scaling coefficients must be unique, finite, and positive.")
    pair_r2l = np.repeat(values, len(values))
    pair_u2r = np.tile(values, len(values))
    per_seed: list[pd.DataFrame] = []
    general_index = stack.EXPERTS.index("general")
    for seed_index, seed in enumerate(architecture_input.seeds):
        probabilities = architecture_input.probabilities[
            seed_index, general_index, indices
        ]
        confusions = scaling.score_pair_confusions(
            labels,
            probabilities,
            pair_r2l,
            pair_u2r,
            chunk_size,
        )
        frame = scaling.metrics_from_confusions(confusions)
        frame.insert(0, "u2r_score_coefficient", pair_u2r)
        frame.insert(0, "r2l_score_coefficient", pair_r2l)
        frame.insert(0, "seed", int(seed))
        per_seed.append(frame)
    runs = pd.concat(per_seed, ignore_index=True)
    rows: list[Dict[str, Any]] = []
    for (r2l_value, u2r_value), group in runs.groupby(
        ["r2l_score_coefficient", "u2r_score_coefficient"], sort=True
    ):
        row: Dict[str, Any] = {
            "r2l_score_coefficient": float(r2l_value),
            "u2r_score_coefficient": float(u2r_value),
            "scaling_log_distance": float(
                abs(np.log(float(r2l_value))) + abs(np.log(float(u2r_value)))
            ),
        }
        for metric in METRICS:
            metric_values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(metric_values.mean())
            row[f"{metric}_std"] = (
                float(metric_values.std(ddof=1)) if len(metric_values) > 1 else 0.0
            )
        rows.append(row)
    result = pd.DataFrame(rows)
    return (
        result.sort_values(
            [
                "rare_f1_mean",
                "macro_f1_mean",
                "scaling_log_distance",
                "rare_f1_std",
                "r2l_score_coefficient",
                "u2r_score_coefficient",
            ],
            ascending=[False, False, True, True, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def scaled_predictions(
    probabilities: np.ndarray,
    r2l_coefficient: float,
    u2r_coefficient: float,
) -> np.ndarray:
    return core.apply_class_score_scaling(
        probabilities,
        {
            stack.R2L_CLASS: float(r2l_coefficient),
            stack.U2R_CLASS: float(u2r_coefficient),
        },
    )


def load_scaling_source(
    repo_root: Path,
    results_dir: Path,
    architecture: str,
    template: str,
    search_protocol: Mapping[str, Any],
) -> ScalingSource:
    pointer_path = stack.resolve_template(repo_root, template, architecture).resolve()
    pointer = read_json(pointer_path)
    if pointer.get("architecture") != architecture:
        raise ValueError(f"{pointer_path} has the wrong architecture.")
    required_pointer = ("protocol", "ranking", "baseline_summary_or_best_scaling")
    missing = [name for name in required_pointer if name not in pointer]
    if missing:
        raise KeyError(f"{pointer_path} is missing fields: {missing}")
    protocol_path = resolve_recorded_path(pointer["protocol"], pointer_path, results_dir)
    ranking_path_value = resolve_recorded_path(
        pointer["ranking"], pointer_path, results_dir
    )
    best_path = resolve_recorded_path(
        pointer["baseline_summary_or_best_scaling"], pointer_path, results_dir
    )
    protocol = read_json(protocol_path)
    best = read_json(best_path)
    if protocol.get("kddtest_accessed") is not False:
        raise ValueError(f"{protocol_path} is not validation-only.")
    training_settings = protocol.get("training_settings")
    if not isinstance(training_settings, dict) or training_settings.get(
        "training_mode"
    ) != "baseline_ce":
        raise ValueError(f"{protocol_path} is not the cross-entropy baseline search.")
    if best.get("architecture") != architecture or best.get("training_mode") != "baseline_ce":
        raise ValueError(f"{best_path} is not {architecture} baseline score scaling.")
    coefficients = tuple(float(value) for value in best.get("coefficient_values", []))
    if len(coefficients) <= 1 or len(coefficients) ** 2 != int(best.get("pair_count", -1)):
        raise ValueError(f"{best_path} does not contain a score-scaling grid.")
    if sorted(int(seed) for seed in best.get("training_seeds", [])) != list(SEEDS):
        raise ValueError(f"{best_path} does not use seeds {list(SEEDS)}.")
    if int(best.get("fold_count", -1)) != len(FOLDS):
        raise ValueError(f"{best_path} does not use {len(FOLDS)} folds.")
    for key in ("training_key", "scoring_key"):
        identities = {str(pointer.get(key, "")), str(protocol.get(key, "")), str(best.get(key, ""))}
        if "" in identities or len(identities) != 1:
            raise ValueError(
                f"{pointer_path}, {protocol_path}, and {best_path} disagree on {key}."
            )
    scoring_settings = protocol.get("scoring_settings")
    if not isinstance(scoring_settings, dict):
        raise KeyError(f"{protocol_path} is missing scoring settings.")
    protocol_coefficients = tuple(
        float(value) for value in scoring_settings.get("coefficient_values", [])
    )
    if protocol_coefficients != coefficients:
        raise ValueError(f"{best_path} coefficient grid differs from {protocol_path}.")
    if int(scoring_settings.get("pair_count", -1)) != len(coefficients) ** 2:
        raise ValueError(f"{protocol_path} has an inconsistent scaling pair count.")
    source = search_protocol.get("sources", {}).get(architecture)
    if not isinstance(source, dict):
        raise KeyError(f"Search protocol is missing {architecture} source lineage.")
    pointer_hash = core.sha256_file(pointer_path)
    protocol_hash = core.sha256_file(protocol_path)
    if source.get("pointer_hashes", {}).get("general") != pointer_hash:
        raise ValueError(
            f"{pointer_path} is not the General pointer frozen by the search."
        )
    if source.get("protocol_hashes", {}).get("general") != protocol_hash:
        raise ValueError(
            f"{protocol_path} is not the General protocol frozen by the search."
        )
    recorded_ranking = resolve_recorded_path(best["ranking_path"], best_path, results_dir)
    if recorded_ranking != ranking_path_value:
        raise ValueError(f"Scaling ranking mismatch between {pointer_path} and {best_path}.")
    return ScalingSource(
        pointer_path=pointer_path,
        pointer_sha256=pointer_hash,
        protocol_path=protocol_path,
        protocol_sha256=protocol_hash,
        ranking_path=ranking_path_value,
        ranking_sha256=core.sha256_file(ranking_path_value),
        best_path=best_path,
        best_sha256=core.sha256_file(best_path),
        coefficients=coefficients,
        stored_best=best,
    )


def validate_stored_scaling_winner(
    source: ScalingSource,
    selected: pd.Series,
) -> None:
    expected = source.stored_best
    comparisons = {
        "r2l_score_coefficient": float(selected["r2l_score_coefficient"]),
        "u2r_score_coefficient": float(selected["u2r_score_coefficient"]),
    }
    for name, observed in comparisons.items():
        if not np.isclose(
            observed, float(expected[name]), atol=1e-12, rtol=0.0
        ):
            raise ValueError(
                f"Recomputed scaling winner has {name}={observed}; "
                f"stored winner has {expected[name]}."
            )
    stored_rare = float(expected["metrics"]["rare_f1"]["mean"])
    if not np.isclose(
        float(selected["rare_f1_mean"]), stored_rare, atol=1e-12, rtol=0.0
    ):
        raise ValueError("Recomputed score-scaling Rare F1 disagrees with its artifact.")


def baseline_metrics_with_scaling(
    architecture_input: stack.ArchitectureInput,
    indices: np.ndarray,
    scaling_row: pd.Series,
) -> Dict[str, Dict[str, float]]:
    output = standalone_metrics(architecture_input, indices)
    predictions: Dict[int, np.ndarray] = {}
    general_index = stack.EXPERTS.index("general")
    for seed_index, seed in enumerate(architecture_input.seeds):
        probabilities = architecture_input.probabilities[
            seed_index, general_index, indices
        ]
        predictions[seed] = scaled_predictions(
            probabilities,
            float(scaling_row["r2l_score_coefficient"]),
            float(scaling_row["u2r_score_coefficient"]),
        )
    output["scaling"] = aggregate_prediction_metrics(
        architecture_input.labels[indices], predictions, architecture_input.seeds
    )
    return output


def outer_split_indices(
    fold_ids: np.ndarray,
    outer_fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(fold_ids, dtype=np.int64)
    if outer_fold not in FOLDS:
        raise ValueError(f"Outer fold must be one of {list(FOLDS)}.")
    train_indices = np.flatnonzero(values != outer_fold).astype(np.int64)
    heldout_indices = np.flatnonzero(values == outer_fold).astype(np.int64)
    if len(train_indices) == 0 or len(heldout_indices) == 0:
        raise ValueError(f"Outer fold {outer_fold} produces an empty partition.")
    if np.intersect1d(train_indices, heldout_indices).size:
        raise RuntimeError("Outer training and held-out indices overlap.")
    if not np.array_equal(
        np.sort(np.concatenate((train_indices, heldout_indices))),
        np.arange(len(values), dtype=np.int64),
    ):
        raise RuntimeError("Outer training and held-out indices do not cover all rows.")
    return train_indices, heldout_indices


def refit_outer_candidate(
    architecture_input: stack.ArchitectureInput,
    config: Mapping[str, Any],
    outer_fold: int,
    settings: stack.SearchSettings,
) -> tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    train_indices, heldout_indices = outer_split_indices(
        architecture_input.fold_ids,
        outer_fold,
    )
    predictions: Dict[int, np.ndarray] = {}
    for seed in architecture_input.seeds:
        values, _probabilities, _state = stack.fit_selected_candidate(
            architecture_input,
            seed,
            config,
            train_indices,
            heldout_indices,
            settings,
        )
        values = np.asarray(values, dtype=np.int64)
        if values.shape != heldout_indices.shape:
            raise RuntimeError(
                f"{architecture_input.architecture}:s{seed}:outer_{outer_fold} "
                "returned the wrong prediction shape."
            )
        predictions[int(seed)] = values
    return train_indices, heldout_indices, predictions


def selection_row(
    architecture: str,
    stage: str,
    result: Mapping[str, Any],
    scaling_row: pd.Series,
    ranking_file: Path,
) -> Dict[str, Any]:
    candidate = result["candidate"]
    previous_eligible: bool | None = None
    if "eligible" in candidate.index:
        previous_eligible = bool_series(
            pd.Series([candidate["eligible"]]), "eligible"
        ).iloc[0]
    row: Dict[str, Any] = {
        "architecture": architecture,
        "model": stack.ARCHITECTURE_LABELS[architecture],
        "stage": stage,
        "candidate_id": str(candidate["candidate_id"]),
        "family": str(candidate["family"]),
        "calibration": str(candidate["calibration"]),
        "feature_set": str(candidate["feature_set"]),
        "q": None if pd.isna(candidate["q"]) else float(candidate["q"]),
        "C": None if pd.isna(candidate["C"]) else float(candidate["C"]),
        "rho": float(candidate["rho"]),
        "delta_r2l": float(candidate["delta_r2l"]),
        "delta_u2r": float(candidate["delta_u2r"]),
        "fusion_candidates_considered": int(result["candidate_count"]),
        "best_standalone_method": str(result["best_standalone_method"]),
        "best_standalone_label": METHOD_LABELS[str(result["best_standalone_method"])],
        "best_standalone_rare_f1": float(result["best_standalone_rare_f1"]),
        "minimum_improvement": float(
            result["required_rare_f1"] - result["best_standalone_rare_f1"]
        ),
        "required_rare_f1": float(result["required_rare_f1"]),
        "fusion_rare_f1": float(result["fusion_rare_f1"]),
        "improvement_over_best": float(result["improvement_over_best"]),
        "eligibility_pass": bool(result["passes"]),
        "scaling_r2l_coefficient": float(scaling_row["r2l_score_coefficient"]),
        "scaling_u2r_coefficient": float(scaling_row["u2r_score_coefficient"]),
        "ranking_path": str(ranking_file),
        "ranking_sha256": core.sha256_file(ranking_file),
        "previous_search_eligible_diagnostic": previous_eligible,
    }
    for metric in METRICS:
        row[f"fusion_{metric}_mean"] = float(candidate[f"{metric}_mean"])
        row[f"fusion_{metric}_std"] = float(candidate[f"{metric}_std"])
    return row


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--train-data", default="data/KDDTrain+.txt")
    parser.add_argument("--results-dir", default="results")
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
    parser.add_argument(
        "--scaling-template",
        default="results/{architecture}_baseline_cv_latest.json",
    )
    parser.add_argument("--minimum-improvement", type=float, default=0.005)
    parser.add_argument("--score-chunk-size", type=int, default=20000)
    parser.add_argument(
        "--output-prefix",
        default="validation_fusion_vs_all_standalone",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate all artifacts and print inner/final selections without "
            "refitting selected candidates or writing outputs."
        ),
    )
    args = parser.parse_args(argv)
    args.architectures = list(dict.fromkeys(str(value) for value in args.architectures))
    if len(args.architectures) == 0:
        parser.error("At least one architecture is required.")
    if sorted(args.seeds) != list(SEEDS) or len(args.seeds) != len(set(args.seeds)):
        parser.error("This frozen experiment requires exactly seeds 0 1 2.")
    if not np.isfinite(args.minimum_improvement) or args.minimum_improvement < 0.0:
        parser.error("--minimum-improvement must be finite and nonnegative.")
    if args.score_chunk_size <= 0:
        parser.error("--score-chunk-size must be positive.")
    if (
        not args.output_prefix.strip()
        or Path(args.output_prefix).name != args.output_prefix
    ):
        parser.error("--output-prefix must be a nonempty filename-safe value.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    started = time.perf_counter()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    train_path = stack.safe.resolve_cli_path(repo_root, args.train_data)
    results_dir = stack.safe.resolve_cli_path(repo_root, args.results_dir)
    if not train_path.is_file():
        raise SystemExit(f"KDDTrain+ does not exist: {train_path}")
    protocol_path, search_protocol, search_settings = load_search_protocol(results_dir)
    architectures = list(args.architectures)
    seeds = tuple(sorted(int(seed) for seed in args.seeds))

    print("Loading and validating saved OOF inputs...", flush=True)
    inputs = [
        stack.load_architecture_input(
            repo_root,
            train_path,
            architecture,
            seeds,
            args.general_template,
            args.focal_template,
            args.batching_template,
        )
        for architecture in architectures
    ]
    stack.validate_cross_architecture_alignment(inputs)
    input_by_architecture = {item.architecture: item for item in inputs}
    for architecture_input in inputs:
        validate_oof_lineage(search_protocol, architecture_input)
    scaling_sources = {
        architecture: load_scaling_source(
            repo_root,
            results_dir,
            architecture,
            args.scaling_template,
            search_protocol,
        )
        for architecture in architectures
    }

    print("Validation-only fusion-versus-standalone selection")
    print(f"Architectures: {architectures}")
    print(f"Seeds: {list(seeds)}; outer folds: {list(FOLDS)}")
    print(f"Rare-F1 minimum improvement: {100.0 * args.minimum_improvement:.2f} points")
    print("Fusion eligibility/ranking metric: Rare F1 only")
    print("Macro-F1/MCC guards: NOT USED")
    print("Backbone retraining: NO")
    print("KDDTest+ accessed: NO")

    final_rows: list[Dict[str, Any]] = []
    outer_rows: list[Dict[str, Any]] = []
    nested_per_seed_rows: list[Dict[str, Any]] = []
    nested_summary_rows: list[Dict[str, Any]] = []
    frozen_configs: Dict[str, Any] = {}
    output_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    with threadpool_limits(limits=1):
        for architecture in architectures:
            architecture_input = input_by_architecture[architecture]
            source = scaling_sources[architecture]
            all_indices = np.arange(len(architecture_input.labels), dtype=np.int64)
            final_scaling = scaling_ranking(
                architecture_input,
                all_indices,
                source.coefficients,
                args.score_chunk_size,
            ).iloc[0]
            validate_stored_scaling_winner(source, final_scaling)
            final_baselines = baseline_metrics_with_scaling(
                architecture_input, all_indices, final_scaling
            )
            final_path = ranking_path(results_dir, architecture, "final_cv")
            validate_final_ranking_hash(
                results_dir,
                architecture,
                final_path,
                protocol_path,
            )
            final_ranking = pd.read_csv(final_path)
            validate_ranking(final_ranking, final_path, architecture, "final_cv")
            validate_fixed_ranking_rows(final_ranking, final_baselines, final_path)
            final_result = select_best_fusion(
                final_ranking,
                final_baselines,
                args.minimum_improvement,
            )
            final_record = selection_row(
                architecture,
                "final_cv",
                final_result,
                final_scaling,
                final_path,
            )
            for method in STANDALONE_METHODS:
                final_record[f"{method}_rare_f1_mean"] = float(
                    final_baselines[method]["rare_f1_mean"]
                )
                final_record[f"{method}_rare_f1_std"] = float(
                    final_baselines[method]["rare_f1_std"]
                )
            final_rows.append(final_record)
            config = stack.candidate_config(final_result["candidate"])
            frozen_configs[architecture] = {
                "status": "pending_nested_audit",
                "all_oof_eligibility_pass": bool(final_result["passes"]),
                "minimum_improvement": float(args.minimum_improvement),
                "best_standalone_method": final_result["best_standalone_method"],
                "best_standalone_rare_f1": float(
                    final_result["best_standalone_rare_f1"]
                ),
                "required_rare_f1": float(final_result["required_rare_f1"]),
                "best_fusion_config": config,
                "best_fusion_rare_f1": float(final_result["fusion_rare_f1"]),
                "improvement_over_best": float(
                    final_result["improvement_over_best"]
                ),
                "frozen_config": None,
            }
            print(
                f"  {stack.ARCHITECTURE_LABELS[architecture]} final: "
                f"{config['candidate_id']} Rare F1={100.0 * final_result['fusion_rare_f1']:.2f}%; "
                f"required={100.0 * final_result['required_rare_f1']:.2f}%; "
                f"{'PASS' if final_result['passes'] else 'FAIL'}",
                flush=True,
            )

            selected_predictions = {
                seed: np.full(len(architecture_input.labels), -1, dtype=np.int64)
                for seed in seeds
            }
            baseline_predictions = {
                method: {
                    seed: np.full(len(architecture_input.labels), -1, dtype=np.int64)
                    for seed in seeds
                }
                for method in STANDALONE_METHODS
            }
            for outer_fold in FOLDS:
                train_indices, heldout_indices = outer_split_indices(
                    architecture_input.fold_ids,
                    outer_fold,
                )
                inner_scaling = scaling_ranking(
                    architecture_input,
                    train_indices,
                    source.coefficients,
                    args.score_chunk_size,
                ).iloc[0]
                inner_baselines = baseline_metrics_with_scaling(
                    architecture_input, train_indices, inner_scaling
                )
                stage = f"outer_{outer_fold}"
                outer_path = ranking_path(results_dir, architecture, stage)
                outer_ranking = pd.read_csv(outer_path)
                validate_ranking(outer_ranking, outer_path, architecture, stage)
                validate_fixed_ranking_rows(
                    outer_ranking,
                    inner_baselines,
                    outer_path,
                )
                outer_result = select_best_fusion(
                    outer_ranking,
                    inner_baselines,
                    args.minimum_improvement,
                )
                outer_config = stack.candidate_config(outer_result["candidate"])
                record = selection_row(
                    architecture,
                    stage,
                    outer_result,
                    inner_scaling,
                    outer_path,
                )
                record["outer_fold"] = int(outer_fold)
                record["inner_rows"] = int(len(train_indices))
                record["heldout_rows"] = int(len(heldout_indices))

                if not args.dry_run:
                    audit_train, audit_heldout, fold_fusion_predictions = (
                        refit_outer_candidate(
                            architecture_input,
                            outer_config,
                            outer_fold,
                            search_settings,
                        )
                    )
                    if not np.array_equal(audit_train, train_indices) or not np.array_equal(
                        audit_heldout, heldout_indices
                    ):
                        raise RuntimeError("Outer-refit partition audit failed.")
                    for seed_index, seed in enumerate(seeds):
                        selected_predictions[seed][heldout_indices] = (
                            fold_fusion_predictions[seed]
                        )
                        for expert_index, expert in enumerate(stack.EXPERTS):
                            baseline_predictions[expert][seed][heldout_indices] = np.argmax(
                                architecture_input.probabilities[
                                    seed_index, expert_index, heldout_indices
                                ],
                                axis=1,
                            ).astype(np.int64)
                        general_probabilities = architecture_input.probabilities[
                            seed_index,
                            stack.EXPERTS.index("general"),
                            heldout_indices,
                        ]
                        baseline_predictions["scaling"][seed][heldout_indices] = (
                            scaled_predictions(
                                general_probabilities,
                                float(inner_scaling["r2l_score_coefficient"]),
                                float(inner_scaling["u2r_score_coefficient"]),
                            )
                        )
                    heldout_labels = architecture_input.labels[heldout_indices]
                    heldout_fusion = aggregate_prediction_metrics(
                        heldout_labels,
                        {
                            seed: selected_predictions[seed][heldout_indices]
                            for seed in seeds
                        },
                        seeds,
                    )
                    heldout_baselines = {
                        method: aggregate_prediction_metrics(
                            heldout_labels,
                            {
                                seed: baseline_predictions[method][seed][heldout_indices]
                                for seed in seeds
                            },
                            seeds,
                        )
                        for method in STANDALONE_METHODS
                    }
                    heldout_best_method = max(
                        STANDALONE_METHODS,
                        key=lambda name: (
                            round(
                                heldout_baselines[name]["rare_f1_mean"],
                                COMPARISON_DECIMALS,
                            ),
                            -STANDALONE_METHODS.index(name),
                        ),
                    )
                    heldout_best = heldout_baselines[heldout_best_method][
                        "rare_f1_mean"
                    ]
                    heldout_pass = passes_minimum_gain(
                        heldout_fusion["rare_f1_mean"],
                        heldout_best,
                        args.minimum_improvement,
                    )
                    record.update(
                        {
                            "heldout_fusion_rare_f1": heldout_fusion[
                                "rare_f1_mean"
                            ],
                            "heldout_best_standalone_method": heldout_best_method,
                            "heldout_best_standalone_rare_f1": heldout_best,
                            "heldout_improvement_over_best": heldout_fusion[
                                "rare_f1_mean"
                            ]
                            - heldout_best,
                            "heldout_required_rare_f1": heldout_best
                            + args.minimum_improvement,
                            "heldout_pass": heldout_pass,
                        }
                    )
                outer_rows.append(record)
                print(
                    f"    {stage}: {outer_config['candidate_id']}; inner "
                    f"{'PASS' if outer_result['passes'] else 'FAIL'}",
                    flush=True,
                )

            if args.dry_run:
                continue
            architecture_outer_rows = sorted(
                (
                    row
                    for row in outer_rows
                    if row["architecture"] == architecture
                ),
                key=lambda row: int(row["outer_fold"]),
            )
            if len(architecture_outer_rows) != len(FOLDS):
                raise RuntimeError(f"{architecture} is missing an outer selection row.")
            prediction_methods = {
                **baseline_predictions,
                "selected_fusion": selected_predictions,
            }
            output_predictions[architecture] = {
                "row_indices": np.arange(
                    len(architecture_input.labels), dtype=np.int64
                ),
                "labels": architecture_input.labels,
                "fold_ids": architecture_input.fold_ids,
                "seeds": np.asarray(seeds, dtype=np.int64),
                "outer_folds": np.asarray(FOLDS, dtype=np.int64),
                "outer_candidate_ids": np.asarray(
                    [row["candidate_id"] for row in architecture_outer_rows],
                    dtype="U96",
                ),
                "outer_inner_eligibility_pass": np.asarray(
                    [row["eligibility_pass"] for row in architecture_outer_rows],
                    dtype=bool,
                ),
                "outer_scaling_r2l_coefficients": np.asarray(
                    [
                        row["scaling_r2l_coefficient"]
                        for row in architecture_outer_rows
                    ],
                    dtype=np.float64,
                ),
                "outer_scaling_u2r_coefficients": np.asarray(
                    [
                        row["scaling_u2r_coefficient"]
                        for row in architecture_outer_rows
                    ],
                    dtype=np.float64,
                ),
                **{
                    f"{method}_predictions": np.stack(
                        [prediction_methods[method][seed] for seed in seeds], axis=0
                    )
                    for method in (*STANDALONE_METHODS, "selected_fusion")
                },
            }
            method_summaries: Dict[str, Dict[str, float]] = {}
            for method, predictions in prediction_methods.items():
                for seed in seeds:
                    metrics = core.calculate_metrics(
                        architecture_input.labels, predictions[seed]
                    )
                    nested_per_seed_rows.append(
                        {
                            "architecture": architecture,
                            "model": stack.ARCHITECTURE_LABELS[architecture],
                            "method": method,
                            "method_label": METHOD_LABELS[method],
                            "seed": int(seed),
                            **metrics,
                        }
                    )
                summary = aggregate_prediction_metrics(
                    architecture_input.labels, predictions, seeds
                )
                method_summaries[method] = summary
                nested_summary_rows.append(
                    {
                        "architecture": architecture,
                        "model": stack.ARCHITECTURE_LABELS[architecture],
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "runs": len(seeds),
                        "seeds": ",".join(str(seed) for seed in seeds),
                        **summary,
                    }
                )
            nested_best_method = max(
                STANDALONE_METHODS,
                key=lambda name: (
                    round(
                        method_summaries[name]["rare_f1_mean"],
                        COMPARISON_DECIMALS,
                    ),
                    -STANDALONE_METHODS.index(name),
                ),
            )
            nested_best = method_summaries[nested_best_method]["rare_f1_mean"]
            nested_fusion = method_summaries["selected_fusion"]["rare_f1_mean"]
            inner_pass_count = int(
                sum(
                    bool(row["eligibility_pass"])
                    for row in outer_rows
                    if row["architecture"] == architecture
                )
            )
            acceptance = nested_acceptance(
                bool(final_result["passes"]),
                inner_pass_count,
                len(FOLDS),
                nested_fusion,
                nested_best,
                args.minimum_improvement,
            )
            pooled_pass = acceptance["pooled_performance_pass"]
            nested_pass = acceptance["nested_audit_pass"]
            nested_audit = {
                "best_standalone_method": nested_best_method,
                "best_standalone_rare_f1": nested_best,
                "selected_procedure_rare_f1": nested_fusion,
                "improvement_over_best": nested_fusion - nested_best,
                "required_rare_f1": nested_best + args.minimum_improvement,
                "pooled_performance_pass": bool(pooled_pass),
                "outer_inner_eligibility_passes": inner_pass_count,
                "all_outer_inner_selections_pass": inner_pass_count == len(FOLDS),
                "pass": bool(nested_pass),
            }
            frozen_configs[architecture]["nested_audit"] = nested_audit
            advance = acceptance["advance_to_test"]
            frozen_configs[architecture]["advance_to_test"] = advance
            frozen_configs[architecture]["status"] = (
                "selected" if advance else "validation_rule_failed"
            )
            frozen_configs[architecture]["frozen_config"] = config if advance else None
            final_record.update(
                {
                    "nested_best_standalone_method": nested_best_method,
                    "nested_best_standalone_rare_f1": nested_best,
                    "nested_fusion_rare_f1": nested_fusion,
                    "nested_improvement_over_best": nested_fusion - nested_best,
                    "nested_required_rare_f1": nested_best
                    + args.minimum_improvement,
                    "nested_pooled_performance_pass": bool(pooled_pass),
                    "nested_audit_pass": bool(nested_pass),
                    "outer_inner_eligibility_passes": inner_pass_count,
                    "advance_to_test": advance,
                }
            )

    if args.dry_run:
        print(
            "Dry run complete; rankings, source lineage, scaling grids, and all "
            "inner/final selections are valid. No candidate was refitted, no "
            "output was written, and KDDTest+ was not accessed."
        )
        return

    definition = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "validation-only fusion superiority selection and nested audit",
        "search_experiment_key": SEARCH_EXPERIMENT_KEY,
        "architectures": architectures,
        "seeds": list(seeds),
        "folds": list(FOLDS),
        "standalone_methods": list(STANDALONE_METHODS),
        "fusion_families": sorted(FUSION_FAMILIES),
        "fixed_average_is_fusion": True,
        "eligibility_metric": "mean Rare F1 across seeds",
        "minimum_improvement": float(args.minimum_improvement),
        "eligibility_rule": (
            "fusion Rare F1 >= max(General,Focal,Batching,Scaling) Rare F1 "
            "+ minimum_improvement"
        ),
        "ranking_rule": "highest Rare F1; candidate ID only for an exact tie",
        "scaling_comparator_ranking_rule": (
            "its frozen validation protocol: Rare F1, Macro-F1 tie-break, "
            "distance to (1,1), Rare-F1 SD, coefficient values"
        ),
        "macro_f1_guard_used": False,
        "mcc_guard_used": False,
        "old_search_eligible_column_used": False,
        "technical_validity_required": "valid_all_seeds only",
        "nested_protocol": (
            "select score scaling and fusion on the three non-outer folds; refit "
            "the selected fusion on those folds per seed; evaluate the held-out fold"
        ),
        "claim_scope": (
            "nested meta-level validation conditional on frozen cross-fitted expert "
            "probabilities; not fully nested end-to-end because base experts and "
            "their training-time settings are not retrained inside meta outer folds"
        ),
        "nested_failure_policy": (
            "evaluate an unqualified stage-best fusion for diagnostics, mark the "
            "outer stage failed, and never relax the minimum improvement"
        ),
        "advance_to_test_rule": (
            "final all-OOF eligibility passes, all four outer inner selections "
            "pass, and pooled nested Rare F1 clears the same standalone margin"
        ),
        "final_selection": "one all-OOF final_cv candidate per architecture",
        "kddtest_accessed": False,
        "backbone_models_retrained": False,
        "script_sha256": core.sha256_file(script_path),
        "search_protocol": str(protocol_path),
        "search_protocol_sha256": core.sha256_file(protocol_path),
        "search_settings": asdict(search_settings),
        "scaling_sources": {
            architecture: {
                "pointer": str(source.pointer_path),
                "pointer_sha256": source.pointer_sha256,
                "protocol": str(source.protocol_path),
                "protocol_sha256": source.protocol_sha256,
                "ranking": str(source.ranking_path),
                "ranking_sha256": source.ranking_sha256,
                "best": str(source.best_path),
                "best_sha256": source.best_sha256,
                "coefficients": list(source.coefficients),
            }
            for architecture, source in scaling_sources.items()
        },
        "final_selections": frozen_configs,
        "all_architectures_advance_to_test": all(
            bool(frozen_configs[architecture]["advance_to_test"])
            for architecture in architectures
        ),
    }
    experiment_key = stable_hash(definition)
    stem = f"{args.output_prefix}_{experiment_key}"
    final_path = results_dir / f"{stem}_final_selections.csv"
    outer_path = results_dir / f"{stem}_outer_selections.csv"
    per_seed_path = results_dir / f"{stem}_nested_per_seed.csv"
    summary_path = results_dir / f"{stem}_nested_summary.csv"
    config_path = results_dir / f"{stem}_frozen_configs.json"
    predictions_path = results_dir / f"{stem}_nested_predictions.npz"
    protocol_output_path = results_dir / f"{stem}_protocol.json"
    latest_suffix = (
        ""
        if architectures == list(ARCHITECTURES)
        else "_" + "_".join(architectures)
    )
    latest_path = results_dir / f"{args.output_prefix}{latest_suffix}_latest.json"

    core.atomic_csv(final_path, pd.DataFrame(final_rows))
    core.atomic_csv(outer_path, pd.DataFrame(outer_rows))
    core.atomic_csv(per_seed_path, pd.DataFrame(nested_per_seed_rows))
    core.atomic_csv(summary_path, pd.DataFrame(nested_summary_rows))
    core.atomic_json(
        config_path,
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_key": experiment_key,
            "minimum_improvement": float(args.minimum_improvement),
            "architectures": frozen_configs,
            "all_architectures_advance_to_test": all(
                bool(frozen_configs[architecture]["advance_to_test"])
                for architecture in architectures
            ),
            "kddtest_accessed": False,
        },
    )
    flat_predictions: Dict[str, np.ndarray] = {}
    for architecture, arrays in output_predictions.items():
        for name, values in arrays.items():
            flat_predictions[f"{architecture}_{name}"] = np.asarray(values)
    core.atomic_npz(predictions_path, **flat_predictions)
    output_hashes = {
        "final_selections_sha256": core.sha256_file(final_path),
        "outer_selections_sha256": core.sha256_file(outer_path),
        "nested_per_seed_sha256": core.sha256_file(per_seed_path),
        "nested_summary_sha256": core.sha256_file(summary_path),
        "frozen_configs_sha256": core.sha256_file(config_path),
        "nested_predictions_sha256": core.sha256_file(predictions_path),
    }
    output_protocol = {
        **definition,
        "experiment_key": experiment_key,
        "runtime_seconds": float(time.perf_counter() - started),
        "outputs": {
            "final_selections": str(final_path),
            "outer_selections": str(outer_path),
            "nested_per_seed": str(per_seed_path),
            "nested_summary": str(summary_path),
            "frozen_configs": str(config_path),
            "nested_predictions": str(predictions_path),
            **output_hashes,
        },
    }
    core.atomic_json(protocol_output_path, output_protocol)
    latest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "architectures": architectures,
        "minimum_improvement": float(args.minimum_improvement),
        "protocol": str(protocol_output_path),
        "protocol_sha256": core.sha256_file(protocol_output_path),
        "final_selections": str(final_path),
        "outer_selections": str(outer_path),
        "nested_per_seed": str(per_seed_path),
        "nested_summary": str(summary_path),
        "frozen_configs": str(config_path),
        "nested_predictions": str(predictions_path),
        **output_hashes,
        "kddtest_accessed": False,
    }
    core.atomic_json(latest_path, latest)

    display_columns = [
        "model",
        "candidate_id",
        "best_standalone_label",
        "best_standalone_rare_f1",
        "required_rare_f1",
        "fusion_rare_f1",
        "improvement_over_best",
        "eligibility_pass",
        "nested_improvement_over_best",
        "nested_audit_pass",
        "outer_inner_eligibility_passes",
        "advance_to_test",
    ]
    display = pd.DataFrame(final_rows)[display_columns].copy()
    for column in (
        "best_standalone_rare_f1",
        "required_rare_f1",
        "fusion_rare_f1",
        "improvement_over_best",
        "nested_improvement_over_best",
    ):
        display[column] = display[column].map(lambda value: f"{100.0 * value:.2f}%")
    print("\n=== FINAL ALL-OOF VALIDATION SELECTIONS ===")
    print(display.to_string(index=False))
    print("\nSaved results:")
    print(f"  Final selections: {final_path}")
    print(f"  Outer selections: {outer_path}")
    print(f"  Nested summary: {summary_path}")
    print(f"  Frozen configs: {config_path}")
    print(f"  Protocol: {protocol_output_path}")
    print(f"  Latest pointer: {latest_path}")
    print("KDDTest+ accessed: NO")


if __name__ == "__main__":
    main()
