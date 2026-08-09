"""Evaluate frozen natural-Rare-F1 Super-Stack winners on KDDTest+.

This is a CPU-only final-test postprocessor.  It does not retrain any neural
backbone.  For each architecture and seed it:

1. validates the immutable all-OOF ranking produced by experiment
   ``6c7924f8d3a8``;
2. verifies that the predeclared candidate is the eligible winner when
   ordinary KDDTrain+ OOF Rare F1 is the primary selection metric;
3. refits only that candidate's temperature scalars, feature standardizer,
   and multinomial logistic stacker on all saved KDDTrain+ OOF predictions;
4. applies the frozen meta-model, blend, and rare-class offsets to the
   already-saved raw KDDTest+ probabilities for General, Focal, and Batching;
5. reports the frozen Super-Stack beside simple probability averaging.

KDDTest+ labels are used only after every configuration has been validated and
frozen from KDDTrain+.  Because earlier KDDTest+ results motivated this later
fusion study, the output is explicitly a post-hoc generalization analysis and
not a pristine untouched-test claim.

Examples:

    python -u src/evaluate_final_natural_rare_super_stack_kddtest.py --dry-run
    python -u src/evaluate_final_natural_rare_super_stack_kddtest.py
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import evaluate_final_simple_average_vs_safe_stack_kddtest as final_sources
import run_final_baseline_vs_full_kddtest_4gpu as final_core
import run_no_ctgan_model_ablation_4gpu as core
import tune_robust_calibrated_super_stack_all as stack


SCHEMA_VERSION = 1
SEARCH_EXPERIMENT_KEY = "6c7924f8d3a8"
SEARCH_PREFIX = "robust_calibrated_super_stack"
ARCHITECTURES = tuple(stack.ARCHITECTURES)
EXPERTS = tuple(stack.EXPERTS)
SEEDS = (0, 1, 2)
METRICS = tuple(core.METRICS)
METHODS = ("simple_average", "natural_rare_super_stack")
METHOD_LABELS = {
    "simple_average": "Simple probability average",
    "natural_rare_super_stack": "Frozen natural-Rare-F1 Super-Stack",
}

# These candidate identities were selected from the all-OOF final_cv rankings
# using ordinary Rare F1 first, followed by minimum rare-class F1, Macro-F1,
# MCC, and lower Rare-F1 seed SD.  They are intentionally embedded here so a
# mutable ranking or test result cannot silently alter the final policy.
FROZEN_NATURAL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "conv2d": {
        "candidate_id": "stack_c0_f0_q1_C5_r1_o30",
        "calibration": "raw",
        "feature_set": "F0",
        "q": 0.25,
        "C": 100.0,
        "rho": 0.25,
        "delta_r2l": -0.25,
        "delta_u2r": -0.25,
    },
    "conv1d": {
        "candidate_id": "stack_c0_f0_q0_C3_r2_o41",
        "calibration": "raw",
        "feature_set": "F0",
        "q": 0.0,
        "C": 1.0,
        "rho": 0.50,
        "delta_r2l": 0.0,
        "delta_u2r": 0.25,
    },
    "transformer": {
        "candidate_id": "stack_c0_f0_q4_C1_r1_o30",
        "calibration": "raw",
        "feature_set": "F0",
        "q": 1.0,
        "C": 0.01,
        "rho": 0.25,
        "delta_r2l": -0.25,
        "delta_u2r": -0.25,
    },
    "mlp": {
        "candidate_id": "stack_c1_f0_q1_C4_r4_o11",
        "calibration": "temperature",
        "feature_set": "F0",
        "q": 0.25,
        "C": 10.0,
        "rho": 1.0,
        "delta_r2l": -0.75,
        "delta_u2r": -0.50,
    },
}

NATURAL_SORT_COLUMNS = (
    "rare_f1_mean",
    "minimum_rare_f1_mean",
    "macro_f1_mean",
    "mcc_mean",
    "rare_f1_std",
    "candidate_id",
)
NATURAL_SORT_ASCENDING = (False, False, False, False, True, True)


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - {"true", "false"})
    if unknown:
        raise ValueError(f"Boolean ranking column contains unexpected values: {unknown}")
    return normalized.eq("true")


def natural_ranking(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_id",
        "eligible",
        "family",
        "calibration",
        "feature_set",
        "q",
        "C",
        "rho",
        "delta_r2l",
        "delta_u2r",
        *NATURAL_SORT_COLUMNS[:-1],
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Ranking is missing columns: {missing}")
    eligible = frame.loc[bool_series(frame["eligible"])].copy()
    if eligible.empty:
        raise RuntimeError("Natural-Rare-F1 ranking contains no eligible candidate.")
    return eligible.sort_values(
        list(NATURAL_SORT_COLUMNS),
        ascending=list(NATURAL_SORT_ASCENDING),
        kind="mergesort",
    ).reset_index(drop=True)


def validate_scalar(
    architecture: str,
    candidate_id: str,
    name: str,
    observed: Any,
    expected: Any,
) -> None:
    if isinstance(expected, float):
        try:
            matches = np.isclose(
                float(observed), expected, atol=1e-12, rtol=0.0
            )
        except (TypeError, ValueError):
            matches = False
    else:
        matches = observed == expected
    if not bool(matches):
        raise ValueError(
            f"Frozen {architecture} candidate {candidate_id} has {name}={observed!r}; "
            f"expected {expected!r}."
        )


def load_search_protocol(
    results_dir: Path,
) -> tuple[Path, Dict[str, Any], stack.SearchSettings]:
    path = results_dir / f"{SEARCH_PREFIX}_{SEARCH_EXPERIMENT_KEY}_protocol.json"
    protocol = final_sources.read_json(path)
    expected = {
        "schema_version": stack.SCHEMA_VERSION,
        "experiment_key": SEARCH_EXPERIMENT_KEY,
        "kddtest_accessed": False,
    }
    for name, value in expected.items():
        if protocol.get(name) != value:
            raise ValueError(
                f"Unexpected search protocol field {name}={protocol.get(name)!r}; "
                f"expected {value!r}."
            )
    definition = protocol.get("definition")
    if not isinstance(definition, dict):
        raise KeyError(f"Missing experiment definition in {path}.")
    if definition.get("architectures") != list(ARCHITECTURES):
        raise ValueError(f"{path} is not the frozen all-architecture experiment.")
    if definition.get("seeds") != list(SEEDS):
        raise ValueError(f"{path} does not use frozen seeds {list(SEEDS)}.")
    raw_settings = definition.get("settings")
    if not isinstance(raw_settings, dict):
        raise KeyError(f"Missing search settings in {path}.")
    try:
        settings = stack.SearchSettings(**raw_settings)
    except TypeError as error:
        raise ValueError(f"Invalid search settings in {path}: {error}") from error
    return path.resolve(), protocol, settings


def ranking_paths(results_dir: Path, architecture: str) -> tuple[Path, Path]:
    stem = f"{SEARCH_PREFIX}_{SEARCH_EXPERIMENT_KEY}_{architecture}"
    return (
        results_dir / f"{stem}_final_cv_ranking.csv.gz",
        results_dir / f"{stem}_best_config.json",
    )


def load_frozen_natural_config(
    results_dir: Path,
    protocol_path: Path,
    architecture: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    ranking_path, robust_record_path = ranking_paths(results_dir, architecture)
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Missing final-CV ranking: {ranking_path}")
    robust_record = final_sources.read_json(robust_record_path)
    if robust_record.get("experiment_key") != SEARCH_EXPERIMENT_KEY:
        raise ValueError(f"Unexpected experiment key in {robust_record_path}.")
    if robust_record.get("architecture") != architecture:
        raise ValueError(f"Unexpected architecture in {robust_record_path}.")
    if robust_record.get("protocol_sha256") != core.sha256_file(protocol_path):
        raise ValueError(f"Search protocol hash mismatch in {robust_record_path}.")
    ranking_hash = core.sha256_file(ranking_path)
    if robust_record.get("final_cv_ranking_sha256") != ranking_hash:
        raise ValueError(f"Final-CV ranking hash mismatch for {ranking_path}.")

    frame = pd.read_csv(ranking_path)
    if "stage" in frame and set(frame["stage"].astype(str)) != {"final_cv"}:
        raise ValueError(f"{ranking_path} contains a non-final_cv stage.")
    if "architecture" in frame and set(frame["architecture"].astype(str)) != {
        architecture
    }:
        raise ValueError(f"{ranking_path} contains another architecture.")
    ordered = natural_ranking(frame)
    winner = ordered.iloc[0]
    expected = FROZEN_NATURAL_CONFIGS[architecture]
    candidate_id = str(winner["candidate_id"])
    if candidate_id != expected["candidate_id"]:
        raise ValueError(
            f"Natural-Rare-F1 winner changed for {architecture}: ranking selects "
            f"{candidate_id}, frozen policy requires {expected['candidate_id']}."
        )
    validate_scalar(architecture, candidate_id, "family", winner["family"], "stack")
    for name in (
        "calibration",
        "feature_set",
        "q",
        "C",
        "rho",
        "delta_r2l",
        "delta_u2r",
    ):
        validate_scalar(architecture, candidate_id, name, winner[name], expected[name])

    config = stack.candidate_config(winner)
    selection = {
        "architecture": architecture,
        "search_experiment_key": SEARCH_EXPERIMENT_KEY,
        "candidate_id": candidate_id,
        "ranking_path": str(ranking_path.resolve()),
        "ranking_sha256": ranking_hash,
        "robust_record_path": str(robust_record_path.resolve()),
        "robust_record_sha256": core.sha256_file(robust_record_path),
        "natural_rank": 1,
        "ordinary_rare_f1_mean": float(winner["rare_f1_mean"]),
        "ordinary_rare_f1_std": float(winner["rare_f1_std"]),
        "r2l_f1_mean": float(winner["r2l_f1_mean"]),
        "u2r_f1_mean": float(winner["u2r_f1_mean"]),
        "macro_f1_mean": float(winner["macro_f1_mean"]),
        "mcc_mean": float(winner["mcc_mean"]),
        "robust_rare_f1_mean": float(winner["robust_rare_f1_mean"]),
    }
    return config, selection


def normalized_seed_map(value: Mapping[str | int, Any]) -> Dict[str, str]:
    return {str(int(seed)): str(item) for seed, item in value.items()}


def validate_oof_lineage(
    protocol: Mapping[str, Any], architecture_input: stack.ArchitectureInput
) -> None:
    architecture = architecture_input.architecture
    sources = protocol.get("sources", {}).get(architecture)
    if not isinstance(sources, dict):
        raise KeyError(f"Search protocol is missing {architecture} sources.")
    recorded_hashes = sources.get("oof_hashes")
    if not isinstance(recorded_hashes, dict):
        raise KeyError(f"Search protocol is missing {architecture} OOF hashes.")
    for expert in EXPERTS:
        observed = normalized_seed_map(architecture_input.source_hashes[expert])
        recorded = normalized_seed_map(recorded_hashes[expert])
        if observed != recorded:
            raise ValueError(
                f"Current {architecture}:{expert} OOF artifacts differ from the "
                "frozen search inputs."
            )
    recorded_pointer_hashes = sources.get("pointer_hashes")
    if recorded_pointer_hashes != architecture_input.pointer_hashes:
        raise ValueError(f"{architecture} OOF pointer hashes changed after selection.")
    recorded_protocol_hashes = sources.get("protocol_hashes")
    if recorded_protocol_hashes != architecture_input.protocol_hashes:
        raise ValueError(f"{architecture} OOF source-protocol hashes changed.")


def predict_with_fitted_state(
    expert_probabilities: np.ndarray,
    config: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
    settings: stack.SearchSettings,
) -> Dict[str, np.ndarray]:
    raw = np.asarray(expert_probabilities, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[1:] != (len(EXPERTS), stack.CLASS_COUNT):
        raise ValueError(f"Expected expert probabilities (N,3,5), got {raw.shape}.")
    if not np.isfinite(raw).all():
        raise ValueError("Expert probabilities contain non-finite values.")
    if np.any(raw < -1e-7) or np.any(raw > 1.0 + 1e-7):
        raise ValueError("Expert probabilities contain a value outside [0,1].")
    if not np.allclose(raw.sum(axis=2), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError("Expert probability vectors do not sum to one.")
    if config.get("family") != "stack":
        raise ValueError("Frozen natural-Rare-F1 candidate must be a stacker.")

    active = raw.copy()
    temperatures = np.asarray(state["temperatures"], dtype=np.float64)
    if temperatures.shape != (len(EXPERTS),):
        raise ValueError(f"Unexpected temperature shape: {temperatures.shape}")
    calibration = str(config["calibration"])
    if calibration == "temperature":
        for expert_index, temperature in enumerate(temperatures):
            active[:, expert_index, :] = stack.temperature_scale(
                active[:, expert_index, :], float(temperature), settings.epsilon
            )
    elif calibration == "raw":
        if not np.allclose(temperatures, 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("Raw-calibration model state contains a non-unit temperature.")
    else:
        raise ValueError(f"Unknown calibration: {calibration}")

    features = stack.build_features(
        active, str(config["feature_set"]), settings.epsilon
    )
    features = stack.apply_standardizer(
        features,
        np.asarray(state["feature_mean"], dtype=np.float64),
        np.asarray(state["feature_scale"], dtype=np.float64),
    )
    classes = np.asarray(state["classes"], dtype=np.int64)
    if not np.array_equal(classes, np.arange(stack.CLASS_COUNT)):
        raise ValueError(f"Unexpected fitted classes: {classes.tolist()}")
    coef = np.asarray(state["coef"], dtype=np.float64)
    intercept = np.asarray(state["intercept"], dtype=np.float64)
    if coef.shape != (stack.CLASS_COUNT, features.shape[1]):
        raise ValueError(f"Unexpected coefficient shape: {coef.shape}")
    if intercept.shape != (stack.CLASS_COUNT,):
        raise ValueError(f"Unexpected intercept shape: {intercept.shape}")
    logits = features @ coef.T + intercept[None, :]
    logits -= stack.logsumexp(logits, axis=1, keepdims=True)
    stack_probabilities = np.exp(logits)
    average_probabilities = raw.mean(axis=1, dtype=np.float64)
    rho = float(config["rho"])
    blended_probabilities = (
        (1.0 - rho) * average_probabilities + rho * stack_probabilities
    )
    decision_probabilities = stack.offset_decision_probabilities(
        blended_probabilities,
        float(config["delta_r2l"]),
        float(config["delta_u2r"]),
        settings.epsilon,
    )
    predictions = np.argmax(decision_probabilities, axis=1).astype(np.int64)
    for name, values in (
        ("stack", stack_probabilities),
        ("average", average_probabilities),
        ("blended", blended_probabilities),
        ("decision", decision_probabilities),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} probabilities contain non-finite values.")
        tolerance = 2e-4 if name in {"average", "blended"} else 1e-10
        if not np.allclose(values.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
            raise RuntimeError(f"{name} probabilities do not sum to one.")
    return {
        "average_probabilities": average_probabilities,
        "stack_probabilities": stack_probabilities,
        "blended_probabilities": blended_probabilities,
        "decision_probabilities": decision_probabilities,
        "predictions": predictions,
    }


def aggregate_metrics(per_seed: pd.DataFrame, seeds: Sequence[int]) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    expected_seeds = sorted(int(seed) for seed in seeds)
    for (architecture, method), group in per_seed.groupby(
        ["architecture", "method"], sort=False
    ):
        observed = sorted(int(seed) for seed in group["seed"])
        if observed != expected_seeds:
            raise RuntimeError(f"{architecture}:{method} is missing a seed.")
        row: Dict[str, Any] = {
            "architecture": architecture,
            "model": str(group.iloc[0]["model"]),
            "method": method,
            "method_label": str(group.iloc[0]["method_label"]),
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
            "changed_vs_average",
            "r2l_predictions",
            "u2r_predictions",
        ):
            values = pd.to_numeric(group[diagnostic], errors="raise")
            row[f"{diagnostic}_mean"] = float(values.mean())
            row[f"{diagnostic}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    architecture_order = {name: index for index, name in enumerate(ARCHITECTURES)}
    method_order = {name: index for index, name in enumerate(METHODS)}
    summary["_architecture_order"] = summary["architecture"].map(architecture_order)
    summary["_method_order"] = summary["method"].map(method_order)
    return (
        summary.sort_values(["_architecture_order", "_method_order"])
        .drop(columns=["_architecture_order", "_method_order"])
        .reset_index(drop=True)
    )


def formatted_summary(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary[["model", "method_label", "runs", "seeds"]].copy()
    for metric in METRICS:
        output[metric] = summary.apply(
            lambda row, name=metric: (
                f"{100.0 * row[f'{name}_mean']:.2f}% +/- "
                f"{100.0 * row[f'{name}_std']:.2f}%"
            ),
            axis=1,
        )
    return output


def paired_deltas(
    per_seed: pd.DataFrame, architectures: Sequence[str], seeds: Sequence[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[Dict[str, Any]] = []
    for architecture in architectures:
        group = per_seed[per_seed["architecture"] == architecture]
        average = group[group["method"] == "simple_average"].set_index("seed")
        frozen = group[group["method"] == "natural_rare_super_stack"].set_index(
            "seed"
        )
        for seed in seeds:
            row: Dict[str, Any] = {
                "architecture": architecture,
                "model": final_core.FROZEN_CONFIG[architecture]["label"],
                "seed": int(seed),
            }
            for metric in METRICS:
                row[f"{metric}_delta_stack_minus_average"] = float(
                    frozen.loc[int(seed), metric] - average.loc[int(seed), metric]
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary_rows: list[Dict[str, Any]] = []
    for architecture, group in frame.groupby("architecture", sort=False):
        row: Dict[str, Any] = {
            "architecture": architecture,
            "model": str(group.iloc[0]["model"]),
            "runs": int(len(group)),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"])),
        }
        for metric in METRICS:
            column = f"{metric}_delta_stack_minus_average"
            values = pd.to_numeric(group[column], errors="raise")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        summary_rows.append(row)
    return frame, pd.DataFrame(summary_rows)


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
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument(
        "--source-results-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory containing hashed final KDDTest run artifacts; repeatable.",
    )
    parser.add_argument(
        "--source-experiment",
        action="append",
        default=[],
        metavar="ARCH:EXPERT=KEY",
        help="Resolve an ambiguous final source without consulting test metrics.",
    )
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
        "--output-prefix",
        default="final_natural_rare_super_stack_kddtest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate frozen rankings, OOF lineage, and final source metadata/hashes "
            "without fitting meta-models, loading KDDTest arrays, or writing outputs."
        ),
    )
    args = parser.parse_args(argv)
    architectures = list(dict.fromkeys(str(value) for value in args.architectures))
    if len(architectures) != len(args.architectures):
        parser.error("--architectures must be unique.")
    args.architectures = architectures
    if sorted(args.seeds) != list(SEEDS) or len(args.seeds) != len(set(args.seeds)):
        parser.error("This frozen evaluation requires exactly seeds 0 1 2.")
    if (
        not args.output_prefix.strip()
        or Path(args.output_prefix).name != args.output_prefix
    ):
        parser.error("--output-prefix must be a nonempty filename-safe value.")
    try:
        args.source_overrides = final_sources.parse_source_overrides(
            args.source_experiment
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    started = time.perf_counter()
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    results_dir = (args.results_dir or (repo_root / "results")).expanduser().resolve()
    train_path = stack.safe.resolve_cli_path(repo_root, args.train_data)
    if not train_path.is_file():
        raise SystemExit(f"KDDTrain+ not found: {train_path}")
    architectures = list(args.architectures)
    seeds = sorted(int(seed) for seed in args.seeds)

    protocol_path, search_protocol, search_settings = load_search_protocol(results_dir)
    configs: Dict[str, Dict[str, Any]] = {}
    selections: Dict[str, Dict[str, Any]] = {}
    for architecture in architectures:
        config, selection = load_frozen_natural_config(
            results_dir, protocol_path, architecture
        )
        configs[architecture] = config
        selections[architecture] = selection

    print("Loading and validating frozen KDDTrain+ OOF inputs...", flush=True)
    architecture_inputs = [
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
    stack.validate_cross_architecture_alignment(architecture_inputs)
    input_by_architecture = {item.architecture: item for item in architecture_inputs}
    for item in architecture_inputs:
        validate_oof_lineage(search_protocol, item)

    source_roots = final_sources.unique_paths(
        [
            *(path.expanduser() for path in args.source_results_dir),
            results_dir,
        ]
    )
    source_validation = {
        architecture: {"focal_best": input_by_architecture[architecture].focal_best}
        for architecture in architectures
    }
    sources = final_sources.select_sources(
        source_roots,
        architectures,
        seeds,
        source_validation,
        args.source_overrides,
    )

    print("Frozen natural-Rare-F1 Super-Stack KDDTest+ evaluation")
    print(f"Architectures: {architectures}")
    print(f"Seeds: {seeds}")
    print("Experts: General=baseline, Focal=focal_only, Batching=batch_only")
    print("Backbone retraining: NO")
    print("Meta-model refit: KDDTrain+ OOF only")
    print("Configuration selection on KDDTest+: NO")
    print("Evaluation interpretation: post-hoc generalization analysis")
    for architecture in architectures:
        config = configs[architecture]
        groups = {
            expert: sources[architecture][expert][seeds[0]].experiment_key
            for expert in EXPERTS
        }
        print(
            f"  {architecture}: candidate={config['candidate_id']}; "
            f"calibration={config['calibration']}; features={config['feature_set']}; "
            f"q={config['q']}; C={config['C']}; rho={config['rho']}; "
            f"offsets=({config['delta_r2l']},{config['delta_u2r']}); "
            f"sources={groups}"
        )
    if args.dry_run:
        print(
            "Dry run complete; frozen rankings, OOF lineage, and KDDTest source "
            "metadata/hashes are valid. No meta-model was fitted, no KDDTest "
            "prediction array was loaded, and no output was written."
        )
        return

    loaded: Dict[str, Dict[str, Dict[int, final_sources.LoadedSource]]] = {
        architecture: {expert: {} for expert in EXPERTS}
        for architecture in architectures
    }
    reference_labels: np.ndarray | None = None
    for architecture in architectures:
        for expert in EXPERTS:
            for seed in seeds:
                artifact = final_sources.load_prediction(
                    sources[architecture][expert][seed]
                )
                if reference_labels is None:
                    reference_labels = artifact.labels
                elif not np.array_equal(artifact.labels, reference_labels):
                    raise ValueError(
                        f"KDDTest+ label/order mismatch for "
                        f"{architecture}:{expert}:s{seed}."
                    )
                loaded[architecture][expert][seed] = artifact
    if reference_labels is None:
        raise RuntimeError("No KDDTest+ prediction artifact was loaded.")

    source_identity = {
        architecture: {
            expert: {
                str(seed): {
                    "experiment_key": sources[architecture][expert][
                        seed
                    ].experiment_key,
                    "variant": sources[architecture][expert][seed].variant,
                    "result_sha256": sources[architecture][expert][seed].result_sha256,
                    "prediction_sha256": sources[architecture][expert][
                        seed
                    ].prediction_sha256,
                    "cache_metadata_sha256": sources[architecture][expert][
                        seed
                    ].cache_metadata_sha256,
                }
                for seed in seeds
            }
            for expert in EXPERTS
        }
        for architecture in architectures
    }
    experiment_definition = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "frozen natural-Rare-F1 Super-Stack KDDTest+ evaluation",
        "architectures": architectures,
        "seeds": seeds,
        "expert_order": list(EXPERTS),
        "expert_variants": final_sources.EXPERT_VARIANTS,
        "methods": list(METHODS),
        "search_experiment_key": SEARCH_EXPERIMENT_KEY,
        "search_protocol_sha256": core.sha256_file(protocol_path),
        "selection_rule": {
            "eligible_guard_required": True,
            "columns": list(NATURAL_SORT_COLUMNS),
            "ascending": list(NATURAL_SORT_ASCENDING),
        },
        "frozen_selections": selections,
        "frozen_configs": configs,
        "search_settings": asdict(search_settings),
        "source_identity": source_identity,
        "simple_average_rule": "argmax(mean raw G/F/B probabilities)",
        "stack_average_anchor": "raw arithmetic mean of G/F/B probabilities",
        "selection_data": "KDDTrain+ cross-fitted OOF predictions only",
        "evaluation_data": "saved KDDTest+ raw expert probabilities",
        "kddtest_used_for_selection": False,
        "backbone_models_retrained": False,
        "meta_models_refit": True,
        "meta_model_refit_data": "all KDDTrain+ OOF rows, separately per seed",
        "evaluation_interpretation": (
            "post-hoc generalization analysis because earlier KDDTest+ results "
            "motivated the later fusion design"
        ),
        "script_sha256": core.sha256_file(script_path),
        "stack_implementation_sha256": core.sha256_file(Path(stack.__file__)),
        "source_loader_sha256": core.sha256_file(Path(final_sources.__file__)),
    }
    experiment_key = stable_hash(experiment_definition)
    stem = f"{args.output_prefix}_{experiment_key}"
    prediction_dir = results_dir / f"{stem}_predictions"
    model_dir = results_dir / f"{stem}_models"
    per_seed_path = results_dir / f"{stem}_per_seed.csv"
    summary_path = results_dir / f"{stem}_summary.csv"
    formatted_path = results_dir / f"{stem}_summary_formatted.csv"
    delta_path = results_dir / f"{stem}_paired_deltas.csv"
    delta_summary_path = results_dir / f"{stem}_paired_delta_summary.csv"
    source_path = results_dir / f"{stem}_source_artifacts.csv"
    refit_path = results_dir / f"{stem}_meta_refits.csv"
    protocol_output_path = results_dir / f"{stem}_protocol.json"
    latest_path = results_dir / f"{args.output_prefix}_latest.json"

    run_rows: list[Dict[str, Any]] = []
    source_rows: list[Dict[str, Any]] = []
    refit_rows: list[Dict[str, Any]] = []
    pending_predictions: list[tuple[Path, Dict[str, np.ndarray]]] = []
    pending_models: list[tuple[Path, Dict[str, np.ndarray]]] = []
    with threadpool_limits(limits=1):
        for architecture in architectures:
            architecture_input = input_by_architecture[architecture]
            config = configs[architecture]
            all_train_indices = np.arange(
                len(architecture_input.labels), dtype=np.int64
            )
            model_payload: Dict[str, np.ndarray] = {
                "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
                "architecture": np.asarray(architecture),
                "seeds": np.asarray(seeds, dtype=np.int64),
                "candidate_id": np.asarray(config["candidate_id"]),
                "calibration": np.asarray(config["calibration"]),
                "feature_set": np.asarray(config["feature_set"]),
                "q": np.asarray(float(config["q"]), dtype=np.float64),
                "C": np.asarray(float(config["C"]), dtype=np.float64),
                "rho": np.asarray(float(config["rho"]), dtype=np.float64),
                "delta_r2l": np.asarray(
                    float(config["delta_r2l"]), dtype=np.float64
                ),
                "delta_u2r": np.asarray(
                    float(config["delta_u2r"]), dtype=np.float64
                ),
                "feature_names": np.asarray(
                    stack.feature_names(str(config["feature_set"])), dtype="U96"
                ),
                "search_experiment_key": np.asarray(SEARCH_EXPERIMENT_KEY),
                "evaluation_experiment_key": np.asarray(experiment_key),
            }
            for seed in seeds:
                print(
                    f"Refitting {architecture} seed {seed} frozen meta-model...",
                    flush=True,
                )
                audit_predictions, audit_probabilities, state = (
                    stack.fit_selected_candidate(
                        architecture_input,
                        seed,
                        config,
                        all_train_indices,
                        all_train_indices,
                        search_settings,
                    )
                )
                train_raw = architecture_input.probabilities[
                    architecture_input.seeds.index(seed)
                ].transpose(1, 0, 2)
                audit = predict_with_fitted_state(
                    train_raw, config, state, search_settings
                )
                if not np.array_equal(audit_predictions, audit["predictions"]):
                    raise RuntimeError(
                        f"Inference audit prediction mismatch for {architecture}:s{seed}."
                    )
                if not np.allclose(
                    audit_probabilities,
                    audit["decision_probabilities"],
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise RuntimeError(
                        f"Inference audit probability mismatch for {architecture}:s{seed}."
                    )

                test_probabilities = np.stack(
                    [
                        loaded[architecture][expert][seed].probabilities
                        for expert in EXPERTS
                    ],
                    axis=1,
                )
                labels = loaded[architecture][EXPERTS[0]][seed].labels
                result = predict_with_fitted_state(
                    test_probabilities, config, state, search_settings
                )
                average_predictions = np.argmax(
                    result["average_probabilities"], axis=1
                ).astype(np.int64)
                frozen_predictions = result["predictions"]
                prediction_path = prediction_dir / f"{architecture}_s{seed}.npz"
                pending_predictions.append(
                    (
                        prediction_path,
                        {
                            "row_indices": np.arange(len(labels), dtype=np.int64),
                            "labels": labels,
                            "expert_probabilities": test_probabilities.astype(
                                np.float32
                            ),
                            "simple_average_probabilities": result[
                                "average_probabilities"
                            ].astype(np.float32),
                            "simple_average_predictions": average_predictions,
                            "stack_probabilities": result["stack_probabilities"].astype(
                                np.float32
                            ),
                            "blended_probabilities": result[
                                "blended_probabilities"
                            ].astype(np.float32),
                            "decision_probabilities": result[
                                "decision_probabilities"
                            ].astype(np.float32),
                            "natural_rare_super_stack_predictions": frozen_predictions,
                            "candidate_id": np.asarray(config["candidate_id"]),
                        },
                    )
                )

                common = {
                    "architecture": architecture,
                    "model": final_core.FROZEN_CONFIG[architecture]["label"],
                    "seed": int(seed),
                    "candidate_id": config["candidate_id"],
                    "calibration": config["calibration"],
                    "feature_set": config["feature_set"],
                    "q": float(config["q"]),
                    "C": float(config["C"]),
                    "rho": float(config["rho"]),
                    "delta_r2l": float(config["delta_r2l"]),
                    "delta_u2r": float(config["delta_u2r"]),
                    "prediction_path": str(prediction_path),
                }
                run_rows.append(
                    {
                        **common,
                        "method": "simple_average",
                        "method_label": METHOD_LABELS["simple_average"],
                        "changed_vs_average": 0,
                        "r2l_predictions": int(
                            np.sum(average_predictions == stack.R2L_CLASS)
                        ),
                        "u2r_predictions": int(
                            np.sum(average_predictions == stack.U2R_CLASS)
                        ),
                        **core.calculate_metrics(labels, average_predictions),
                    }
                )
                run_rows.append(
                    {
                        **common,
                        "method": "natural_rare_super_stack",
                        "method_label": METHOD_LABELS[
                            "natural_rare_super_stack"
                        ],
                        "changed_vs_average": int(
                            np.sum(frozen_predictions != average_predictions)
                        ),
                        "r2l_predictions": int(
                            np.sum(frozen_predictions == stack.R2L_CLASS)
                        ),
                        "u2r_predictions": int(
                            np.sum(frozen_predictions == stack.U2R_CLASS)
                        ),
                        **core.calculate_metrics(labels, frozen_predictions),
                    }
                )

                prefix = f"seed_{seed}"
                for name, values in state.items():
                    model_payload[f"{prefix}_{name}"] = np.asarray(values)
                refit_rows.append(
                    {
                        "architecture": architecture,
                        "seed": int(seed),
                        "candidate_id": config["candidate_id"],
                        "temperature_general": float(state["temperatures"][0]),
                        "temperature_focal": float(state["temperatures"][1]),
                        "temperature_batching": float(state["temperatures"][2]),
                        "temperature_boundary_general": bool(
                            state["temperature_boundary"][0]
                        ),
                        "temperature_boundary_focal": bool(
                            state["temperature_boundary"][1]
                        ),
                        "temperature_boundary_batching": bool(
                            state["temperature_boundary"][2]
                        ),
                        "feature_count": int(len(state["feature_mean"])),
                        "max_iterations": int(np.max(state["n_iter"])),
                    }
                )
                for expert in EXPERTS:
                    source = sources[architecture][expert][seed]
                    source_rows.append(
                        {
                            "architecture": architecture,
                            "expert": expert,
                            "variant": source.variant,
                            "seed": int(seed),
                            "experiment_key": source.experiment_key,
                            "result_path": str(source.result_path),
                            "result_sha256": source.result_sha256,
                            "prediction_path": str(source.prediction_path),
                            "prediction_sha256": source.prediction_sha256,
                            "cache_metadata_path": str(source.cache_metadata_path),
                            "cache_metadata_sha256": source.cache_metadata_sha256,
                        }
                    )
            pending_models.append(
                (model_dir / f"{architecture}_final_seed_models.npz", model_payload)
            )

    per_seed = pd.DataFrame(run_rows)
    expected_rows = len(architectures) * len(seeds) * len(METHODS)
    if len(per_seed) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} metric rows, got {len(per_seed)}.")
    summary = aggregate_metrics(per_seed, seeds)
    formatted = formatted_summary(summary)
    deltas, delta_summary = paired_deltas(per_seed, architectures, seeds)

    prediction_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    prediction_hashes: Dict[str, str] = {}
    model_hashes: Dict[str, str] = {}
    for path, arrays in pending_predictions:
        core.atomic_npz(path, **arrays)
        prediction_hashes[str(path)] = core.sha256_file(path)
    for path, arrays in pending_models:
        core.atomic_npz(path, **arrays)
        model_hashes[str(path)] = core.sha256_file(path)
    core.atomic_csv(source_path, pd.DataFrame(source_rows))
    core.atomic_csv(refit_path, pd.DataFrame(refit_rows))
    core.atomic_csv(per_seed_path, per_seed)
    core.atomic_csv(summary_path, summary)
    core.atomic_csv(formatted_path, formatted)
    core.atomic_csv(delta_path, deltas)
    core.atomic_csv(delta_summary_path, delta_summary)

    output_protocol = {
        **experiment_definition,
        "experiment_key": experiment_key,
        "runtime_seconds": float(time.perf_counter() - started),
        "kddtest_accessed": True,
        "kddtest_access_mode": "saved final prediction artifacts only",
        "outputs": {
            "source_artifacts": str(source_path),
            "meta_refits": str(refit_path),
            "per_seed": str(per_seed_path),
            "summary": str(summary_path),
            "formatted_summary": str(formatted_path),
            "paired_deltas": str(delta_path),
            "paired_delta_summary": str(delta_summary_path),
            "prediction_directory": str(prediction_dir),
            "prediction_sha256": prediction_hashes,
            "model_directory": str(model_dir),
            "model_sha256": model_hashes,
        },
    }
    core.atomic_json(protocol_output_path, output_protocol)
    latest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "architectures": architectures,
        "protocol": str(protocol_output_path),
        "protocol_sha256": core.sha256_file(protocol_output_path),
        "source_artifacts": str(source_path),
        "meta_refits": str(refit_path),
        "per_seed": str(per_seed_path),
        "summary": str(summary_path),
        "formatted_summary": str(formatted_path),
        "paired_deltas": str(delta_path),
        "paired_delta_summary": str(delta_summary_path),
        "prediction_directory": str(prediction_dir),
        "model_directory": str(model_dir),
        "kddtest_accessed": True,
        "evaluation_interpretation": "post-hoc generalization analysis",
    }
    core.atomic_json(latest_path, latest)

    print("\n=== Final KDDTest+ natural-Rare-F1 Super-Stack evaluation ===")
    print(formatted.to_string(index=False))
    print("\nSaved results:")
    print(f"  Summary: {summary_path}")
    print(f"  Per-seed metrics: {per_seed_path}")
    print(f"  Paired deltas: {delta_summary_path}")
    print(f"  Meta-model refits: {refit_path}")
    print(f"  Protocol: {protocol_output_path}")
    print(f"  Latest pointer: {latest_path}")
    print("Interpretation: post-hoc generalization analysis, not a pristine test.")


if __name__ == "__main__":
    main()
