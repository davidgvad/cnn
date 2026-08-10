"""Post-hoc validation-filtered Super-Stack tolerance audit on KDDTest+.

This script answers an explicitly exploratory question:

* among the 68,124 canonical fusion rules already evaluated on KDDTrain+ OOF
  predictions, which rules are no more than a fixed Rare-F1 tolerance below
  the strongest standalone validation configuration; and
* after applying every such survivor unchanged to saved KDDTest+
  probabilities, which are also no more than that tolerance below the
  strongest standalone KDDTest+ configuration?

The four standalone comparators are General (baseline), Focal-only,
Batching-only, and Baseline + the architecture-specific frozen score scaling
used by the original three-stage ablation.  Neural backbones are never
retrained.  Stack-family survivors require a full-OOF refit of their small
multinomial logistic meta-model before test inference.  Refits are shared by
all rho/offset rules with the same calibration, feature set, q, and C.

KDDTest+ participates in the final pass/fail audit.  Consequently the output
is a post-hoc validation-test tolerance/Pareto analysis, not an untouched-test
model-selection result.

Examples:

    python -u src/audit_validation_filtered_fusion_kddtest.py --dry-run
    python -u src/audit_validation_filtered_fusion_kddtest.py \
        --workers 18 --threads-per-worker 1
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import evaluate_final_natural_rare_super_stack_kddtest as natural_eval
import evaluate_final_simple_average_vs_safe_stack_kddtest as final_sources
import run_final_baseline_vs_full_kddtest_4gpu as final_core
import run_no_ctgan_model_ablation_4gpu as core
import select_fusion_against_all_validation_baselines as selector
import tune_robust_calibrated_super_stack_all as stack


SCHEMA_VERSION = 1
ARCHITECTURES = tuple(stack.ARCHITECTURES)
EXPERTS = tuple(stack.EXPERTS)
SEEDS = (0, 1, 2)
METRICS = tuple(core.METRICS)
STANDALONES = ("general", "focal", "batching", "scaling")
STANDALONE_LABELS = {
    "general": "Baseline",
    "focal": "Baseline + focal loss",
    "batching": "Baseline + minority batching",
    "scaling": "Baseline + frozen scaling",
}
DEFAULT_TOLERANCE = 0.005
NUMERIC_TOLERANCE = 1e-12


_AUDIT_CONTEXT: Dict[str, Any] | None = None
_WORKER_LIMITER: Any = None


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def fusion_mask(frame: pd.DataFrame) -> pd.Series:
    family = frame["family"].astype(str)
    candidate = frame["candidate_id"].astype(str)
    return family.isin({"average_offset", "stack"}) | candidate.eq(
        "fixed_average"
    )


def aggregate_prediction_metrics(
    labels: np.ndarray,
    predictions: Mapping[int, np.ndarray],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, Dict[str, float]]:
    per_seed_rows: list[Dict[str, float]] = []
    labels = np.asarray(labels, dtype=np.int64)
    for seed in seeds:
        values = np.asarray(predictions[int(seed)], dtype=np.int64)
        if values.shape != labels.shape:
            raise ValueError(
                f"Seed {seed} predictions {values.shape} do not match {labels.shape}."
            )
        per_seed_rows.append(
            {"seed": int(seed), **core.calculate_metrics(labels, values)}
        )
    per_seed = pd.DataFrame(per_seed_rows)
    summary: Dict[str, float] = {}
    for metric in METRICS:
        values = pd.to_numeric(per_seed[metric], errors="raise")
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )
    return per_seed, summary


def standalone_metrics_from_probabilities(
    labels: np.ndarray,
    probabilities_by_seed: Mapping[int, np.ndarray],
    seeds: Sequence[int],
    r2l_coefficient: float,
    u2r_coefficient: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: Dict[str, Dict[int, np.ndarray]] = {
        method: {} for method in STANDALONES
    }
    for seed in seeds:
        raw = np.asarray(probabilities_by_seed[int(seed)], dtype=np.float64)
        if raw.ndim != 3 or raw.shape[1:] != (
            len(EXPERTS),
            stack.CLASS_COUNT,
        ):
            raise ValueError(f"Seed {seed} has unexpected probability shape {raw.shape}.")
        for expert_index, expert in enumerate(EXPERTS):
            predictions[expert][int(seed)] = np.argmax(
                raw[:, expert_index, :], axis=1
            ).astype(np.int64)
        predictions["scaling"][int(seed)] = core.apply_class_score_scaling(
            raw[:, EXPERTS.index("general"), :],
            {
                stack.R2L_CLASS: float(r2l_coefficient),
                stack.U2R_CLASS: float(u2r_coefficient),
            },
        )

    per_seed_frames: list[pd.DataFrame] = []
    summary_rows: list[Dict[str, Any]] = []
    for method in STANDALONES:
        per_seed, summary = aggregate_prediction_metrics(
            labels, predictions[method], seeds
        )
        per_seed.insert(0, "method_label", STANDALONE_LABELS[method])
        per_seed.insert(0, "method", method)
        per_seed_frames.append(per_seed)
        summary_rows.append(
            {
                "method": method,
                "method_label": STANDALONE_LABELS[method],
                "runs": len(seeds),
                "seeds": ",".join(str(int(seed)) for seed in seeds),
                **summary,
            }
        )
    return (
        pd.concat(per_seed_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def validation_survivors(
    ranking: pd.DataFrame,
    standalone_summary: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("Tolerance must be finite and nonnegative.")
    required = {
        "candidate_id",
        "family",
        "valid_all_seeds",
        "rare_f1_mean",
        "rare_f1_std",
    }
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise KeyError(f"Ranking is missing columns: {missing}")
    if set(standalone_summary["method"].astype(str)) != set(STANDALONES):
        raise ValueError("Standalone summary does not contain the four methods.")
    standalone_values = {
        str(row["method"]): float(row["rare_f1_mean"])
        for _, row in standalone_summary.iterrows()
    }
    best_method = max(
        STANDALONES,
        key=lambda name: (
            round(standalone_values[name], 12),
            -STANDALONES.index(name),
        ),
    )
    best_value = standalone_values[best_method]
    threshold = best_value - float(tolerance)
    valid = selector.bool_series(ranking["valid_all_seeds"], "valid_all_seeds")
    candidates = ranking.loc[fusion_mask(ranking) & valid].copy()
    candidates["validation_best_standalone_method"] = best_method
    candidates["validation_best_standalone_label"] = STANDALONE_LABELS[best_method]
    candidates["validation_best_standalone_rare_f1"] = best_value
    candidates["validation_tolerance"] = float(tolerance)
    candidates["validation_threshold"] = threshold
    candidates["validation_gap"] = (
        pd.to_numeric(candidates["rare_f1_mean"], errors="raise") - best_value
    )
    candidates["validation_pass"] = (
        pd.to_numeric(candidates["rare_f1_mean"], errors="raise")
        >= threshold - NUMERIC_TOLERANCE
    )
    survivors = candidates.loc[candidates["validation_pass"]].copy()
    survivors = survivors.sort_values(
        ["rare_f1_mean", "rare_f1_std", "candidate_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    details = {
        "best_method": best_method,
        "best_label": STANDALONE_LABELS[best_method],
        "best_rare_f1": best_value,
        "threshold": threshold,
        "fusion_candidate_count": int(len(candidates)),
        "survivor_count": int(len(survivors)),
    }
    return survivors, details


def score_probability_candidates(
    labels: np.ndarray,
    average_probabilities: np.ndarray,
    candidates: pd.DataFrame,
    settings: stack.SearchSettings,
    stack_probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Score a same-family candidate subset for one seed."""
    labels = np.asarray(labels, dtype=np.int64)
    average = np.asarray(average_probabilities, dtype=np.float64)
    frame = candidates.reset_index(drop=True).copy()
    if len(frame) == 0:
        return pd.DataFrame(columns=["candidate_id", *METRICS])
    if average.shape != (len(labels), stack.CLASS_COUNT):
        raise ValueError("Average probabilities do not align with labels.")
    output = np.empty((len(frame), len(METRICS)), dtype=np.float64)
    families = set(frame["family"].astype(str))

    if families.issubset({"fixed", "average_offset"}):
        for start in range(0, len(frame), settings.candidate_chunk_size):
            stop = min(start + settings.candidate_chunk_size, len(frame))
            chunk = frame.iloc[start:stop]
            scores = np.broadcast_to(
                np.log(np.clip(average, settings.epsilon, 1.0))[:, None, :],
                (len(labels), len(chunk), stack.CLASS_COUNT),
            ).copy()
            scores[:, :, stack.R2L_CLASS] += pd.to_numeric(
                chunk["delta_r2l"], errors="raise"
            ).to_numpy(dtype=np.float64)[None, :]
            scores[:, :, stack.U2R_CLASS] += pd.to_numeric(
                chunk["delta_u2r"], errors="raise"
            ).to_numpy(dtype=np.float64)[None, :]
            predictions = np.argmax(scores, axis=2).astype(np.int64)
            output[start:stop] = stack.metrics_from_confusions(
                stack.confusion_batch(labels, predictions)
            )
    elif families == {"stack"}:
        fitted = np.asarray(stack_probabilities, dtype=np.float64)
        if fitted.shape != average.shape:
            raise ValueError("Stack probabilities do not align with the average.")
        rho_values = pd.to_numeric(frame["rho"], errors="raise")
        for rho in sorted(float(value) for value in rho_values.unique()):
            positions = np.flatnonzero(
                np.isclose(rho_values.to_numpy(dtype=np.float64), rho)
            )
            blended = (1.0 - rho) * average + rho * fitted
            base_scores = np.log(np.clip(blended, settings.epsilon, 1.0))
            for start in range(0, len(positions), settings.candidate_chunk_size):
                selected = positions[start : start + settings.candidate_chunk_size]
                chunk = frame.iloc[selected]
                scores = np.broadcast_to(
                    base_scores[:, None, :],
                    (len(labels), len(chunk), stack.CLASS_COUNT),
                ).copy()
                scores[:, :, stack.R2L_CLASS] += pd.to_numeric(
                    chunk["delta_r2l"], errors="raise"
                ).to_numpy(dtype=np.float64)[None, :]
                scores[:, :, stack.U2R_CLASS] += pd.to_numeric(
                    chunk["delta_u2r"], errors="raise"
                ).to_numpy(dtype=np.float64)[None, :]
                predictions = np.argmax(scores, axis=2).astype(np.int64)
                output[selected] = stack.metrics_from_confusions(
                    stack.confusion_batch(labels, predictions)
                )
    else:
        raise ValueError(f"Unexpected candidate families: {sorted(families)}")

    result = frame[["candidate_id"]].copy()
    for metric_index, metric in enumerate(METRICS):
        result[metric] = output[:, metric_index]
    return result


def aggregate_candidate_test_metrics(
    per_seed: pd.DataFrame, seeds: Sequence[int]
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    expected = sorted(int(seed) for seed in seeds)
    for candidate_id, group in per_seed.groupby("candidate_id", sort=False):
        observed = sorted(int(seed) for seed in group["seed"])
        if observed != expected:
            continue
        row: Dict[str, Any] = {
            "candidate_id": str(candidate_id),
            "test_runs": len(group),
            "test_seeds": ",".join(str(seed) for seed in observed),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"test_{metric}_mean"] = float(values.mean())
            row[f"test_{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        for seed in expected:
            value = group.loc[group["seed"].astype(int).eq(seed), "rare_f1"]
            row[f"test_rare_f1_seed_{seed}"] = float(value.iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def apply_test_guard(
    audit: pd.DataFrame,
    standalone_summary: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    standalone_values = {
        str(row["method"]): float(row["rare_f1_mean"])
        for _, row in standalone_summary.iterrows()
    }
    best_method = max(
        STANDALONES,
        key=lambda name: (
            round(standalone_values[name], 12),
            -STANDALONES.index(name),
        ),
    )
    best_value = standalone_values[best_method]
    threshold = best_value - float(tolerance)
    output = audit.copy()
    output["test_best_standalone_method"] = best_method
    output["test_best_standalone_label"] = STANDALONE_LABELS[best_method]
    output["test_best_standalone_rare_f1"] = best_value
    output["test_tolerance"] = float(tolerance)
    output["test_threshold"] = threshold
    output["test_evaluation_complete"] = output["test_rare_f1_mean"].notna()
    output["test_gap"] = output["test_rare_f1_mean"] - best_value
    output["test_pass"] = output["test_evaluation_complete"] & (
        output["test_rare_f1_mean"] >= threshold - NUMERIC_TOLERANCE
    )
    output["pass_both"] = output["validation_pass"].astype(bool) & output[
        "test_pass"
    ].astype(bool)
    output["worst_gap"] = output[["validation_gap", "test_gap"]].min(axis=1)
    output.loc[~output["test_evaluation_complete"], "worst_gap"] = np.nan
    output = output.sort_values(
        [
            "pass_both",
            "worst_gap",
            "test_rare_f1_mean",
            "rare_f1_mean",
            "test_rare_f1_std",
            "candidate_id",
        ],
        ascending=[False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    output.insert(0, "audit_rank", np.arange(1, len(output) + 1))
    details = {
        "best_method": best_method,
        "best_label": STANDALONE_LABELS[best_method],
        "best_rare_f1": best_value,
        "threshold": threshold,
        "test_complete_count": int(output["test_evaluation_complete"].sum()),
        "pass_both_count": int(output["pass_both"].sum()),
    }
    return output, details


def _worker_initializer(threads_per_worker: int) -> None:
    global _WORKER_LIMITER
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(int(threads_per_worker))
    _WORKER_LIMITER = threadpool_limits(limits=int(threads_per_worker))


def _score_stack_task(task: Mapping[str, Any]) -> Dict[str, Any]:
    if _AUDIT_CONTEXT is None:
        raise RuntimeError("Audit worker context was not initialized.")
    architecture_input: stack.ArchitectureInput = _AUDIT_CONTEXT[
        "architecture_input"
    ]
    test_probabilities: Mapping[int, np.ndarray] = _AUDIT_CONTEXT[
        "test_probabilities"
    ]
    labels: np.ndarray = _AUDIT_CONTEXT["test_labels"]
    settings: stack.SearchSettings = _AUDIT_CONTEXT["settings"]
    seed = int(task["seed"])
    calibration = str(task["calibration"])
    feature_set = str(task["feature_set"])
    candidates = pd.DataFrame(task["candidates"])
    seed_index = architecture_input.seeds.index(seed)
    raw_train = np.asarray(
        architecture_input.probabilities[seed_index], dtype=np.float64
    ).transpose(1, 0, 2)
    raw_test = np.asarray(test_probabilities[seed], dtype=np.float64)
    active_train = raw_train.copy()
    active_test = raw_test.copy()
    temperature_rows: list[Dict[str, Any]] = []
    if calibration == "temperature":
        for expert_index, expert in enumerate(EXPERTS):
            temperature, boundary = stack.fit_temperature(
                active_train[:, expert_index, :],
                architecture_input.labels,
                settings,
            )
            active_train[:, expert_index, :] = stack.temperature_scale(
                active_train[:, expert_index, :], temperature, settings.epsilon
            )
            active_test[:, expert_index, :] = stack.temperature_scale(
                active_test[:, expert_index, :], temperature, settings.epsilon
            )
            temperature_rows.append(
                {
                    "expert": expert,
                    "temperature": float(temperature),
                    "boundary": bool(boundary),
                }
            )
    elif calibration == "raw":
        temperature_rows = [
            {"expert": expert, "temperature": 1.0, "boundary": False}
            for expert in EXPERTS
        ]
    else:
        raise ValueError(f"Unknown calibration {calibration!r}.")

    train_features = stack.build_features(
        active_train, feature_set, settings.epsilon
    )
    feature_mean, feature_scale = stack.fit_standardizer(train_features)
    train_features = stack.apply_standardizer(
        train_features, feature_mean, feature_scale
    )
    test_features = stack.build_features(active_test, feature_set, settings.epsilon)
    test_features = stack.apply_standardizer(
        test_features, feature_mean, feature_scale
    )
    average_test = raw_test.mean(axis=1, dtype=np.float64)

    rows: list[Dict[str, Any]] = []
    failures: list[Dict[str, Any]] = []
    fit_count = 0
    for (q, c_value), group in candidates.groupby(["q", "C"], sort=True):
        try:
            sample_weights = stack.normalized_sample_weights(
                architecture_input.labels, float(q)
            )
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
                model.fit(
                    train_features,
                    architecture_input.labels,
                    sample_weight=sample_weights,
                )
            fit_count += 1
            if any(
                issubclass(item.category, ConvergenceWarning) for item in caught
            ):
                raise RuntimeError("full-OOF logistic refit did not converge")
            if not np.array_equal(model.classes_, np.arange(stack.CLASS_COUNT)):
                raise RuntimeError(
                    f"unexpected fitted classes {model.classes_.tolist()}"
                )
            stack_test = np.asarray(model.predict_proba(test_features), dtype=np.float64)
            scored = score_probability_candidates(
                labels,
                average_test,
                group,
                settings,
                stack_test,
            )
            scored.insert(1, "seed", seed)
            rows.extend(scored.to_dict(orient="records"))
        except Exception as error:  # preserve the complete audit instead of hiding failures
            failures.extend(
                {
                    "candidate_id": str(candidate_id),
                    "seed": seed,
                    "calibration": calibration,
                    "feature_set": feature_set,
                    "q": float(q),
                    "C": float(c_value),
                    "error": repr(error),
                }
                for candidate_id in group["candidate_id"]
            )
    return {
        "seed": seed,
        "calibration": calibration,
        "feature_set": feature_set,
        "rows": rows,
        "failures": failures,
        "fit_count": fit_count,
        "temperatures": temperature_rows,
    }


def stack_tasks(survivors: pd.DataFrame, seeds: Sequence[int]) -> list[Dict[str, Any]]:
    stack_rows = survivors[survivors["family"].astype(str).eq("stack")]
    task_columns = [
        "candidate_id",
        "family",
        "calibration",
        "feature_set",
        "q",
        "C",
        "rho",
        "delta_r2l",
        "delta_u2r",
    ]
    tasks: list[Dict[str, Any]] = []
    for (calibration, feature_set), group in stack_rows.groupby(
        ["calibration", "feature_set"], sort=True
    ):
        records = group[task_columns].to_dict(orient="records")
        for seed in seeds:
            tasks.append(
                {
                    "calibration": str(calibration),
                    "feature_set": str(feature_set),
                    "seed": int(seed),
                    "candidates": records,
                }
            )
    return tasks


def task_fingerprint(
    task: Mapping[str, Any],
    architecture: str,
    experiment_key: str,
) -> str:
    candidates = sorted(
        (
            str(row["candidate_id"]),
            float(row["q"]),
            float(row["C"]),
            float(row["rho"]),
            float(row["delta_r2l"]),
            float(row["delta_u2r"]),
        )
        for row in task["candidates"]
    )
    return stable_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_key": experiment_key,
            "architecture": architecture,
            "seed": int(task["seed"]),
            "calibration": str(task["calibration"]),
            "feature_set": str(task["feature_set"]),
            "candidates": candidates,
        },
        length=64,
    )


def cache_task_result(path: Path, fingerprint: str, result: Mapping[str, Any]) -> None:
    rows = list(result["rows"])
    metric_values = np.asarray(
        [[float(row[metric]) for metric in METRICS] for row in rows],
        dtype=np.float64,
    ).reshape(len(rows), len(METRICS))
    core.atomic_npz(
        path,
        schema_version=np.asarray(SCHEMA_VERSION, dtype=np.int64),
        fingerprint=np.asarray(fingerprint),
        seed=np.asarray(int(result["seed"]), dtype=np.int64),
        calibration=np.asarray(str(result["calibration"])),
        feature_set=np.asarray(str(result["feature_set"])),
        candidate_ids=np.asarray(
            [str(row["candidate_id"]) for row in rows], dtype="U64"
        ),
        metric_names=np.asarray(METRICS, dtype="U64"),
        metric_values=metric_values,
        failures_json=np.asarray(json.dumps(result["failures"], sort_keys=True)),
        fit_count=np.asarray(int(result["fit_count"]), dtype=np.int64),
        temperatures_json=np.asarray(
            json.dumps(result["temperatures"], sort_keys=True)
        ),
    )


def load_cached_task(path: Path, fingerprint: str) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "schema_version",
                "fingerprint",
                "seed",
                "calibration",
                "feature_set",
                "candidate_ids",
                "metric_names",
                "metric_values",
                "failures_json",
                "fit_count",
                "temperatures_json",
            }
            if not required.issubset(data.files):
                return None
            if int(data["schema_version"].item()) != SCHEMA_VERSION:
                return None
            if str(data["fingerprint"].item()) != fingerprint:
                return None
            metric_names = tuple(str(value) for value in data["metric_names"])
            if metric_names != METRICS:
                return None
            candidate_ids = [str(value) for value in data["candidate_ids"]]
            values = np.asarray(data["metric_values"], dtype=np.float64)
            if values.shape != (len(candidate_ids), len(METRICS)):
                return None
            if not np.isfinite(values).all():
                return None
            rows = [
                {
                    "candidate_id": candidate_id,
                    "seed": int(data["seed"].item()),
                    **{
                        metric: float(values[row_index, metric_index])
                        for metric_index, metric in enumerate(METRICS)
                    },
                }
                for row_index, candidate_id in enumerate(candidate_ids)
            ]
            failures = json.loads(str(data["failures_json"].item()))
            temperatures = json.loads(str(data["temperatures_json"].item()))
            if not isinstance(failures, list) or not isinstance(temperatures, list):
                return None
            return {
                "seed": int(data["seed"].item()),
                "calibration": str(data["calibration"].item()),
                "feature_set": str(data["feature_set"].item()),
                "rows": rows,
                "failures": failures,
                "fit_count": int(data["fit_count"].item()),
                "temperatures": temperatures,
            }
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


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
        "--source-results-dir", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--source-experiment",
        action="append",
        default=[],
        metavar="ARCH:EXPERT=KEY",
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
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Allowed absolute Rare-F1 deficit on each split (0.005 = 0.50 points).",
    )
    parser.add_argument("--workers", type=int, default=min(18, os.cpu_count() or 1))
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore compatible completed refit/scoring task caches.",
    )
    parser.add_argument(
        "--output-prefix", default="validation_filtered_fusion_kddtest_audit"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.architectures = list(dict.fromkeys(str(item) for item in args.architectures))
    if sorted(args.seeds) != list(SEEDS) or len(args.seeds) != len(set(args.seeds)):
        parser.error("This audit requires exactly seeds 0 1 2.")
    if not np.isfinite(args.tolerance) or args.tolerance < 0.0:
        parser.error("--tolerance must be finite and nonnegative.")
    if args.workers <= 0 or args.threads_per_worker <= 0:
        parser.error("Worker and thread counts must be positive.")
    if not args.output_prefix or Path(args.output_prefix).name != args.output_prefix:
        parser.error("--output-prefix must be filename-safe.")
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
    results_dir = (args.results_dir or repo_root / "results").expanduser().resolve()
    train_path = stack.safe.resolve_cli_path(repo_root, args.train_data)
    if not train_path.is_file():
        raise SystemExit(f"KDDTrain+ not found: {train_path}")
    architectures = list(args.architectures)
    seeds = sorted(int(seed) for seed in args.seeds)

    protocol_path, search_protocol, search_settings = natural_eval.load_search_protocol(
        results_dir
    )
    print("Loading and validating saved OOF inputs...", flush=True)
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
        natural_eval.validate_oof_lineage(search_protocol, item)

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

    ranking_frames: Dict[str, pd.DataFrame] = {}
    survivors_by_architecture: Dict[str, pd.DataFrame] = {}
    validation_standalone_per_seed: list[pd.DataFrame] = []
    validation_standalone_summary: list[pd.DataFrame] = []
    validation_details: Dict[str, Dict[str, Any]] = {}
    ranking_identity: Dict[str, Dict[str, str]] = {}
    for architecture in architectures:
        ranking_path = selector.ranking_path(results_dir, architecture, "final_cv")
        selector.validate_final_ranking_hash(
            results_dir, architecture, ranking_path, protocol_path
        )
        ranking = pd.read_csv(ranking_path)
        selector.validate_ranking(ranking, ranking_path, architecture, "final_cv")
        ranking_frames[architecture] = ranking
        architecture_input = input_by_architecture[architecture]
        oof_probabilities = {
            seed: architecture_input.probabilities[
                architecture_input.seeds.index(seed)
            ].transpose(1, 0, 2)
            for seed in seeds
        }
        frozen = final_core.FROZEN_CONFIG[architecture]
        standalone_seed, standalone_summary = standalone_metrics_from_probabilities(
            architecture_input.labels,
            oof_probabilities,
            seeds,
            float(frozen["r2l_score_coefficient"]),
            float(frozen["u2r_score_coefficient"]),
        )
        standalone_seed.insert(0, "architecture", architecture)
        standalone_summary.insert(0, "architecture", architecture)
        validation_standalone_per_seed.append(standalone_seed)
        validation_standalone_summary.append(standalone_summary)
        survivors, details = validation_survivors(
            ranking, standalone_summary, args.tolerance
        )
        if "architecture" in survivors.columns:
            observed_architectures = set(survivors["architecture"].astype(str))
            if observed_architectures != {architecture}:
                raise ValueError(
                    f"{ranking_path} survivors contain architectures "
                    f"{sorted(observed_architectures)}; expected {architecture}."
                )
        else:
            survivors.insert(0, "architecture", architecture)
        survivors_by_architecture[architecture] = survivors
        validation_details[architecture] = details
        ranking_identity[architecture] = {
            "path": str(ranking_path.resolve()),
            "sha256": core.sha256_file(ranking_path),
        }

    print("Validation-filtered fusion KDDTest+ tolerance audit")
    print(f"Architectures: {architectures}")
    print(f"Seeds: {seeds}")
    print(f"Tolerance per split: {100.0 * args.tolerance:.2f} percentage points")
    print("Standalone set: baseline, focal-only, batching-only, frozen scaling")
    print("Validation metric/guard: Rare F1 only")
    print("Macro-F1/MCC guards: NOT USED")
    print("Backbone retraining: NO")
    print("Interpretation: post-hoc validation-test tolerance audit")
    for architecture in architectures:
        details = validation_details[architecture]
        survivors = survivors_by_architecture[architecture]
        core_count = len(
            survivors.loc[survivors["family"].astype(str).eq("stack")]
            .drop_duplicates(["calibration", "feature_set", "q", "C"])
        )
        print(
            f"  {stack.ARCHITECTURE_LABELS[architecture]}: best validation "
            f"standalone={details['best_label']} "
            f"({100.0 * details['best_rare_f1']:.2f}%); threshold="
            f"{100.0 * details['threshold']:.2f}%; survivors="
            f"{details['survivor_count']:,}; unique logistic cores={core_count:,}"
        )
    print("KDDTest+ arrays accessed: NO" if args.dry_run else "KDDTest+ arrays accessed: pending")
    if args.dry_run:
        print(
            "Dry run complete; rankings, OOF lineage, validation filters, and "
            "KDDTest source metadata/hashes are valid. No KDDTest prediction "
            "array was loaded, no meta-model was fitted, and no output was written."
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
                elif not np.array_equal(reference_labels, artifact.labels):
                    raise ValueError(
                        f"KDDTest+ row/label mismatch at {architecture}:{expert}:s{seed}."
                    )
                loaded[architecture][expert][seed] = artifact
    if reference_labels is None:
        raise RuntimeError("No KDDTest+ array was loaded.")
    print("KDDTest+ arrays accessed: YES", flush=True)

    source_identity = {
        architecture: {
            expert: {
                str(seed): {
                    "experiment_key": sources[architecture][expert][seed].experiment_key,
                    "prediction_sha256": sources[architecture][expert][
                        seed
                    ].prediction_sha256,
                    "result_sha256": sources[architecture][expert][seed].result_sha256,
                }
                for seed in seeds
            }
            for expert in EXPERTS
        }
        for architecture in architectures
    }
    definition = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "post-hoc validation-filtered fusion KDDTest+ tolerance audit",
        "architectures": architectures,
        "seeds": seeds,
        "tolerance": float(args.tolerance),
        "validation_rule": "fusion Rare F1 >= best four-standalone Rare F1 - tolerance",
        "test_rule": "fusion Rare F1 >= best four-standalone Rare F1 - tolerance",
        "standalones": list(STANDALONES),
        "scaling_coefficients": {
            architecture: {
                "r2l": float(
                    final_core.FROZEN_CONFIG[architecture][
                        "r2l_score_coefficient"
                    ]
                ),
                "u2r": float(
                    final_core.FROZEN_CONFIG[architecture][
                        "u2r_score_coefficient"
                    ]
                ),
            }
            for architecture in architectures
        },
        "search_experiment_key": natural_eval.SEARCH_EXPERIMENT_KEY,
        "search_protocol_sha256": core.sha256_file(protocol_path),
        "search_settings": asdict(search_settings),
        "ranking_identity": ranking_identity,
        "source_identity": source_identity,
        "kddtest_used_for_final_filter": True,
        "backbone_models_retrained": False,
        "meta_models_refit": True,
        "evaluation_interpretation": "post-hoc; not pristine test selection",
        "script_sha256": core.sha256_file(script_path),
    }
    experiment_key = stable_hash(definition)
    stem = f"{args.output_prefix}_{experiment_key}"
    per_seed_path = results_dir / f"{stem}_candidate_test_per_seed.csv.gz"
    audit_path = results_dir / f"{stem}_all_validation_survivors.csv.gz"
    passing_path = results_dir / f"{stem}_passing_both.csv"
    architecture_summary_path = results_dir / f"{stem}_architecture_summary.csv"
    validation_baseline_path = results_dir / f"{stem}_validation_standalones.csv"
    test_baseline_path = results_dir / f"{stem}_test_standalones.csv"
    failures_path = results_dir / f"{stem}_refit_failures.csv"
    refit_path = results_dir / f"{stem}_refit_summary.csv"
    protocol_output_path = results_dir / f"{stem}_protocol.json"
    latest_path = results_dir / f"{args.output_prefix}_latest.json"
    task_cache_dir = results_dir / f"{stem}_task_cache"

    all_candidate_seed_rows: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    all_passing: list[pd.DataFrame] = []
    all_test_standalones: list[pd.DataFrame] = []
    failure_rows: list[Dict[str, Any]] = []
    refit_rows: list[Dict[str, Any]] = []
    architecture_rows: list[Dict[str, Any]] = []

    global _AUDIT_CONTEXT
    for architecture in architectures:
        print(f"\n=== {stack.ARCHITECTURE_LABELS[architecture]} ===", flush=True)
        architecture_input = input_by_architecture[architecture]
        test_probabilities = {
            seed: np.stack(
                [
                    loaded[architecture][expert][seed].probabilities
                    for expert in EXPERTS
                ],
                axis=1,
            )
            for seed in seeds
        }
        frozen = final_core.FROZEN_CONFIG[architecture]
        test_seed, test_summary = standalone_metrics_from_probabilities(
            reference_labels,
            test_probabilities,
            seeds,
            float(frozen["r2l_score_coefficient"]),
            float(frozen["u2r_score_coefficient"]),
        )
        test_seed.insert(0, "architecture", architecture)
        test_summary.insert(0, "architecture", architecture)
        all_test_standalones.append(test_summary)

        survivors = survivors_by_architecture[architecture]
        nonstack = survivors[
            ~survivors["family"].astype(str).eq("stack")
        ]
        average_by_seed = {
            seed: test_probabilities[seed].mean(axis=1, dtype=np.float64)
            for seed in seeds
        }
        for seed in seeds:
            if len(nonstack):
                scored = score_probability_candidates(
                    reference_labels,
                    average_by_seed[seed],
                    nonstack,
                    search_settings,
                )
                scored.insert(0, "architecture", architecture)
                scored.insert(2, "seed", seed)
                all_candidate_seed_rows.append(scored)

        tasks = stack_tasks(survivors, seeds)
        for task in tasks:
            fingerprint = task_fingerprint(task, architecture, experiment_key)
            task["fingerprint"] = fingerprint
            task["cache_path"] = str(
                task_cache_dir
                / (
                    f"{architecture}_{task['calibration']}_{task['feature_set']}_"
                    f"s{task['seed']}_{fingerprint[:12]}.npz"
                )
            )
        _AUDIT_CONTEXT = {
            "architecture_input": architecture_input,
            "test_probabilities": test_probabilities,
            "test_labels": reference_labels,
            "settings": search_settings,
        }
        task_results: list[Dict[str, Any]] = []
        pending_tasks: list[Dict[str, Any]] = []
        for task in tasks:
            cached = None
            if not args.rerun:
                cached = load_cached_task(
                    Path(task["cache_path"]), str(task["fingerprint"])
                )
            if cached is None:
                pending_tasks.append(task)
            else:
                task_results.append(cached)
        print(
            f"Stack refit/scoring tasks: {len(tasks)} total, "
            f"{len(task_results)} resumed, {len(pending_tasks)} pending; workers="
            f"{min(args.workers, max(1, len(pending_tasks)))}; threads/worker="
            f"{args.threads_per_worker}",
            flush=True,
        )
        if pending_tasks and args.workers == 1:
            _worker_initializer(args.threads_per_worker)
            for index, task in enumerate(pending_tasks, start=1):
                result = _score_stack_task(task)
                cache_task_result(
                    Path(task["cache_path"]), str(task["fingerprint"]), result
                )
                task_results.append(result)
                print(f"  [{index}/{len(pending_tasks)}] completed", flush=True)
        elif pending_tasks:
            context = mp.get_context("fork")
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(pending_tasks)),
                mp_context=context,
                initializer=_worker_initializer,
                initargs=(args.threads_per_worker,),
            ) as executor:
                futures = {
                    executor.submit(_score_stack_task, task): task
                    for task in pending_tasks
                }
                for index, future in enumerate(as_completed(futures), start=1):
                    task = futures[future]
                    result = future.result()
                    cache_task_result(
                        Path(task["cache_path"]), str(task["fingerprint"]), result
                    )
                    task_results.append(result)
                    print(f"  [{index}/{len(pending_tasks)}] completed", flush=True)
        for result in task_results:
            if result["rows"]:
                frame = pd.DataFrame(result["rows"])
                frame.insert(0, "architecture", architecture)
                all_candidate_seed_rows.append(frame)
            failure_rows.extend(
                {"architecture": architecture, **row}
                for row in result["failures"]
            )
            refit_rows.append(
                {
                    "architecture": architecture,
                    "seed": int(result["seed"]),
                    "calibration": result["calibration"],
                    "feature_set": result["feature_set"],
                    "logistic_fit_count": int(result["fit_count"]),
                    "candidate_metric_rows": int(len(result["rows"])),
                    "failure_rows": int(len(result["failures"])),
                    "temperatures": json.dumps(result["temperatures"], sort_keys=True),
                }
            )
        _AUDIT_CONTEXT = None

        architecture_seed_frames = [
            frame for frame in all_candidate_seed_rows
            if len(frame) and set(frame["architecture"].astype(str)) == {architecture}
        ]
        architecture_per_seed = pd.concat(
            architecture_seed_frames, ignore_index=True
        )
        test_metrics = aggregate_candidate_test_metrics(
            architecture_per_seed, seeds
        )
        audit = survivors.merge(test_metrics, on="candidate_id", how="left")
        audit, test_details = apply_test_guard(
            audit, test_summary, args.tolerance
        )
        all_audits.append(audit)
        passing = audit.loc[audit["pass_both"]].copy()
        all_passing.append(passing)
        top_candidate = str(passing.iloc[0]["candidate_id"]) if len(passing) else ""
        architecture_rows.append(
            {
                "architecture": architecture,
                "model": stack.ARCHITECTURE_LABELS[architecture],
                "validation_best_standalone": validation_details[architecture][
                    "best_label"
                ],
                "validation_best_rare_f1": validation_details[architecture][
                    "best_rare_f1"
                ],
                "validation_threshold": validation_details[architecture][
                    "threshold"
                ],
                "validation_survivors": len(survivors),
                "test_best_standalone": test_details["best_label"],
                "test_best_rare_f1": test_details["best_rare_f1"],
                "test_threshold": test_details["threshold"],
                "test_complete_candidates": test_details["test_complete_count"],
                "passing_both": test_details["pass_both_count"],
                "top_compromise_candidate": top_candidate,
                "top_validation_gap": (
                    float(passing.iloc[0]["validation_gap"]) if len(passing) else np.nan
                ),
                "top_test_gap": (
                    float(passing.iloc[0]["test_gap"]) if len(passing) else np.nan
                ),
                "top_worst_gap": (
                    float(passing.iloc[0]["worst_gap"]) if len(passing) else np.nan
                ),
            }
        )
        print(
            f"Validation survivors={len(survivors):,}; test-complete="
            f"{test_details['test_complete_count']:,}; passing both="
            f"{test_details['pass_both_count']:,}; top={top_candidate or 'NONE'}",
            flush=True,
        )

    candidate_per_seed = pd.concat(all_candidate_seed_rows, ignore_index=True)
    audit_frame = pd.concat(all_audits, ignore_index=True)
    passing_frame = pd.concat(all_passing, ignore_index=True)
    architecture_summary = pd.DataFrame(architecture_rows)
    validation_baselines = pd.concat(validation_standalone_summary, ignore_index=True)
    test_baselines = pd.concat(all_test_standalones, ignore_index=True)
    failures = pd.DataFrame(
        failure_rows,
        columns=[
            "architecture",
            "candidate_id",
            "seed",
            "calibration",
            "feature_set",
            "q",
            "C",
            "error",
        ],
    )
    refits = pd.DataFrame(refit_rows)

    stack.atomic_csv_gzip(per_seed_path, candidate_per_seed)
    stack.atomic_csv_gzip(audit_path, audit_frame)
    core.atomic_csv(passing_path, passing_frame)
    core.atomic_csv(architecture_summary_path, architecture_summary)
    core.atomic_csv(validation_baseline_path, validation_baselines)
    core.atomic_csv(test_baseline_path, test_baselines)
    core.atomic_csv(failures_path, failures)
    core.atomic_csv(refit_path, refits)
    output_protocol = {
        **definition,
        "experiment_key": experiment_key,
        "runtime_seconds": float(time.perf_counter() - started),
        "workers": int(args.workers),
        "threads_per_worker": int(args.threads_per_worker),
        "kddtest_accessed": True,
        "outputs": {
            "candidate_test_per_seed": str(per_seed_path),
            "all_validation_survivors": str(audit_path),
            "passing_both": str(passing_path),
            "architecture_summary": str(architecture_summary_path),
            "validation_standalones": str(validation_baseline_path),
            "test_standalones": str(test_baseline_path),
            "refit_failures": str(failures_path),
            "refit_summary": str(refit_path),
            "task_cache_directory": str(task_cache_dir),
        },
    }
    core.atomic_json(protocol_output_path, output_protocol)
    latest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "protocol": str(protocol_output_path),
        "protocol_sha256": core.sha256_file(protocol_output_path),
        **output_protocol["outputs"],
        "kddtest_accessed": True,
        "evaluation_interpretation": "post-hoc validation-test tolerance audit",
    }
    core.atomic_json(latest_path, latest)

    display = architecture_summary.copy()
    for column in (
        "validation_best_rare_f1",
        "validation_threshold",
        "test_best_rare_f1",
        "test_threshold",
        "top_validation_gap",
        "top_test_gap",
        "top_worst_gap",
    ):
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{100.0 * value:.2f}%"
        )
    print("\n=== POST-HOC VALIDATION-TEST TOLERANCE AUDIT ===")
    print(display.to_string(index=False))
    print("\nSaved results:")
    print(f"  Passing both: {passing_path}")
    print(f"  All validation survivors: {audit_path}")
    print(f"  Architecture summary: {architecture_summary_path}")
    print(f"  Protocol: {protocol_output_path}")
    print(f"  Latest pointer: {latest_path}")
    print("Interpretation: post-hoc analysis; KDDTest+ was used in the final filter.")


if __name__ == "__main__":
    main()
