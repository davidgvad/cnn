"""Evaluate frozen simple-average and SAFE-Stack decisions on KDDTest+.

This is a CPU-only postprocessor for the already completed final neural-model
runs.  It reuses the raw KDDTest+ probability vectors produced by three fixed
experts for each architecture and seed:

* General: cross-entropy with ordinary shuffled batches (``baseline``).
* Focal: class-balanced focal loss with ordinary batches (``focal_only``).
* Batching: cross-entropy with minority-guaranteed batches (``batch_only``).

The script evaluates two predeclared decision policies without retraining:

* arithmetic mean of the three unscaled probability vectors, then argmax;
* the architecture-specific SAFE-Stack configuration frozen on KDDTrain+ OOF
  predictions before this final evaluation.

No configuration is selected, ranked, or changed using KDDTest+.  Historical
``*_latest.json`` pointers are deliberately not used for source final runs,
because architecture-subset runs can overwrite them.  Immutable hashed result
artifacts are discovered instead, and ambiguous distinct runs cause an error.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import run_final_baseline_vs_full_kddtest_4gpu as final_core
import run_no_ctgan_model_ablation_4gpu as core
import tune_conv2d_safe_stack_fusion as fusion


SCHEMA_VERSION = 1
ARCHITECTURES = ("conv2d", "conv1d", "transformer", "mlp")
EXPERTS = tuple(fusion.EXPERTS)
EXPERT_VARIANTS = {
    "general": "baseline",
    "focal": "focal_only",
    "batching": "batch_only",
}
METHODS = ("simple_average", "safe_stack")
METHOD_LABELS = {
    "simple_average": "Simple probability average",
    "safe_stack": "Frozen SAFE-Stack fusion",
}
METRICS = tuple(core.METRICS)
EXPECTED_EPOCHS = 25
EXPECTED_BATCH_SIZE = 256
EXPECTED_MINORITY_PER_BATCH = 1

# These configurations were selected using KDDTrain+ pooled OOF predictions.
# Pinning the immutable experiment/candidate identities prevents a mutable
# latest-results pointer from silently changing a final-test decision policy.
FROZEN_SAFE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "conv2d": {
        "experiment_key": "e50b0615c0cf",
        "candidate_id": 5316,
        "base_expert": "general",
        "r2l_weights": (0.0, 0.0, 1.0),
        "u2r_weights": (0.0, 0.25, 0.75),
        "r2l_margin": 0.0,
        "u2r_margin": 0.025,
        "minimum_support": 2,
    },
    "conv1d": {
        "experiment_key": "c9391f0b9098",
        "candidate_id": 5353,
        "base_expert": "focal",
        "r2l_weights": (0.0, 0.0, 1.0),
        "u2r_weights": (0.5, 0.0, 0.5),
        "r2l_margin": 0.025,
        "u2r_margin": 0.10,
        "minimum_support": 2,
    },
    "transformer": {
        "experiment_key": "75a67bdc44c7",
        "candidate_id": 54,
        "base_expert": "batching",
        "r2l_weights": (1.0, 0.0, 0.0),
        "u2r_weights": (0.0, 1.0, 0.0),
        "r2l_margin": 0.0,
        "u2r_margin": 0.15,
        "minimum_support": 2,
    },
    "mlp": {
        "experiment_key": "c6964485d79e",
        "candidate_id": 3004,
        "base_expert": "focal",
        "r2l_weights": (0.25, 0.25, 0.50),
        "u2r_weights": (1.0, 0.0, 0.0),
        "r2l_margin": 0.0,
        "u2r_margin": 0.15,
        "minimum_support": 2,
    },
}


@dataclass(frozen=True)
class SelectedSource:
    architecture: str
    expert: str
    variant: str
    seed: int
    experiment_key: str
    result_path: Path
    result_sha256: str
    prediction_path: Path
    prediction_sha256: str
    cache_metadata_path: Path
    cache_metadata_sha256: str
    cache_identity: Dict[str, Any]
    result: Dict[str, Any]


@dataclass(frozen=True)
class LoadedSource:
    selected: SelectedSource
    labels: np.ndarray
    probabilities: np.ndarray
    raw_predictions: np.ndarray


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        normalized = str(path.expanduser().resolve())
        if normalized not in seen:
            seen.add(normalized)
            output.append(Path(normalized))
    return output


def resolve_prediction_path(
    recorded_value: str | Path,
    result_path: Path,
    source_roots: Sequence[Path],
) -> Path:
    recorded = Path(recorded_value).expanduser()
    run_directory = result_path.parent
    run_prefix = (
        run_directory.name[: -len("_runs")]
        if run_directory.name.endswith("_runs")
        else run_directory.name
    )
    basename = recorded.name or f"{result_path.stem}.npz"
    candidates = [recorded]
    if not recorded.is_absolute():
        candidates.extend(
            [result_path.parent / recorded, result_path.parents[1] / recorded]
        )
    candidates.append(run_directory.parent / f"{run_prefix}_predictions" / basename)
    existing = [path.resolve() for path in unique_paths(candidates) if path.is_file()]
    if existing:
        return existing[0]

    matches: list[Path] = []
    for root in source_roots:
        matches.extend(
            path.resolve() for path in root.glob(f"**/{basename}") if path.is_file()
        )
    matches = unique_paths(matches)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not resolve prediction {recorded_value!s} from {result_path}."
        )
    raise RuntimeError(
        f"Prediction basename {basename!r} is ambiguous: "
        + ", ".join(str(path) for path in matches)
    )


def cache_metadata_path_for_result(result_path: Path) -> Path:
    run_directory = result_path.parent
    if not run_directory.name.endswith("_runs"):
        raise ValueError(f"Unexpected final-run directory: {run_directory}")
    prefix = run_directory.name[: -len("_runs")]
    return run_directory.parent / f"{prefix}_feature_cache.json"


def cache_identity(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    required = (
        "train_sha256",
        "test_sha256",
        "feature_order_sha256",
        "train_rows",
        "test_rows",
        "feature_count",
        "train_class_counts",
        "test_class_counts",
        "preprocessor_fit_partition",
        "test_fit_involvement",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise KeyError(f"Feature-cache metadata is missing fields: {missing}")
    return {key: metadata[key] for key in required}


def parse_source_overrides(values: Sequence[str]) -> Dict[tuple[str, str], str]:
    overrides: Dict[tuple[str, str], str] = {}
    for value in values:
        try:
            left, experiment_key = value.split("=", 1)
            architecture, expert = left.split(":", 1)
        except ValueError as error:
            raise ValueError(
                "--source-experiment must use ARCHITECTURE:EXPERT=EXPERIMENT_KEY"
            ) from error
        architecture = architecture.strip().lower()
        expert = expert.strip().lower()
        experiment_key = experiment_key.strip()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown override architecture: {architecture!r}")
        if expert not in EXPERTS:
            raise ValueError(f"Unknown override expert: {expert!r}")
        if not experiment_key:
            raise ValueError("Source experiment key cannot be empty.")
        key = (architecture, expert)
        if key in overrides:
            raise ValueError(f"Duplicate source override for {architecture}:{expert}.")
        overrides[key] = experiment_key
    return overrides


def load_frozen_safe_config(
    results_dir: Path, architecture: str
) -> tuple[Path, Dict[str, Any]]:
    expected = FROZEN_SAFE_CONFIGS[architecture]
    config_path = results_dir / (
        f"{architecture}_safe_stack_fusion_"
        f"{expected['experiment_key']}_best_config.json"
    )
    best = read_json(config_path)
    scalar_expected = {
        "architecture": architecture,
        "experiment_key": expected["experiment_key"],
        "candidate_id": expected["candidate_id"],
        "selected_base": expected["base_expert"],
        "minimum_support": expected["minimum_support"],
    }
    for key, value in scalar_expected.items():
        if best.get(key) != value:
            raise ValueError(
                f"Frozen config mismatch in {config_path}: {key}={best.get(key)!r}, "
                f"expected {value!r}."
            )
    for rare_class in ("r2l", "u2r"):
        observed = np.asarray(
            [best[f"{rare_class}_weights"][expert] for expert in EXPERTS],
            dtype=np.float64,
        )
        wanted = np.asarray(expected[f"{rare_class}_weights"], dtype=np.float64)
        if not np.allclose(observed, wanted, atol=1e-12, rtol=0.0):
            raise ValueError(
                f"Frozen {rare_class.upper()} weights in {config_path} are "
                f"{observed.tolist()}, expected {wanted.tolist()}."
            )
        margin = float(best[f"{rare_class}_margin"])
        if not np.isclose(
            margin,
            float(expected[f"{rare_class}_margin"]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(
                f"Frozen {rare_class.upper()} margin in {config_path} is {margin}, "
                f"expected {expected[f'{rare_class}_margin']}."
            )
    if best.get("kddtest_accessed") is not False:
        raise ValueError(f"{config_path} is not a KDDTrain+-only selection artifact.")
    if best.get("base_rare_predictions_preserved") is not True:
        raise ValueError(f"{config_path} does not preserve base rare predictions.")
    if best.get("support_definition") != "rare class is in an expert's stable top two":
        raise ValueError(f"{config_path} has an unexpected support definition.")
    return config_path.resolve(), best


def candidate_result_paths(source_roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in source_roots:
        paths.extend(root.glob("final_*_kddtest_*_runs/*.json"))
    return sorted(unique_paths(path for path in paths if path.is_file()))


def validate_source_metadata(
    result_path: Path,
    result: Mapping[str, Any],
    architecture: str,
    expert: str,
    seed: int,
    safe_best: Mapping[str, Any],
) -> None:
    expected_variant = EXPERT_VARIANTS[expert]
    expected = {
        "schema_version": final_core.SCHEMA_VERSION,
        "architecture": architecture,
        "variant": expected_variant,
        "seed": int(seed),
        "epochs_requested": EXPECTED_EPOCHS,
        "epochs_completed": EXPECTED_EPOCHS,
        "batch_size": EXPECTED_BATCH_SIZE,
        "evaluation_partition": "untouched KDDTest+",
        "kddtest_used_for_selection": False,
        "ctgan_used": False,
        "score_scaling_used": False,
        "decision_policy": "raw_argmax",
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"Invalid source {result_path}: {key}={result.get(key)!r}, "
                f"expected {value!r}."
            )
    expected_parameters = int(
        final_core.FROZEN_CONFIG[architecture]["backbone"]["expected_parameters"]
    )
    if int(result.get("model_parameters", -1)) != expected_parameters:
        raise ValueError(f"{result_path} has an unexpected model parameter count.")

    if expert == "general":
        expected_loss = "sparse_categorical_crossentropy"
        expected_batching = "ordinary_shuffled"
        expected_minority = 0
    elif expert == "focal":
        expected_loss = "class_balanced_focal"
        expected_batching = "ordinary_shuffled"
        expected_minority = 0
        focal_best = safe_best.get("focal_best", {})
        expected_beta = float(focal_best.get("beta"))
        expected_gamma = float(focal_best.get("focal_gamma"))
        if not np.isclose(float(result.get("cb_beta")), expected_beta):
            raise ValueError(f"{result_path} uses the wrong focal beta.")
        if not np.isclose(float(result.get("focal_gamma")), expected_gamma):
            raise ValueError(f"{result_path} uses the wrong focal gamma.")
    else:
        expected_loss = "sparse_categorical_crossentropy"
        expected_batching = "minority_guaranteed_with_replacement"
        expected_minority = EXPECTED_MINORITY_PER_BATCH

    if result.get("loss") != expected_loss:
        raise ValueError(f"{result_path} has loss={result.get('loss')!r}.")
    if result.get("batching") != expected_batching:
        raise ValueError(f"{result_path} has batching={result.get('batching')!r}.")
    if int(result.get("minority_per_batch_per_class", -1)) != expected_minority:
        raise ValueError(f"{result_path} has the wrong minority batch guarantee.")


def select_sources(
    source_roots: Sequence[Path],
    architectures: Sequence[str],
    seeds: Sequence[int],
    safe_configs: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[tuple[str, str], str],
) -> Dict[str, Dict[str, Dict[int, SelectedSource]]]:
    requested_architectures = set(architectures)
    requested_seeds = set(int(seed) for seed in seeds)
    grouped: Dict[
        tuple[str, str, str], Dict[int, list[tuple[Path, Dict[str, Any]]]]
    ] = {}
    for path in candidate_result_paths(source_roots):
        try:
            result = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        architecture = str(result.get("architecture", ""))
        variant = str(result.get("variant", ""))
        seed_value = result.get("seed")
        if (
            architecture not in requested_architectures
            or variant not in EXPERT_VARIANTS.values()
        ):
            continue
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            continue
        if seed not in requested_seeds:
            continue
        expert = next(
            name for name, value in EXPERT_VARIANTS.items() if value == variant
        )
        experiment_key = str(result.get("experiment_key", ""))
        if not experiment_key:
            continue
        grouped.setdefault((architecture, expert, experiment_key), {}).setdefault(
            seed, []
        ).append((path.resolve(), result))

    selected: Dict[str, Dict[str, Dict[int, SelectedSource]]] = {
        architecture: {expert: {} for expert in EXPERTS}
        for architecture in architectures
    }
    for architecture in architectures:
        for expert in EXPERTS:
            override = overrides.get((architecture, expert))
            complete: list[tuple[str, Dict[int, tuple[Path, Dict[str, Any]]]]] = []
            available: list[str] = []
            for (
                candidate_architecture,
                candidate_expert,
                experiment_key,
            ), seed_rows in grouped.items():
                if candidate_architecture != architecture or candidate_expert != expert:
                    continue
                available.append(
                    f"{experiment_key}[{','.join(str(seed) for seed in sorted(seed_rows))}]"
                )
                if override is not None and experiment_key != override:
                    continue
                if set(seed_rows) != requested_seeds:
                    continue
                resolved_rows: Dict[int, tuple[Path, Dict[str, Any]]] = {}
                duplicate_conflict = False
                for seed, candidates in seed_rows.items():
                    signatures = {
                        str(result.get("prediction_sha256", ""))
                        for _, result in candidates
                    }
                    if len(signatures) != 1 or "" in signatures:
                        duplicate_conflict = True
                        break
                    resolved_rows[seed] = sorted(
                        candidates, key=lambda value: str(value[0])
                    )[0]
                if not duplicate_conflict:
                    complete.append((experiment_key, resolved_rows))

            if not complete:
                detail = ", ".join(sorted(available)) or "none"
                raise RuntimeError(
                    f"No complete source run for {architecture}:{expert}; "
                    f"available experiment groups: {detail}."
                )
            if len(complete) > 1:
                signatures = {
                    tuple(rows[seed][1]["prediction_sha256"] for seed in sorted(rows))
                    for _, rows in complete
                }
                if len(signatures) == 1:
                    complete = [sorted(complete, key=lambda value: value[0])[0]]
                else:
                    choices = ", ".join(key for key, _ in sorted(complete))
                    raise RuntimeError(
                        f"Ambiguous source runs for {architecture}:{expert}: {choices}. "
                        "Choose without consulting test metrics by passing "
                        f"--source-experiment {architecture}:{expert}=EXPERIMENT_KEY."
                    )

            experiment_key, rows = complete[0]
            for seed in sorted(rows):
                result_path, result = rows[seed]
                validate_source_metadata(
                    result_path,
                    result,
                    architecture,
                    expert,
                    seed,
                    safe_configs[architecture],
                )
                prediction_path = resolve_prediction_path(
                    str(result["prediction_path"]), result_path, source_roots
                )
                prediction_sha256 = core.sha256_file(prediction_path)
                if result.get("prediction_sha256") != prediction_sha256:
                    raise ValueError(f"Prediction hash mismatch for {prediction_path}.")
                metadata_path = cache_metadata_path_for_result(result_path)
                metadata = read_json(metadata_path)
                if metadata.get("experiment_key") != experiment_key:
                    raise ValueError(f"Cache metadata mismatch for {metadata_path}.")
                if result.get("feature_cache_sha256") != metadata.get("cache_sha256"):
                    raise ValueError(f"Feature-cache hash mismatch for {result_path}.")
                selected[architecture][expert][seed] = SelectedSource(
                    architecture=architecture,
                    expert=expert,
                    variant=EXPERT_VARIANTS[expert],
                    seed=seed,
                    experiment_key=experiment_key,
                    result_path=result_path,
                    result_sha256=core.sha256_file(result_path),
                    prediction_path=prediction_path,
                    prediction_sha256=prediction_sha256,
                    cache_metadata_path=metadata_path.resolve(),
                    cache_metadata_sha256=core.sha256_file(metadata_path),
                    cache_identity=cache_identity(metadata),
                    result=dict(result),
                )

    identities = {
        json.dumps(source.cache_identity, sort_keys=True)
        for architecture in selected.values()
        for expert in architecture.values()
        for source in expert.values()
    }
    if len(identities) != 1:
        raise ValueError(
            "Selected source runs do not share the same train/test data and "
            "feature-order identity."
        )
    return selected


def load_prediction(source: SelectedSource) -> LoadedSource:
    with np.load(source.prediction_path, allow_pickle=False) as artifact:
        required = {"labels", "probabilities", "raw_predictions", "final_predictions"}
        missing = required - set(artifact.files)
        if missing:
            raise KeyError(
                f"{source.prediction_path} is missing arrays: {sorted(missing)}"
            )
        labels = np.asarray(artifact["labels"], dtype=np.int64)
        probabilities = np.asarray(artifact["probabilities"], dtype=np.float32)
        raw_predictions = np.asarray(artifact["raw_predictions"], dtype=np.int64)
        final_predictions = np.asarray(artifact["final_predictions"], dtype=np.int64)

    row_count = len(labels)
    if labels.shape != (row_count,):
        raise ValueError(f"Invalid labels shape in {source.prediction_path}.")
    if probabilities.shape != (row_count, 5):
        raise ValueError(f"Invalid probability shape in {source.prediction_path}.")
    if raw_predictions.shape != (row_count,) or final_predictions.shape != (row_count,):
        raise ValueError(f"Invalid prediction shape in {source.prediction_path}.")
    if row_count == 0:
        raise ValueError(f"{source.prediction_path} contains no rows.")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{source.prediction_path} contains non-finite probabilities.")
    if np.any(probabilities < -1e-7) or np.any(probabilities > 1.0 + 1e-7):
        raise ValueError(f"{source.prediction_path} contains values outside [0,1].")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError(
            f"Probability rows in {source.prediction_path} do not sum to one."
        )
    expected_raw = np.argmax(probabilities, axis=1).astype(np.int64)
    if not np.array_equal(raw_predictions, expected_raw):
        raise ValueError(
            f"Raw predictions disagree with argmax in {source.prediction_path}."
        )
    if not np.array_equal(final_predictions, raw_predictions):
        raise ValueError(
            f"Source {source.prediction_path} is not an unscaled raw-argmax expert."
        )
    if np.any((labels < 0) | (labels > 4)):
        raise ValueError(f"{source.prediction_path} contains a label outside 0..4.")
    return LoadedSource(
        selected=source,
        labels=labels,
        probabilities=probabilities,
        raw_predictions=raw_predictions,
    )


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
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        for diagnostic in (
            "r2l_overrides",
            "u2r_overrides",
            "changed_predictions",
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
    per_seed: pd.DataFrame,
    architectures: Sequence[str],
    seeds: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[Dict[str, Any]] = []
    for architecture in architectures:
        group = per_seed[per_seed["architecture"] == architecture]
        average = group[group["method"] == "simple_average"].set_index("seed")
        safe = group[group["method"] == "safe_stack"].set_index("seed")
        for seed in seeds:
            row: Dict[str, Any] = {
                "architecture": architecture,
                "model": final_core.FROZEN_CONFIG[architecture]["label"],
                "seed": int(seed),
            }
            for metric in METRICS:
                row[f"{metric}_delta_safe_minus_average"] = float(
                    safe.loc[int(seed), metric] - average.loc[int(seed), metric]
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    summary_rows: list[Dict[str, Any]] = []
    for architecture, group in frame.groupby("architecture", sort=False):
        row = {
            "architecture": architecture,
            "model": str(group.iloc[0]["model"]),
            "runs": int(len(group)),
            "seeds": ",".join(str(seed) for seed in sorted(group["seed"])),
        }
        for metric in METRICS:
            column = f"{metric}_delta_safe_minus_average"
            values = pd.to_numeric(group[column], errors="raise")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Output directory and default location of frozen SAFE configs.",
    )
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
        help=(
            "Resolve an ambiguous source without using test metrics; EXPERT is "
            "general, focal, or batching. Repeat as needed."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default="final_simple_average_vs_safe_stack_kddtest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate frozen configurations and source metadata/hashes without "
            "loading prediction arrays, computing metrics, or writing outputs."
        ),
    )
    args = parser.parse_args(argv)
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique values.")
    if any(seed < 0 for seed in args.seeds):
        parser.error("Seeds cannot be negative.")
    if (
        not args.output_prefix.strip()
        or Path(args.output_prefix).name != args.output_prefix
    ):
        parser.error("--output-prefix must be a nonempty filename-safe value.")
    try:
        args.source_overrides = parse_source_overrides(args.source_experiment)
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_arguments(argv)
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = (args.results_dir or (repo_root / "results")).expanduser().resolve()
    source_roots = unique_paths(
        [
            *(path.expanduser() for path in args.source_results_dir),
            results_dir,
        ]
    )
    architectures = list(dict.fromkeys(str(value) for value in args.architectures))
    seeds = sorted(int(seed) for seed in args.seeds)

    config_paths: Dict[str, Path] = {}
    safe_configs: Dict[str, Dict[str, Any]] = {}
    for architecture in architectures:
        config_path, best = load_frozen_safe_config(results_dir, architecture)
        config_paths[architecture] = config_path
        safe_configs[architecture] = best

    sources = select_sources(
        source_roots,
        architectures,
        seeds,
        safe_configs,
        args.source_overrides,
    )
    print("Final simple-average versus frozen SAFE-Stack evaluation")
    print(f"Architectures: {architectures}")
    print(f"Seeds: {seeds}")
    print("Experts: General=baseline, Focal=focal_only, Batching=batch_only")
    print("Source probabilities: saved raw KDDTest+ probabilities")
    print("Configuration selection on KDDTest+: NO")
    for architecture in architectures:
        groups = {
            expert: sources[architecture][expert][seeds[0]].experiment_key
            for expert in EXPERTS
        }
        frozen = FROZEN_SAFE_CONFIGS[architecture]
        print(
            f"  {architecture}: sources={groups}; SAFE experiment="
            f"{frozen['experiment_key']}, candidate={frozen['candidate_id']}"
        )
    if args.dry_run:
        print(
            "Dry run complete; source metadata and hashes are valid. "
            "Prediction arrays were not loaded and no outputs were written."
        )
        return

    loaded: Dict[str, Dict[str, Dict[int, LoadedSource]]] = {
        architecture: {expert: {} for expert in EXPERTS}
        for architecture in architectures
    }
    reference_labels: np.ndarray | None = None
    for architecture in architectures:
        for expert in EXPERTS:
            for seed in seeds:
                artifact = load_prediction(sources[architecture][expert][seed])
                if reference_labels is None:
                    reference_labels = artifact.labels
                elif not np.array_equal(artifact.labels, reference_labels):
                    raise ValueError(
                        f"KDDTest+ label/order mismatch for {architecture}:{expert}:s{seed}."
                    )
                loaded[architecture][expert][seed] = artifact
    if reference_labels is None:
        raise RuntimeError("No final prediction artifacts were loaded.")

    script_path = Path(__file__).resolve()
    fusion_path = Path(fusion.__file__).resolve()
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
    frozen_identity = {
        architecture: {
            **FROZEN_SAFE_CONFIGS[architecture],
            "best_config_sha256": core.sha256_file(config_paths[architecture]),
        }
        for architecture in architectures
    }
    settings = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "final simple-average versus frozen SAFE-Stack evaluation",
        "architectures": architectures,
        "seeds": seeds,
        "expert_order": list(EXPERTS),
        "expert_variants": EXPERT_VARIANTS,
        "methods": list(METHODS),
        "simple_average_rule": "argmax(mean raw expert probabilities)",
        "safe_stack_rule": (
            "architecture-specific OOF-frozen base/weights/margins; stable top-two "
            "support; preserve base rare predictions; larger excess wins"
        ),
        "class_order": ["DoS", "Probe", "R2L", "U2R", "Normal"],
        "selection_data": "KDDTrain+ pooled OOF predictions only",
        "evaluation_data": "saved untouched KDDTest+ probabilities",
        "kddtest_used_for_selection": False,
        "models_retrained": False,
        "source_identity": source_identity,
        "frozen_safe_configs": frozen_identity,
        "script_sha256": core.sha256_file(script_path),
        "fusion_implementation_sha256": core.sha256_file(fusion_path),
    }
    experiment_key = stable_hash(settings)
    stem = f"{args.output_prefix}_{experiment_key}"
    prediction_dir = results_dir / f"{stem}_predictions"
    per_seed_path = results_dir / f"{stem}_per_seed.csv"
    summary_path = results_dir / f"{stem}_summary.csv"
    formatted_path = results_dir / f"{stem}_summary_formatted.csv"
    delta_path = results_dir / f"{stem}_paired_deltas.csv"
    delta_summary_path = results_dir / f"{stem}_paired_delta_summary.csv"
    source_path = results_dir / f"{stem}_source_artifacts.csv"
    protocol_path = results_dir / f"{stem}_protocol.json"
    latest_path = results_dir / f"{args.output_prefix}_latest.json"

    run_rows: list[Dict[str, Any]] = []
    source_rows: list[Dict[str, Any]] = []
    pending_predictions: list[tuple[Path, Dict[str, np.ndarray], Dict[str, Any]]] = []
    for architecture in architectures:
        best = safe_configs[architecture]
        base_expert = str(best["selected_base"])
        base_index = EXPERTS.index(base_expert)
        r2l_weights = np.asarray(
            [best["r2l_weights"][expert] for expert in EXPERTS], dtype=np.float64
        )
        u2r_weights = np.asarray(
            [best["u2r_weights"][expert] for expert in EXPERTS], dtype=np.float64
        )
        for seed in seeds:
            expert_probabilities = np.stack(
                [
                    loaded[architecture][expert][seed].probabilities
                    for expert in EXPERTS
                ],
                axis=1,
            )
            labels = loaded[architecture][EXPERTS[0]][seed].labels
            average_probabilities = expert_probabilities.mean(axis=1)
            average_predictions = np.argmax(average_probabilities, axis=1).astype(
                np.int64
            )
            safe = fusion.predict_one_fusion(
                expert_probabilities,
                base_index,
                r2l_weights,
                u2r_weights,
                float(best["r2l_margin"]),
                float(best["u2r_margin"]),
                int(best["minimum_support"]),
            )
            safe_predictions = np.asarray(safe["final_predictions"], dtype=np.int64)
            base_predictions = np.asarray(safe["base_predictions"], dtype=np.int64)
            base_is_rare = np.isin(base_predictions, fusion.RARE_CLASSES)
            if not np.array_equal(
                safe_predictions[base_is_rare], base_predictions[base_is_rare]
            ):
                raise RuntimeError(
                    f"SAFE-Stack changed a base rare prediction for {architecture}."
                )

            prediction_path = prediction_dir / f"{architecture}_s{seed}.npz"
            arrays = {
                "row_indices": np.arange(len(labels), dtype=np.int64),
                "labels": labels,
                "expert_probabilities": expert_probabilities.astype(np.float32),
                "simple_average_probabilities": average_probabilities.astype(
                    np.float32
                ),
                "simple_average_predictions": average_predictions,
                "safe_stack_predictions": safe_predictions,
                **{name: np.asarray(values) for name, values in safe.items()},
            }
            pending_predictions.append(
                (
                    prediction_path,
                    arrays,
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "average_predictions": average_predictions,
                        "safe_predictions": safe_predictions,
                        "safe": safe,
                    },
                )
            )

            common = {
                "architecture": architecture,
                "model": final_core.FROZEN_CONFIG[architecture]["label"],
                "seed": int(seed),
                "safe_config_experiment_key": best["experiment_key"],
                "safe_config_candidate_id": int(best["candidate_id"]),
                "safe_base_expert": base_expert,
                "r2l_weights_gfb": json.dumps(r2l_weights.tolist()),
                "u2r_weights_gfb": json.dumps(u2r_weights.tolist()),
                "r2l_margin": float(best["r2l_margin"]),
                "u2r_margin": float(best["u2r_margin"]),
                "minimum_support": int(best["minimum_support"]),
                "prediction_path": str(prediction_path),
            }
            run_rows.append(
                {
                    **common,
                    "method": "simple_average",
                    "method_label": METHOD_LABELS["simple_average"],
                    "r2l_overrides": 0,
                    "u2r_overrides": 0,
                    "changed_predictions": 0,
                    **core.calculate_metrics(labels, average_predictions),
                }
            )
            run_rows.append(
                {
                    **common,
                    "method": "safe_stack",
                    "method_label": METHOD_LABELS["safe_stack"],
                    "r2l_overrides": int(np.asarray(safe["r2l_override"]).sum()),
                    "u2r_overrides": int(np.asarray(safe["u2r_override"]).sum()),
                    "changed_predictions": int(
                        (safe_predictions != base_predictions).sum()
                    ),
                    **core.calculate_metrics(labels, safe_predictions),
                }
            )
            for expert in EXPERTS:
                source = sources[architecture][expert][seed]
                source_rows.append(
                    {
                        "architecture": architecture,
                        "expert": expert,
                        "variant": source.variant,
                        "seed": seed,
                        "experiment_key": source.experiment_key,
                        "result_path": str(source.result_path),
                        "result_sha256": source.result_sha256,
                        "prediction_path": str(source.prediction_path),
                        "prediction_sha256": source.prediction_sha256,
                        "cache_metadata_path": str(source.cache_metadata_path),
                        "cache_metadata_sha256": source.cache_metadata_sha256,
                    }
                )

    per_seed = pd.DataFrame(run_rows)
    expected_rows = len(architectures) * len(seeds) * len(METHODS)
    if len(per_seed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} per-seed rows, got {len(per_seed)}."
        )
    summary = aggregate_metrics(per_seed, seeds)
    expected_summary_rows = len(architectures) * len(METHODS)
    if len(summary) != expected_summary_rows:
        raise RuntimeError(
            f"Expected {expected_summary_rows} summary rows, got {len(summary)}."
        )
    formatted = formatted_summary(summary)
    deltas, delta_summary = paired_deltas(per_seed, architectures, seeds)

    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_hashes: Dict[str, str] = {}
    for path, arrays, _ in pending_predictions:
        core.atomic_npz(path, **arrays)
        prediction_hashes[str(path)] = core.sha256_file(path)

    protocol = {
        **settings,
        "experiment_key": experiment_key,
        "runtime_seconds": float(time.perf_counter() - started),
        "kddtest_accessed": True,
        "kddtest_access_mode": "saved final prediction artifacts only",
        "outputs": {
            "source_artifacts": str(source_path),
            "per_seed": str(per_seed_path),
            "summary": str(summary_path),
            "formatted_summary": str(formatted_path),
            "paired_deltas": str(delta_path),
            "paired_delta_summary": str(delta_summary_path),
            "prediction_directory": str(prediction_dir),
            "prediction_sha256": prediction_hashes,
        },
    }
    core.atomic_csv(source_path, pd.DataFrame(source_rows))
    core.atomic_csv(per_seed_path, per_seed)
    core.atomic_csv(summary_path, summary)
    core.atomic_csv(formatted_path, formatted)
    core.atomic_csv(delta_path, deltas)
    core.atomic_csv(delta_summary_path, delta_summary)
    core.atomic_json(protocol_path, protocol)
    latest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "protocol": str(protocol_path),
        "source_artifacts": str(source_path),
        "per_seed": str(per_seed_path),
        "summary": str(summary_path),
        "formatted_summary": str(formatted_path),
        "paired_deltas": str(delta_path),
        "paired_delta_summary": str(delta_summary_path),
        "prediction_directory": str(prediction_dir),
    }
    core.atomic_json(latest_path, latest)

    print("\n=== Final KDDTest+ simple average versus SAFE-Stack ===")
    print(formatted.to_string(index=False))
    print("\nSaved results:")
    print(f"  Summary: {summary_path}")
    print(f"  Per-seed metrics: {per_seed_path}")
    print(f"  Paired deltas: {delta_summary_path}")
    print(f"  Protocol: {protocol_path}")
    print(f"  Latest pointer: {latest_path}")


if __name__ == "__main__":
    main()
