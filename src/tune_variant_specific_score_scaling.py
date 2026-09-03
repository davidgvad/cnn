"""Select variant-specific score scaling from saved OOF predictions.

This is a post-processing utility: it never trains a neural network.  The
``select`` phase reads KDDTrain+ out-of-fold probabilities and independently
selects R2L/U2R score coefficients for each architecture and each of four
training regimes:

* ordinary cross-entropy;
* class-balanced focal loss;
* cross-entropy with minority-guaranteed batches;
* class-balanced focal loss with minority-guaranteed batches.

The ``evaluate`` phase reads the frozen selection manifest and applies each
pair to the corresponding saved KDDTest+ probabilities.  Keeping these phases
separate makes it explicit that KDDTest+ is not part of coefficient selection.

Run both commands from the repository root::

    python src/tune_variant_specific_score_scaling.py select
    python src/tune_variant_specific_score_scaling.py evaluate

The resulting validation and KDDTest+ CSV files contain the complete 2 x 2 x 2
factorial comparison of focal loss, minority batching, and score scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import run_no_ctgan_model_ablation_4gpu as core
import tune_conv2d_score_scaling_cv_4gpu as scaling


SCHEMA_VERSION = 1
ARCHITECTURES = ("conv2d", "conv1d", "transformer", "mlp")
ARCHITECTURE_LABELS = {
    "conv2d": "Conv2D",
    "conv1d": "Conv1D",
    "transformer": "Transformer",
    "mlp": "MLP",
}
SEEDS = (0, 1, 2)

# Use one pre-existing repository grid for every training regime.  This avoids
# giving any variant a wider search solely because of its observed results.
DEFAULT_COEFFICIENTS = tuple(scaling.BASELINE_SCALING_COEFFICIENTS)

BASE_TRAINING_ORDER = ("baseline", "focal_only", "batch_only", "focal_batch")
BASE_TRAINING = {
    "baseline": {
        "latest_names": ("{architecture}_baseline_cv_latest.json",),
        "oof_layout": "simple",
        "test_variant": "baseline",
        "focal": False,
        "batching": False,
    },
    "focal_only": {
        "latest_names": ("{architecture}_focal_stage1_latest.json",),
        "oof_layout": "focal_stage1",
        "test_variant": "focal_only",
        "focal": True,
        "batching": False,
    },
    "batch_only": {
        "latest_names": ("{architecture}_batch_baseline_cv_latest.json",),
        "oof_layout": "simple",
        "test_variant": "batch_only",
        "focal": False,
        "batching": True,
    },
    "focal_batch": {
        # Prefer the OOF probabilities used by the full score-scaling search.
        # The raw focal+batch runner is a valid fallback when that pointer is
        # unavailable.
        "latest_names": (
            "{architecture}_balanced_score_scaling_latest.json",
            "{architecture}_focal_batch_cv_latest.json",
        ),
        "oof_layout": "simple",
        "test_variant": "full",
        "focal": True,
        "batching": True,
    },
}
SHARED_TRAINING_MODE = {
    "baseline": "baseline_ce",
    "batch_only": "baseline_batch",
    "focal_batch": "focal_balanced",
}

RAW_CONFIGURATION = {
    "baseline": "baseline",
    "focal_only": "focal_only",
    "batch_only": "batch_only",
    "focal_batch": "focal_batch",
}
SCALED_CONFIGURATION = {
    "baseline": "scaling_only_tuned",
    "focal_only": "focal_scaling_tuned",
    "batch_only": "batch_scaling_tuned",
    "focal_batch": "full_retuned",
}
CONFIGURATION_LABELS = {
    "baseline": "Baseline",
    "focal_only": "Focal only",
    "batch_only": "Batching only",
    "scaling_only_tuned": "Scaling only (own tuning)",
    "focal_batch": "Focal + batching",
    "focal_scaling_tuned": "Focal + scaling (own tuning)",
    "batch_scaling_tuned": "Batching + scaling (own tuning)",
    "full_retuned": "Focal + batching + scaling (own tuning)",
}
CONFIGURATION_ORDER = (
    "baseline",
    "focal_only",
    "batch_only",
    "scaling_only_tuned",
    "focal_batch",
    "focal_scaling_tuned",
    "batch_scaling_tuned",
    "full_retuned",
)
METRICS = tuple(core.METRICS)


def stable_hash(value: Any, length: int = 12) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def resolve_recorded_path(
    raw_path: str | Path,
    repo_root: Path,
    recording_path: Path | None = None,
) -> Path:
    """Resolve an absolute or repository-relative path stored in JSON."""
    path = Path(raw_path).expanduser()
    candidates = [path] if path.is_absolute() else []
    candidates.append(repo_root / path)
    if recording_path is not None:
        candidates.append(recording_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Recorded path does not exist: {raw_path!s} "
        f"(recorded by {recording_path!s})"
    )


def find_latest_pointer(
    results_dir: Path,
    architecture: str,
    base_training: str,
) -> Path:
    settings = BASE_TRAINING[base_training]
    attempted = []
    for template in settings["latest_names"]:
        path = results_dir / template.format(architecture=architecture)
        attempted.append(path)
        if path.is_file():
            return path.resolve()
    attempted_text = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(
        f"No OOF latest pointer for {architecture}/{base_training}. Tried:\n"
        f"{attempted_text}"
    )


def load_oof_paths(
    repo_root: Path,
    results_dir: Path,
    architecture: str,
    base_training: str,
    seeds: Sequence[int],
) -> tuple[Dict[int, Path], Dict[str, Any]]:
    """Resolve one complete saved OOF probability matrix per seed."""
    pointer_path = find_latest_pointer(results_dir, architecture, base_training)
    pointer = core.read_json(pointer_path)
    required_pointer_fields = {"oof_directory", "protocol"}
    missing_pointer_fields = sorted(required_pointer_fields - set(pointer))
    if missing_pointer_fields:
        raise KeyError(
            f"Missing fields in {pointer_path}: {missing_pointer_fields}"
        )
    oof_dir = resolve_recorded_path(
        pointer["oof_directory"], repo_root, pointer_path
    )
    protocol_path = resolve_recorded_path(
        pointer["protocol"], repo_root, pointer_path
    )
    protocol = core.read_json(protocol_path)
    if protocol.get("kddtest_accessed") is not False:
        raise ValueError(
            f"OOF protocol does not certify KDDTest+ was excluded: {protocol_path}"
        )
    train_path = repo_root / "data" / "KDDTrain+.txt"
    recorded_train_hash = protocol.get("kddtrain_sha256")
    if recorded_train_hash is not None and train_path.is_file():
        if recorded_train_hash != core.sha256_file(train_path):
            raise ValueError(f"KDDTrain+ hash mismatch in {protocol_path}")
    layout = str(BASE_TRAINING[base_training]["oof_layout"])
    config_id: str | None = None
    best_path: Path | None = None
    if layout == "focal_stage1":
        if "best_config" not in pointer:
            raise KeyError(f"Missing best_config in {pointer_path}")
        best_path = resolve_recorded_path(
            pointer["best_config"], repo_root, pointer_path
        )
        best = core.read_json(best_path)
        config_id = str(best["config_id"])
        settings = protocol.get("settings", {})
        if settings.get("model") != ARCHITECTURE_LABELS[architecture]:
            raise ValueError(
                f"Focal protocol/model mismatch for {architecture}: {protocol_path}"
            )
        if settings.get("batching") != "ordinary_shuffled":
            raise ValueError(f"Focal-only protocol uses unexpected batching: {protocol_path}")
    else:
        settings = protocol.get("training_settings", {})
        expected_mode = SHARED_TRAINING_MODE[base_training]
        if settings.get("architecture") != architecture:
            raise ValueError(
                f"OOF protocol/architecture mismatch for {architecture}: "
                f"{protocol_path}"
            )
        if settings.get("training_mode") != expected_mode:
            raise ValueError(
                f"OOF protocol/training-mode mismatch for {architecture}/"
                f"{base_training}: expected {expected_mode}, got "
                f"{settings.get('training_mode')!r}."
            )

    paths: Dict[int, Path] = {}
    for seed in seeds:
        filename = (
            f"{config_id}_s{int(seed)}_oof_predictions.npz"
            if layout == "focal_stage1"
            else f"seed_{int(seed)}_oof_probabilities.npz"
        )
        path = oof_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing OOF artifact for {architecture}/{base_training}/"
                f"seed {seed}: {path}"
            )
        paths[int(seed)] = path.resolve()

    metadata = {
        "latest_pointer": str(pointer_path),
        "latest_pointer_sha256": core.sha256_file(pointer_path),
        "protocol": str(protocol_path),
        "protocol_sha256": core.sha256_file(protocol_path),
        "kddtrain_sha256": recorded_train_hash,
        "oof_directory": str(oof_dir),
        "best_config": str(best_path) if best_path is not None else None,
        "best_config_sha256": (
            core.sha256_file(best_path) if best_path is not None else None
        ),
        "config_id": config_id,
        "oof_files": {
            str(seed): {
                "path": str(path),
                "sha256": core.sha256_file(path),
            }
            for seed, path in paths.items()
        },
    }
    return paths, metadata


def read_probability_artifact(
    path: Path,
    *,
    require_oof_fields: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    fold_ids: np.ndarray | None = None
    with np.load(path, allow_pickle=False) as artifact:
        required = {"labels", "probabilities", "raw_predictions"}
        missing = sorted(required - set(artifact.files))
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        labels = np.asarray(artifact["labels"], dtype=np.int64)
        probabilities = np.asarray(artifact["probabilities"], dtype=np.float32)
        raw_predictions = np.asarray(artifact["raw_predictions"], dtype=np.int64)
        if require_oof_fields:
            extra_required = {"row_indices", "fold_ids"}
            missing_extra = sorted(extra_required - set(artifact.files))
            if missing_extra:
                raise KeyError(f"{path} is missing OOF arrays: {missing_extra}")
            row_indices = np.asarray(artifact["row_indices"], dtype=np.int64)
            fold_ids = np.asarray(artifact["fold_ids"], dtype=np.int64)
            if not np.array_equal(row_indices, np.arange(len(labels))):
                raise ValueError(f"OOF row indices are not complete/in order: {path}")
            if fold_ids.shape != labels.shape or set(np.unique(fold_ids)) != {
                0,
                1,
                2,
                3,
            }:
                raise ValueError(f"OOF fold IDs are invalid: {path}")

    if labels.ndim != 1:
        raise ValueError(f"Labels must be one-dimensional: {path}")
    if probabilities.shape != (len(labels), 5):
        raise ValueError(
            f"Expected probabilities with shape ({len(labels)}, 5), got "
            f"{probabilities.shape}: {path}"
        )
    if raw_predictions.shape != labels.shape:
        raise ValueError(f"Raw predictions do not align with labels: {path}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite probabilities: {path}")
    if np.any(probabilities < -1e-6) or np.any(probabilities > 1.0 + 1e-6):
        raise ValueError(f"Probability outside [0,1]: {path}")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-4, rtol=0.0):
        raise ValueError(f"Probability rows do not sum to one: {path}")
    expected_raw = np.argmax(probabilities, axis=1).astype(np.int64)
    if not np.array_equal(raw_predictions, expected_raw):
        raise ValueError(f"Saved raw predictions disagree with argmax: {path}")
    return labels, probabilities, raw_predictions, fold_ids


def add_metric_row(
    rows: list[Dict[str, Any]],
    *,
    architecture: str,
    base_training: str,
    configuration: str,
    scaling_enabled: bool,
    seed: int,
    r2l_coefficient: float,
    u2r_coefficient: float,
    labels: np.ndarray,
    predictions: np.ndarray,
    source_path: Path,
) -> None:
    settings = BASE_TRAINING[base_training]
    metrics = core.calculate_metrics(labels, predictions)
    rows.append(
        {
            "architecture": architecture,
            "model": ARCHITECTURE_LABELS[architecture],
            "base_training": base_training,
            "configuration": configuration,
            "configuration_label": CONFIGURATION_LABELS[configuration],
            "focal_loss": bool(settings["focal"]),
            "minority_batching": bool(settings["batching"]),
            "score_scaling": bool(scaling_enabled),
            "seed": int(seed),
            "r2l_score_coefficient": float(r2l_coefficient),
            "u2r_score_coefficient": float(u2r_coefficient),
            "source_probability_path": str(source_path),
            "source_probability_sha256": core.sha256_file(source_path),
            **metrics,
        }
    )


def summarize_seed_metrics(seed_frame: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        "architecture",
        "model",
        "base_training",
        "configuration",
        "configuration_label",
        "focal_loss",
        "minority_batching",
        "score_scaling",
        "r2l_score_coefficient",
        "u2r_score_coefficient",
    ]
    rows: list[Dict[str, Any]] = []
    for identity, group in seed_frame.groupby(identity_columns, sort=False):
        first = dict(zip(identity_columns, identity, strict=True))
        row: Dict[str, Any] = {
            **first,
            "runs": int(len(group)),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"])),
        }
        for metric in METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else 0.0
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    architecture_order = {name: index for index, name in enumerate(ARCHITECTURES)}
    configuration_order = {
        name: index for index, name in enumerate(CONFIGURATION_ORDER)
    }
    summary["_architecture_order"] = summary["architecture"].map(architecture_order)
    summary["_configuration_order"] = summary["configuration"].map(configuration_order)
    return summary.sort_values(
        ["_architecture_order", "_configuration_order"]
    ).drop(columns=["_architecture_order", "_configuration_order"]).reset_index(
        drop=True
    )


def rare_f1_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for configuration in CONFIGURATION_ORDER:
        group = summary[summary["configuration"] == configuration]
        if group.empty:
            continue
        by_architecture = group.set_index("architecture")
        if set(by_architecture.index) != set(ARCHITECTURES):
            continue
        means = np.asarray(
            [by_architecture.loc[name, "rare_f1_mean"] for name in ARCHITECTURES],
            dtype=float,
        )
        row: Dict[str, Any] = {
            "Method": CONFIGURATION_LABELS[configuration],
        }
        for architecture in ARCHITECTURES:
            observed = by_architecture.loc[architecture]
            row[ARCHITECTURE_LABELS[architecture]] = (
                f"{100.0 * float(observed['rare_f1_mean']):.2f} +/- "
                f"{100.0 * float(observed['rare_f1_std']):.2f}"
            )
        row["Architecture mean"] = float(100.0 * means.mean())
        row["Architecture SD"] = float(100.0 * means.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def validate_common_labels(
    expected: np.ndarray | None,
    observed: np.ndarray,
    source_path: Path,
) -> np.ndarray:
    if expected is None:
        return observed.copy()
    if not np.array_equal(expected, observed):
        raise ValueError(f"Labels disagree across probability artifacts: {source_path}")
    return expected


def select_coefficients(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir).expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    architectures = tuple(dict.fromkeys(args.architectures))
    seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    coefficients = tuple(sorted(float(value) for value in args.coefficient_values))

    if len(coefficients) != len(set(coefficients)) or not coefficients:
        raise SystemExit("Coefficient values must be nonempty and unique.")
    if any(not np.isfinite(value) or value <= 0.0 for value in coefficients):
        raise SystemExit("Every coefficient must be finite and positive.")
    if sum(np.isclose(value, 1.0) for value in coefficients) != 1:
        raise SystemExit("The coefficient grid must contain 1.0 exactly once.")

    all_rankings: list[pd.DataFrame] = []
    selected_rows: list[Dict[str, Any]] = []
    validation_rows: list[Dict[str, Any]] = []
    source_metadata: Dict[str, Any] = {}
    common_labels: np.ndarray | None = None
    common_fold_ids: np.ndarray | None = None

    print("Selecting independent score coefficients from KDDTrain+ OOF only")
    print(f"Architectures: {list(architectures)}")
    print(f"Training regimes: {list(BASE_TRAINING_ORDER)}")
    print(f"Seeds: {list(seeds)}")
    print(f"Coefficient grid: {list(coefficients)} ({len(coefficients) ** 2} pairs)")
    print("KDDTest+ accessed: NO", flush=True)

    for architecture in architectures:
        source_metadata[architecture] = {}
        for base_training in BASE_TRAINING_ORDER:
            oof_paths, metadata = load_oof_paths(
                repo_root,
                results_dir,
                architecture,
                base_training,
                seeds,
            )
            source_metadata[architecture][base_training] = metadata
            per_seed, ranking, best = scaling.score_oof_probabilities(
                oof_paths,
                seeds,
                coefficients,
                float(args.macro_f1_retention),
                float(args.minority_precision_retention),
                int(args.score_chunk_size),
            )
            ranking.insert(0, "base_training", base_training)
            ranking.insert(0, "model", ARCHITECTURE_LABELS[architecture])
            ranking.insert(0, "architecture", architecture)
            all_rankings.append(ranking)

            r2l_coefficient = float(best["r2l_score_coefficient"])
            u2r_coefficient = float(best["u2r_score_coefficient"])
            selected_rows.append(
                {
                    "architecture": architecture,
                    "model": ARCHITECTURE_LABELS[architecture],
                    "base_training": base_training,
                    "scaled_configuration": SCALED_CONFIGURATION[base_training],
                    "scaled_configuration_label": CONFIGURATION_LABELS[
                        SCALED_CONFIGURATION[base_training]
                    ],
                    "r2l_score_coefficient": r2l_coefficient,
                    "u2r_score_coefficient": u2r_coefficient,
                    "validation_rare_f1_mean": float(
                        best["metrics"]["rare_f1"]["mean"]
                    ),
                    "validation_rare_f1_std": float(
                        best["metrics"]["rare_f1"]["sample_std"]
                    ),
                    "validation_macro_f1_mean": float(
                        best["metrics"]["macro_f1"]["mean"]
                    ),
                    "validation_macro_f1_std": float(
                        best["metrics"]["macro_f1"]["sample_std"]
                    ),
                    "diagnostic_eligible_mean": bool(best["eligible_mean"]),
                    "diagnostic_eligible_all_seeds": bool(
                        best["eligible_all_seeds"]
                    ),
                    "oof_latest_pointer": metadata["latest_pointer"],
                }
            )

            selected_seed_scores = per_seed[
                np.isclose(per_seed["r2l_score_coefficient"], r2l_coefficient)
                & np.isclose(per_seed["u2r_score_coefficient"], u2r_coefficient)
            ].set_index("seed")
            for seed in seeds:
                path = oof_paths[int(seed)]
                labels, probabilities, raw_predictions, fold_ids = (
                    read_probability_artifact(path, require_oof_fields=True)
                )
                common_labels = validate_common_labels(common_labels, labels, path)
                if fold_ids is None:
                    raise RuntimeError(f"OOF artifact unexpectedly lacks fold IDs: {path}")
                if common_fold_ids is None:
                    common_fold_ids = fold_ids.copy()
                elif not np.array_equal(common_fold_ids, fold_ids):
                    raise ValueError(
                        f"Fold assignments disagree across OOF artifacts: {path}"
                    )
                scaled_predictions = core.apply_class_score_scaling(
                    probabilities,
                    {2: r2l_coefficient, 3: u2r_coefficient},
                )
                add_metric_row(
                    validation_rows,
                    architecture=architecture,
                    base_training=base_training,
                    configuration=RAW_CONFIGURATION[base_training],
                    scaling_enabled=False,
                    seed=int(seed),
                    r2l_coefficient=1.0,
                    u2r_coefficient=1.0,
                    labels=labels,
                    predictions=raw_predictions,
                    source_path=path,
                )
                add_metric_row(
                    validation_rows,
                    architecture=architecture,
                    base_training=base_training,
                    configuration=SCALED_CONFIGURATION[base_training],
                    scaling_enabled=True,
                    seed=int(seed),
                    r2l_coefficient=r2l_coefficient,
                    u2r_coefficient=u2r_coefficient,
                    labels=labels,
                    predictions=scaled_predictions,
                    source_path=path,
                )
                direct_metrics = core.calculate_metrics(labels, scaled_predictions)
                ranked_seed = selected_seed_scores.loc[int(seed)]
                for metric in METRICS:
                    if not np.isclose(
                        direct_metrics[metric],
                        float(ranked_seed[metric]),
                        atol=1e-12,
                        rtol=1e-12,
                    ):
                        raise RuntimeError(
                            f"Direct/ranked validation mismatch for {architecture}/"
                            f"{base_training}/seed {seed}/{metric}."
                        )
            print(
                f"  {ARCHITECTURE_LABELS[architecture]:11s} {base_training:12s} "
                f"-> R2L={r2l_coefficient:g}, U2R={u2r_coefficient:g}, "
                f"Rare F1={100.0 * best['metrics']['rare_f1']['mean']:.2f}%",
                flush=True,
            )

    ranking_frame = pd.concat(all_rankings, ignore_index=True)
    selected_frame = pd.DataFrame(selected_rows)
    seed_frame = pd.DataFrame(validation_rows)
    summary_frame = summarize_seed_metrics(seed_frame)
    table_frame = rare_f1_table(summary_frame)

    source_identity = {
        architecture: {
            base_training: {
                seed: values["sha256"]
                for seed, values in metadata["oof_files"].items()
            }
            for base_training, metadata in by_training.items()
        }
        for architecture, by_training in source_metadata.items()
    }
    identity = {
        "schema_version": SCHEMA_VERSION,
        "architectures": list(architectures),
        "base_training_regimes": list(BASE_TRAINING_ORDER),
        "seeds": list(seeds),
        "coefficient_values": list(coefficients),
        "ranking_rule": (
            "maximize mean OOF Rare Macro-F1; then mean Macro-F1; then "
            "multiplicative closeness to (1,1); then lower Rare Macro-F1 "
            "sample SD; then lower coefficient values"
        ),
        "source_oof_sha256": source_identity,
        "script_sha256": core.sha256_file(Path(__file__).resolve()),
    }
    selection_id = stable_hash(identity)
    prefix = f"variant_specific_scaling_selection_{selection_id}"
    ranking_path = results_dir / f"{prefix}_rankings.csv"
    coefficients_path = results_dir / f"{prefix}_selected_coefficients.csv"
    seed_path = results_dir / f"{prefix}_validation_seed_metrics.csv"
    summary_path = results_dir / f"{prefix}_validation_summary.csv"
    table_path = results_dir / f"{prefix}_validation_rare_f1_table.csv"
    manifest_path = results_dir / f"{prefix}.json"
    latest_path = results_dir / "variant_specific_scaling_selection_latest.json"

    core.atomic_csv(ranking_path, ranking_frame)
    core.atomic_csv(coefficients_path, selected_frame)
    core.atomic_csv(seed_path, seed_frame)
    core.atomic_csv(summary_path, summary_frame)
    core.atomic_csv(table_path, table_frame)
    manifest = {
        **identity,
        "selection_id": selection_id,
        "purpose": "variant-specific score-scaling selection",
        "selection_partition": "KDDTrain+ pooled four-fold OOF predictions",
        "kddtest_accessed": False,
        "retention_indicators": "diagnostic only; no coefficient pair excluded",
        "macro_f1_retention": float(args.macro_f1_retention),
        "minority_precision_retention": float(args.minority_precision_retention),
        "selected_coefficients": selected_frame.to_dict(orient="records"),
        "source_metadata": source_metadata,
        "outputs": {
            "rankings": str(ranking_path),
            "selected_coefficients": str(coefficients_path),
            "validation_seed_metrics": str(seed_path),
            "validation_summary": str(summary_path),
            "validation_rare_f1_table": str(table_path),
        },
    }
    core.atomic_json(manifest_path, manifest)
    core.atomic_json(
        latest_path,
        {
            "selection_id": selection_id,
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": core.sha256_file(manifest_path),
        },
    )

    print("\nOOF validation Rare Macro-F1 (%)")
    print(table_frame.to_string(index=False))
    print("\nFrozen selection saved:")
    print(f"  {manifest_path}")
    print("No KDDTest+ artifact was read. Run the evaluate phase next.")


def load_selection_manifest(
    selection_arg: str | None,
    repo_root: Path,
    results_dir: Path,
) -> tuple[Path, Dict[str, Any]]:
    supplied = (
        Path(selection_arg).expanduser()
        if selection_arg
        else results_dir / "variant_specific_scaling_selection_latest.json"
    )
    supplied = supplied.resolve()
    if not supplied.is_file():
        raise FileNotFoundError(f"Selection file not found: {supplied}")
    data = core.read_json(supplied)
    if "selection_manifest" in data:
        manifest_path = resolve_recorded_path(
            data["selection_manifest"], repo_root, supplied
        )
        expected_hash = data.get("selection_manifest_sha256")
        if expected_hash and core.sha256_file(manifest_path) != expected_hash:
            raise ValueError(f"Selection-manifest hash mismatch: {manifest_path}")
        data = core.read_json(manifest_path)
    else:
        manifest_path = supplied
    if data.get("kddtest_accessed") is not False:
        raise ValueError(
            "Selection manifest does not certify validation-only coefficient selection."
        )
    return manifest_path.resolve(), data


def candidate_test_paths(
    results_dir: Path,
    architecture: str,
    test_variant: str,
    seed: int,
) -> list[Path]:
    suffix = f"_{architecture}_{test_variant}_s{int(seed)}.npz"
    return sorted(
        path.resolve()
        for path in results_dir.rglob(f"*{suffix}")
        if "kddtest" in path.name.lower()
        and "predictions" in path.parent.name.lower()
    )


def find_test_prediction_path(
    results_dir: Path,
    architecture: str,
    test_variant: str,
    seed: int,
) -> Path:
    candidates = candidate_test_paths(
        results_dir, architecture, test_variant, seed
    )
    if not candidates:
        raise FileNotFoundError(
            f"No saved KDDTest+ probability artifact for {architecture}/"
            f"{test_variant}/seed {seed} below {results_dir}."
        )
    if len(candidates) == 1:
        return candidates[0]
    by_hash: Dict[str, list[Path]] = {}
    for candidate in candidates:
        by_hash.setdefault(core.sha256_file(candidate), []).append(candidate)
    if len(by_hash) == 1:
        return candidates[0]
    choices = "\n".join(f"  - {path}" for path in candidates)
    raise RuntimeError(
        f"Multiple distinct KDDTest+ artifacts match {architecture}/"
        f"{test_variant}/seed {seed}; refusing to choose based on test output:\n"
        f"{choices}"
    )


def load_test_run_metadata(
    prediction_path: Path,
    repo_root: Path,
    architecture: str,
    test_variant: str,
    seed: int,
) -> tuple[Path, Dict[str, Any], Path, Dict[str, Any]]:
    prediction_dir = prediction_path.parent
    suffix = "_predictions"
    if not prediction_dir.name.endswith(suffix):
        raise ValueError(f"Unexpected final-prediction directory: {prediction_dir}")
    run_dir = prediction_dir.with_name(
        prediction_dir.name[: -len(suffix)] + "_runs"
    )
    run_path = run_dir / f"{prediction_path.stem}.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"Missing run metadata for {prediction_path}: {run_path}")
    run = core.read_json(run_path)
    expected = {
        "architecture": architecture,
        "variant": test_variant,
        "seed": int(seed),
    }
    mismatches = {
        key: (run.get(key), value)
        for key, value in expected.items()
        if run.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Final-run metadata mismatch in {run_path}: {mismatches}")
    if run.get("kddtest_used_for_selection") is not False:
        raise ValueError(
            f"Final-run metadata does not certify no test selection: {run_path}"
        )
    if run.get("prediction_sha256") != core.sha256_file(prediction_path):
        raise ValueError(f"Final prediction hash mismatch: {prediction_path}")
    experiment_prefix = prediction_dir.name[: -len(suffix)]
    cache_metadata_path = prediction_dir.parent / f"{experiment_prefix}_feature_cache.json"
    if not cache_metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing feature-cache metadata for {prediction_path}: "
            f"{cache_metadata_path}"
        )
    cache_metadata = core.read_json(cache_metadata_path)
    if cache_metadata.get("cache_sha256") != run.get("feature_cache_sha256"):
        raise ValueError(
            f"Run/feature-cache hash mismatch for {prediction_path}"
        )
    train_path = repo_root / "data" / "KDDTrain+.txt"
    test_path = repo_root / "data" / "KDDTest+.txt"
    expected_source_hashes = {
        "train_sha256": core.sha256_file(train_path),
        "test_sha256": core.sha256_file(test_path),
    }
    mismatched_sources = {
        key: (cache_metadata.get(key), value)
        for key, value in expected_source_hashes.items()
        if cache_metadata.get(key) != value
    }
    if mismatched_sources:
        raise ValueError(
            f"Final feature-cache source mismatch in {cache_metadata_path}: "
            f"{mismatched_sources}"
        )
    return run_path.resolve(), run, cache_metadata_path.resolve(), cache_metadata


def verify_oof_lineage(manifest: Mapping[str, Any]) -> None:
    for architecture, by_training in manifest["source_metadata"].items():
        for base_training, metadata in by_training.items():
            for seed, source in metadata["oof_files"].items():
                path = Path(source["path"])
                if not path.is_file():
                    raise FileNotFoundError(
                        f"Selected OOF source disappeared: {architecture}/"
                        f"{base_training}/seed {seed}: {path}"
                    )
                if core.sha256_file(path) != source["sha256"]:
                    raise ValueError(f"Selected OOF source changed: {path}")


def evaluate_kddtest(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = Path(args.results_dir).expanduser().resolve()
    manifest_path, manifest = load_selection_manifest(
        args.selection, repo_root, results_dir
    )
    verify_oof_lineage(manifest)
    architectures = tuple(manifest["architectures"])
    seeds = tuple(int(seed) for seed in manifest["seeds"])
    selected_lookup = {
        (row["architecture"], row["base_training"]): row
        for row in manifest["selected_coefficients"]
    }

    print("Applying frozen, validation-selected coefficients to KDDTest+")
    print(f"Selection: {manifest_path}")
    print("No coefficient will be changed or selected from KDDTest+.", flush=True)

    test_rows: list[Dict[str, Any]] = []
    source_metadata: Dict[str, Any] = {}
    common_labels: np.ndarray | None = None
    for architecture in architectures:
        source_metadata[architecture] = {}
        for base_training in BASE_TRAINING_ORDER:
            selected = selected_lookup[(architecture, base_training)]
            r2l_coefficient = float(selected["r2l_score_coefficient"])
            u2r_coefficient = float(selected["u2r_score_coefficient"])
            test_variant = str(BASE_TRAINING[base_training]["test_variant"])
            source_metadata[architecture][base_training] = {}
            for seed in seeds:
                path = find_test_prediction_path(
                    results_dir, architecture, test_variant, seed
                )
                run_path, run, cache_metadata_path, cache_metadata = (
                    load_test_run_metadata(
                        path,
                        repo_root,
                        architecture,
                        test_variant,
                        int(seed),
                    )
                )
                labels, probabilities, raw_predictions, _ = (
                    read_probability_artifact(path, require_oof_fields=False)
                )
                common_labels = validate_common_labels(common_labels, labels, path)
                scaled_predictions = core.apply_class_score_scaling(
                    probabilities,
                    {2: r2l_coefficient, 3: u2r_coefficient},
                )
                source_metadata[architecture][base_training][str(seed)] = {
                    "path": str(path),
                    "sha256": core.sha256_file(path),
                    "stored_test_variant": test_variant,
                    "run_metadata_path": str(run_path),
                    "run_metadata_sha256": core.sha256_file(run_path),
                    "feature_cache_sha256": run.get("feature_cache_sha256"),
                    "feature_cache_metadata_path": str(cache_metadata_path),
                    "feature_cache_metadata_sha256": core.sha256_file(
                        cache_metadata_path
                    ),
                    "kddtrain_sha256": cache_metadata["train_sha256"],
                    "kddtest_sha256": cache_metadata["test_sha256"],
                }
                add_metric_row(
                    test_rows,
                    architecture=architecture,
                    base_training=base_training,
                    configuration=RAW_CONFIGURATION[base_training],
                    scaling_enabled=False,
                    seed=int(seed),
                    r2l_coefficient=1.0,
                    u2r_coefficient=1.0,
                    labels=labels,
                    predictions=raw_predictions,
                    source_path=path,
                )
                add_metric_row(
                    test_rows,
                    architecture=architecture,
                    base_training=base_training,
                    configuration=SCALED_CONFIGURATION[base_training],
                    scaling_enabled=True,
                    seed=int(seed),
                    r2l_coefficient=r2l_coefficient,
                    u2r_coefficient=u2r_coefficient,
                    labels=labels,
                    predictions=scaled_predictions,
                    source_path=path,
                )

    seed_frame = pd.DataFrame(test_rows)
    summary_frame = summarize_seed_metrics(seed_frame)
    table_frame = rare_f1_table(summary_frame)
    selection_sha256 = core.sha256_file(manifest_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "selection_id": manifest["selection_id"],
        "selection_manifest_sha256": selection_sha256,
        "test_source_sha256": {
            architecture: {
                base_training: {
                    seed: details["sha256"]
                    for seed, details in by_seed.items()
                }
                for base_training, by_seed in by_training.items()
            }
            for architecture, by_training in source_metadata.items()
        },
        "script_sha256": core.sha256_file(Path(__file__).resolve()),
    }
    evaluation_id = stable_hash(identity)
    prefix = f"variant_specific_scaling_kddtest_{evaluation_id}"
    seed_path = results_dir / f"{prefix}_seed_metrics.csv"
    summary_path = results_dir / f"{prefix}_summary.csv"
    table_path = results_dir / f"{prefix}_rare_f1_table.csv"
    protocol_path = results_dir / f"{prefix}_protocol.json"
    latest_path = results_dir / "variant_specific_scaling_kddtest_latest.json"
    core.atomic_csv(seed_path, seed_frame)
    core.atomic_csv(summary_path, summary_frame)
    core.atomic_csv(table_path, table_frame)
    protocol = {
        **identity,
        "evaluation_id": evaluation_id,
        "purpose": "frozen variant-specific score-scaling evaluation",
        "selection_manifest": str(manifest_path),
        "selection_partition": manifest["selection_partition"],
        "evaluation_partition": "KDDTest+",
        "coefficient_selection_on_kddtest": False,
        "network_retraining": False,
        "test_source_metadata": source_metadata,
        "outputs": {
            "seed_metrics": str(seed_path),
            "summary": str(summary_path),
            "rare_f1_table": str(table_path),
        },
    }
    core.atomic_json(protocol_path, protocol)
    core.atomic_json(
        latest_path,
        {
            "evaluation_id": evaluation_id,
            "protocol": str(protocol_path),
            "protocol_sha256": core.sha256_file(protocol_path),
            **protocol["outputs"],
        },
    )

    print("\nKDDTest+ Rare Macro-F1 (%)")
    print(table_frame.to_string(index=False))
    print("\nSaved:")
    print(f"  {seed_path}")
    print(f"  {summary_path}")
    print(f"  {table_path}")
    print(f"  {protocol_path}")


def coefficient_values(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Independently select score scaling for each training variant from "
            "saved OOF probabilities, then evaluate frozen choices on saved "
            "KDDTest+ probabilities."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
    )
    subparsers = parser.add_subparsers(dest="phase", required=True)

    select = subparsers.add_parser(
        "select", help="select coefficients using KDDTrain+ OOF only"
    )
    select.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=list(ARCHITECTURES),
    )
    select.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    select.add_argument(
        "--coefficient-values",
        nargs="+",
        type=float,
        default=coefficient_values(DEFAULT_COEFFICIENTS),
    )
    select.add_argument(
        "--macro-f1-retention",
        type=float,
        default=scaling.DEFAULT_MACRO_F1_RETENTION,
        help="diagnostic only; does not exclude candidates",
    )
    select.add_argument(
        "--minority-precision-retention",
        type=float,
        default=scaling.DEFAULT_MINORITY_PRECISION_RETENTION,
        help="diagnostic only; does not exclude candidates",
    )
    select.add_argument("--score-chunk-size", type=int, default=512)
    select.set_defaults(handler=select_coefficients)

    evaluate = subparsers.add_parser(
        "evaluate", help="apply a frozen selection to KDDTest+ probabilities"
    )
    evaluate.add_argument(
        "--selection",
        default=None,
        help=(
            "selection manifest or latest pointer; defaults to "
            "results/variant_specific_scaling_selection_latest.json"
        ),
    )
    evaluate.set_defaults(handler=evaluate_kddtest)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.phase == "select":
        if not 0.0 < args.macro_f1_retention <= 1.0:
            parser.error("--macro-f1-retention must be in (0,1].")
        if not 0.0 < args.minority_precision_retention <= 1.0:
            parser.error("--minority-precision-retention must be in (0,1].")
        if args.score_chunk_size <= 0:
            parser.error("--score-chunk-size must be positive.")
        if len(args.seeds) != len(set(args.seeds)) or any(
            seed < 0 for seed in args.seeds
        ):
            parser.error("--seeds must be unique nonnegative integers.")
    args.handler(args)


if __name__ == "__main__":
    main()
