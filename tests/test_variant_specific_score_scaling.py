from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import run_no_ctgan_model_ablation_4gpu as core  # noqa: E402
import tune_variant_specific_score_scaling as subject  # noqa: E402


class VariantSpecificScoreScalingTests(unittest.TestCase):
    def test_relocated_pointer_prefers_current_artifact_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original" / "protocol.json"
            relocated = root / "relocated" / "protocol.json"
            original.parent.mkdir(); relocated.parent.mkdir()
            original.write_text('{"copy":"original"}')
            relocated.write_text('{"copy":"relocated"}')
            result = subject.resolve_recorded_path(original, root, relocated.parent / "latest.json")
            self.assertEqual(result, relocated.resolve())

    @staticmethod
    def write_oof(path: Path, labels: np.ndarray, probabilities: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        core.atomic_npz(
            path,
            row_indices=np.arange(len(labels), dtype=np.int64),
            fold_ids=np.arange(len(labels), dtype=np.int64) % 4,
            labels=labels,
            probabilities=probabilities.astype(np.float32),
            raw_predictions=np.argmax(probabilities, axis=1).astype(np.int64),
        )

    def test_focal_stage1_resolves_only_selected_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            results_dir = repo_root / "results"
            oof_dir = results_dir / "conv2d_focal_stage1_example_oof"
            labels = np.tile(np.arange(5, dtype=np.int64), 4)
            probabilities = np.full((len(labels), 5), 0.025, dtype=np.float64)
            probabilities[np.arange(len(labels)), labels] = 0.90
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for seed in subject.SEEDS:
                self.write_oof(
                    oof_dir / f"b0p99_g0p5_s{seed}_oof_predictions.npz",
                    labels,
                    probabilities,
                )
            best_path = results_dir / "conv2d_focal_stage1_best.json"
            protocol_path = results_dir / "conv2d_focal_stage1_protocol.json"
            pointer_path = results_dir / "conv2d_focal_stage1_latest.json"
            results_dir.mkdir(parents=True, exist_ok=True)
            best_path.write_text(
                json.dumps({"config_id": "b0p99_g0p5"}), encoding="utf-8"
            )
            protocol_path.write_text(
                json.dumps(
                    {
                        "kddtest_accessed": False,
                        "settings": {
                            "model": "Conv2D",
                            "batching": "ordinary_shuffled",
                        },
                    }
                ),
                encoding="utf-8",
            )
            pointer_path.write_text(
                json.dumps(
                    {
                        "best_config": str(best_path),
                        "oof_directory": str(oof_dir),
                        "protocol": str(protocol_path),
                    }
                ),
                encoding="utf-8",
            )
            paths, metadata = subject.load_oof_paths(
                repo_root,
                results_dir,
                "conv2d",
                "focal_only",
                subject.SEEDS,
            )
            self.assertEqual(set(paths), set(subject.SEEDS))
            self.assertEqual(metadata["config_id"], "b0p99_g0p5")
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_legacy_focal_batch_protocol_is_identified_from_recorded_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            results_dir = repo_root / "results"
            oof_dir = results_dir / "conv2d_balanced_score_scaling_example_oof"
            labels = np.tile(np.arange(5, dtype=np.int64), 4)
            probabilities = np.full((len(labels), 5), 0.025, dtype=np.float64)
            probabilities[np.arange(len(labels)), labels] = 0.90
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            for seed in subject.SEEDS:
                self.write_oof(
                    oof_dir / f"seed_{seed}_oof_probabilities.npz",
                    labels,
                    probabilities,
                )

            protocol_path = results_dir / "legacy_focal_batch_protocol.json"
            pointer_path = results_dir / "conv2d_balanced_score_scaling_latest.json"
            core.atomic_json(
                protocol_path,
                {
                    "kddtest_accessed": False,
                    "training_settings": {
                        "model": "Conv2D",
                        "cb_beta": 0.99,
                        "focal_gamma": 0.5,
                        "batching": "minority_guaranteed_with_replacement",
                        "minority_per_batch": 1,
                    },
                },
            )
            core.atomic_json(
                pointer_path,
                {
                    "protocol": str(protocol_path),
                    "oof_directory": str(oof_dir),
                },
            )

            paths, metadata = subject.load_oof_paths(
                repo_root,
                results_dir,
                "conv2d",
                "focal_batch",
                subject.SEEDS,
            )
            self.assertEqual(set(paths), set(subject.SEEDS))
            self.assertEqual(
                metadata["training_mode_evidence"],
                "legacy_focal_batch_fields",
            )

            protocol = core.read_json(protocol_path)
            del protocol["training_settings"]["focal_gamma"]
            core.atomic_json(protocol_path, protocol)
            with self.assertRaisesRegex(ValueError, "focal/batching fields"):
                subject.load_oof_paths(
                    repo_root,
                    results_dir,
                    "conv2d",
                    "focal_batch",
                    subject.SEEDS,
                )

    def test_probability_artifact_validation_and_score_application(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_0_oof_probabilities.npz"
            labels = np.asarray([0, 1, 2, 3, 4, 2, 3, 4], dtype=np.int64)
            probabilities = np.asarray(
                [
                    [0.80, 0.05, 0.05, 0.05, 0.05],
                    [0.05, 0.80, 0.05, 0.05, 0.05],
                    [0.05, 0.05, 0.40, 0.05, 0.45],
                    [0.05, 0.05, 0.05, 0.40, 0.45],
                    [0.05, 0.05, 0.05, 0.05, 0.80],
                    [0.05, 0.05, 0.40, 0.05, 0.45],
                    [0.05, 0.05, 0.05, 0.40, 0.45],
                    [0.05, 0.05, 0.05, 0.05, 0.80],
                ],
                dtype=np.float64,
            )
            self.write_oof(path, labels, probabilities)
            observed_labels, observed_probabilities, raw, fold_ids = (
                subject.read_probability_artifact(path, require_oof_fields=True)
            )
            np.testing.assert_array_equal(observed_labels, labels)
            np.testing.assert_array_equal(raw, np.argmax(probabilities, axis=1))
            self.assertEqual(set(np.unique(fold_ids)), {0, 1, 2, 3})
            promoted = core.apply_class_score_scaling(
                observed_probabilities, {2: 0.5, 3: 0.5}
            )
            self.assertGreater(
                core.calculate_metrics(labels, promoted)["rare_f1"],
                core.calculate_metrics(labels, raw)["rare_f1"],
            )

    def test_distinct_duplicate_test_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            labels = np.arange(5, dtype=np.int64)
            first = np.eye(5, dtype=np.float32)
            second = np.roll(first, 1, axis=1)
            for index, probabilities in enumerate((first, second)):
                path = (
                    results_dir
                    / f"final_example_kddtest_{index}_predictions"
                    / "final_example_kddtest_conv2d_baseline_s0.npz"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                core.atomic_npz(
                    path,
                    labels=labels,
                    probabilities=probabilities,
                    raw_predictions=np.argmax(probabilities, axis=1),
                    final_predictions=np.argmax(probabilities, axis=1),
                )
            with self.assertRaisesRegex(RuntimeError, "refusing to choose"):
                subject.find_test_prediction_path(
                    results_dir, "conv2d", "baseline", 0
                )

    def test_summary_keeps_all_eight_factorial_configurations(self) -> None:
        rows = []
        for architecture in subject.ARCHITECTURES:
            for base_training in subject.BASE_TRAINING_ORDER:
                for scaling_enabled, configuration in (
                    (False, subject.RAW_CONFIGURATION[base_training]),
                    (True, subject.SCALED_CONFIGURATION[base_training]),
                ):
                    for seed in subject.SEEDS:
                        settings = subject.BASE_TRAINING[base_training]
                        rows.append(
                            {
                                "architecture": architecture,
                                "model": subject.ARCHITECTURE_LABELS[architecture],
                                "base_training": base_training,
                                "configuration": configuration,
                                "configuration_label": subject.CONFIGURATION_LABELS[
                                    configuration
                                ],
                                "focal_loss": settings["focal"],
                                "minority_batching": settings["batching"],
                                "score_scaling": scaling_enabled,
                                "seed": seed,
                                "r2l_score_coefficient": 0.5 if scaling_enabled else 1.0,
                                "u2r_score_coefficient": 2.0 if scaling_enabled else 1.0,
                                **{metric: 0.5 for metric in subject.METRICS},
                            }
                        )
        summary = subject.summarize_seed_metrics(pd.DataFrame(rows))
        table = subject.rare_f1_table(summary)
        self.assertEqual(len(summary), 32)
        self.assertEqual(len(table), 8)
        self.assertEqual(
            table["Method"].tolist(),
            [
                subject.CONFIGURATION_LABELS[name]
                for name in subject.CONFIGURATION_ORDER
            ],
        )

    def test_two_stage_end_to_end_from_saved_probabilities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory) / "results"
            results_dir.mkdir(parents=True)
            labels = np.tile(np.arange(5, dtype=np.int64), 4)
            probabilities = np.full((len(labels), 5), 0.04, dtype=np.float64)
            probabilities[np.arange(len(labels)), labels] = 0.84
            train_hash = core.sha256_file(REPO_ROOT / "data" / "KDDTrain+.txt")

            for architecture in subject.ARCHITECTURES:
                for base_training in subject.BASE_TRAINING_ORDER:
                    settings = subject.BASE_TRAINING[base_training]
                    pointer_name = settings["latest_names"][0].format(
                        architecture=architecture
                    )
                    pointer_path = results_dir / pointer_name
                    protocol_path = results_dir / (
                        f"{architecture}_{base_training}_protocol.json"
                    )
                    oof_dir = results_dir / f"{architecture}_{base_training}_oof"
                    pointer = {
                        "protocol": str(protocol_path),
                        "oof_directory": str(oof_dir),
                    }
                    training = {
                        "model": subject.ARCHITECTURE_LABELS[architecture],
                        "backbone": subject.scaling.ARCHITECTURE_DEFAULTS[architecture]["stage1"].FIXED_BACKBONE,
                        "training_seeds": list(subject.SEEDS), "fold_count": 4,
                        "fold_seed": 0, "epochs": 25, "batch_size": 256,
                        "deterministic_ops": False, "validation_used_during_training": False,
                        "ctgan": False,
                        "batching": "minority_guaranteed_with_replacement" if settings["batching"] else "ordinary_shuffled",
                    }
                    if base_training == "focal_only":
                        protocol = {
                            "kddtest_accessed": False,
                            "kddtrain_sha256": train_hash,
                            "settings": {
                                **training, "betas": [0.99], "focal_gammas": [0.5],
                            },
                        }
                        best_path = results_dir / (
                            f"{architecture}_{base_training}_best.json"
                        )
                        core.atomic_json(best_path, {
                            "config_id": "selected", "beta": 0.99, "focal_gamma": 0.5,
                            "training_seeds": list(subject.SEEDS), "fold_count": 4,
                            "kddtest_accessed": False,
                        })
                        pointer["best_config"] = str(best_path)
                    else:
                        protocol = {
                            "kddtest_accessed": False,
                            "kddtrain_sha256": train_hash,
                            "training_settings": {
                                **training,
                                "architecture": architecture,
                                "training_mode": subject.SHARED_TRAINING_MODE[
                                    base_training
                                ],
                                "cb_beta": 0.99 if settings["focal"] else None,
                                "focal_gamma": 0.5 if settings["focal"] else None,
                                "minority_per_batch": 1 if settings["batching"] else 0,
                            },
                        }
                    core.atomic_json(protocol_path, protocol)
                    core.atomic_json(pointer_path, pointer)
                    for seed in subject.SEEDS:
                        filename = (
                            f"selected_s{seed}_oof_predictions.npz"
                            if base_training == "focal_only"
                            else f"seed_{seed}_oof_probabilities.npz"
                        )
                        self.write_oof(oof_dir / filename, labels, probabilities)

            select_args = argparse.Namespace(
                results_dir=str(results_dir),
                architectures=list(subject.ARCHITECTURES),
                seeds=list(subject.SEEDS),
                coefficient_values=[0.5, 1.0, 2.0],
                macro_f1_retention=0.90,
                minority_precision_retention=0.80,
                score_chunk_size=8,
            )
            subject.select_coefficients(select_args)
            selection_latest = (
                results_dir / "variant_specific_scaling_selection_latest.json"
            )
            self.assertTrue(selection_latest.is_file())

            for architecture in subject.ARCHITECTURES:
                for base_training in subject.BASE_TRAINING_ORDER:
                    test_variant = subject.BASE_TRAINING[base_training][
                        "test_variant"
                    ]
                    prefix = f"final_synthetic_kddtest_{architecture}_{test_variant}"
                    prediction_dir = results_dir / f"{prefix}_predictions"
                    run_dir = results_dir / f"{prefix}_runs"
                    cache_metadata_path = results_dir / f"{prefix}_feature_cache.json"
                    cache_hash = f"synthetic-{architecture}-{test_variant}"
                    core.atomic_json(
                        cache_metadata_path,
                        {
                            "cache_sha256": cache_hash,
                            "train_sha256": core.sha256_file(
                                REPO_ROOT / "data" / "KDDTrain+.txt"
                            ),
                            "test_sha256": core.sha256_file(
                                REPO_ROOT / "data" / "KDDTest+.txt"
                            ),
                        },
                    )
                    for seed in subject.SEEDS:
                        stem = f"{prefix}_{architecture}_{test_variant}_s{seed}"
                        prediction_path = prediction_dir / f"{stem}.npz"
                        prediction_dir.mkdir(parents=True, exist_ok=True)
                        raw = np.argmax(probabilities, axis=1).astype(np.int64)
                        core.atomic_npz(
                            prediction_path,
                            labels=labels,
                            probabilities=probabilities.astype(np.float32),
                            raw_predictions=raw,
                            final_predictions=raw,
                        )
                        core.atomic_json(
                            run_dir / f"{stem}.json",
                            {
                                "architecture": architecture,
                                "variant": test_variant,
                                "seed": seed,
                                "kddtest_used_for_selection": False,
                                "prediction_sha256": core.sha256_file(
                                    prediction_path
                                ),
                                "feature_cache_sha256": cache_hash,
                                "epochs_requested": 25, "batch_size": 256,
                                "epochs_completed": 25,
                                "backbone": subject.scaling.ARCHITECTURE_DEFAULTS[architecture]["stage1"].FIXED_BACKBONE,
                                "model_parameters": subject.scaling.ARCHITECTURE_DEFAULTS[architecture]["stage1"].FIXED_BACKBONE["expected_parameters"],
                                "batching": "minority_guaranteed_with_replacement" if subject.BASE_TRAINING[base_training]["batching"] else "ordinary_shuffled",
                                "minority_per_batch_per_class": 1 if subject.BASE_TRAINING[base_training]["batching"] else 0,
                                "cb_beta": 0.99 if subject.BASE_TRAINING[base_training]["focal"] else None,
                                "focal_gamma": 0.5 if subject.BASE_TRAINING[base_training]["focal"] else None,
                                "validation_used_during_training": False,
                                "ctgan_used": False, "deterministic_ops_requested": False,
                            },
                        )

            evaluate_args = argparse.Namespace(
                results_dir=str(results_dir), selection=None
            )
            subject.evaluate_kddtest(evaluate_args)
            latest = core.read_json(
                results_dir / "variant_specific_scaling_kddtest_latest.json"
            )
            summary = pd.read_csv(latest["summary"])
            table = pd.read_csv(latest["rare_f1_table"])
            self.assertEqual(len(summary), 32)
            self.assertEqual(len(table), 8)
            self.assertTrue((summary["runs"] == 3).all())
            import export_paper_tables
            output = results_dir.parent / "tables"
            export_paper_tables.analyze(results_dir, output, REPO_ROOT / "data")
            self.assertEqual(len(list(output.glob("table_*.csv"))), 9)
            focal = pd.read_csv(output / "table_03_focal_parameters.csv")
            self.assertTrue((focal["Beta"] == 0.99).all())
            self.assertTrue((focal["Gamma"] == 0.5).all())
            self.assertEqual(len(pd.read_csv(output / "table_06_kddtest_results.csv")), 8)


if __name__ == "__main__":
    unittest.main()
