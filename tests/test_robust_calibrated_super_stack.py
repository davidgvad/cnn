from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import tune_robust_calibrated_super_stack_all as stack  # noqa: E402


class RobustSuperStackUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = stack.SearchSettings(
            epsilon=1e-12,
            max_iter=300,
            tolerance=1e-5,
            temperature_min=0.01,
            temperature_max=100.0,
            temperature_xatol=1e-7,
            temperature_maxiter=300,
            candidate_chunk_size=17,
            macro_guard=0.005,
            mcc_guard=0.005,
        )

    def test_feature_shapes_names_and_finiteness(self) -> None:
        rng = np.random.default_rng(4)
        probabilities = rng.dirichlet(np.ones(5), size=(31, 3))
        for feature_set, expected in (("F0", 15), ("F1", 30), ("F2", 36)):
            values = stack.build_features(probabilities, feature_set, 1e-12)
            self.assertEqual(values.shape, (31, expected))
            self.assertEqual(len(stack.feature_names(feature_set)), expected)
            self.assertTrue(np.isfinite(values).all())

    def test_temperature_scaling_is_normalized_and_preserves_argmax(self) -> None:
        rng = np.random.default_rng(9)
        probabilities = rng.dirichlet(np.ones(5), size=70)
        before = np.argmax(probabilities, axis=1)
        for temperature in (0.1, 0.7, 1.0, 3.0, 20.0):
            scaled = stack.temperature_scale(probabilities, temperature, 1e-12)
            np.testing.assert_allclose(scaled.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
            np.testing.assert_array_equal(np.argmax(scaled, axis=1), before)

    def test_offset_probabilities_reproduce_predictions(self) -> None:
        rng = np.random.default_rng(21)
        probabilities = rng.dirichlet(np.ones(5), size=80)
        adjusted = stack.offset_decision_probabilities(
            probabilities, 0.75, -0.5, 1e-12
        )
        predicted = stack.apply_offsets(probabilities, 0.75, -0.5, 1e-12)
        np.testing.assert_allclose(adjusted.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(predicted, np.argmax(adjusted, axis=1))

    def test_fitted_temperature_does_not_increase_training_nll(self) -> None:
        rng = np.random.default_rng(10)
        labels = np.tile(np.arange(5), 50)
        logits = rng.normal(size=(len(labels), 5))
        logits[np.arange(len(labels)), labels] += 0.7
        logits *= 4.0
        probabilities = np.exp(logits - stack.logsumexp(logits, axis=1, keepdims=True))
        temperature, _boundary = stack.fit_temperature(
            probabilities, labels, self.settings
        )
        calibrated = stack.temperature_scale(probabilities, temperature, 1e-12)
        rows = np.arange(len(labels))
        raw_nll = -np.log(probabilities[rows, labels]).mean()
        calibrated_nll = -np.log(calibrated[rows, labels]).mean()
        self.assertLessEqual(calibrated_nll, raw_nll + 1e-10)

    def test_normalized_class_weights_have_mean_one(self) -> None:
        labels = np.asarray([0] * 30 + [1] * 12 + [2] * 6 + [3] * 3 + [4] * 49)
        for q in stack.Q_VALUES:
            weights = stack.normalized_sample_weights(labels, q)
            self.assertAlmostEqual(float(weights.mean()), 1.0, places=12)
        np.testing.assert_allclose(
            stack.normalized_sample_weights(labels, 0.0), np.ones(len(labels))
        )

    def test_selected_stack_refit_produces_complete_model_state(self) -> None:
        rng = np.random.default_rng(18)
        labels = np.tile(np.repeat(np.arange(5, dtype=np.int64), 4), 4)
        folds = np.repeat(np.arange(4, dtype=np.int64), 20)
        probabilities = np.empty((3, 3, len(labels), 5), dtype=np.float32)
        for seed_index in range(3):
            for expert_index in range(3):
                logits = rng.normal(scale=0.8, size=(len(labels), 5))
                logits[np.arange(len(labels)), labels] += 1.1 + 0.1 * expert_index
                probabilities[seed_index, expert_index] = np.exp(
                    logits - stack.logsumexp(logits, axis=1, keepdims=True)
                )
        architecture_input = stack.ArchitectureInput(
            architecture="conv2d",
            seeds=(0, 1, 2),
            probabilities=probabilities,
            labels=labels,
            fold_ids=folds,
            subtypes=np.asarray([f"class_{label}" for label in labels], dtype="U16"),
            source_paths={},
            source_hashes={},
            pointer_paths={},
            pointer_hashes={},
            protocol_paths={},
            protocol_hashes={},
            focal_best={},
        )
        config = {
            "candidate_id": "synthetic_stack",
            "family": "stack",
            "fixed_expert": "none",
            "calibration": "temperature",
            "feature_set": "F2",
            "q": 0.5,
            "C": 0.1,
            "rho": 0.5,
            "delta_r2l": 0.25,
            "delta_u2r": -0.25,
        }
        train_indices = np.flatnonzero(folds != 0)
        prediction_indices = np.flatnonzero(folds == 0)
        predicted, decision_probabilities, state = stack.fit_selected_candidate(
            architecture_input,
            0,
            config,
            train_indices,
            prediction_indices,
            self.settings,
        )
        self.assertEqual(predicted.shape, (len(prediction_indices),))
        self.assertEqual(decision_probabilities.shape, (len(prediction_indices), 5))
        np.testing.assert_allclose(
            decision_probabilities.sum(axis=1), 1.0, atol=1e-12, rtol=0.0
        )
        np.testing.assert_array_equal(predicted, np.argmax(decision_probabilities, axis=1))
        self.assertEqual(state["temperatures"].shape, (3,))
        self.assertTrue(np.all(state["temperatures"] > 0.0))
        self.assertEqual(state["feature_mean"].shape, (36,))
        self.assertEqual(state["feature_scale"].shape, (36,))
        self.assertEqual(state["coef"].shape, (5, 36))
        self.assertEqual(state["intercept"].shape, (5,))

    def test_process_worker_task_and_resume_cache(self) -> None:
        if "fork" not in stack.mp.get_all_start_methods():
            self.skipTest("The production runner requires POSIX fork.")
        rng = np.random.default_rng(27)
        labels = np.tile(np.repeat(np.arange(5, dtype=np.int64), 2), 4)
        folds = np.repeat(np.arange(4, dtype=np.int64), 10)
        probabilities = np.empty((1, 3, len(labels), 5), dtype=np.float32)
        for expert_index in range(3):
            logits = rng.normal(scale=0.7, size=(len(labels), 5))
            logits[np.arange(len(labels)), labels] += 1.0 + 0.1 * expert_index
            probabilities[0, expert_index] = np.exp(
                logits - stack.logsumexp(logits, axis=1, keepdims=True)
            )
        architecture_input = stack.ArchitectureInput(
            architecture="conv2d",
            seeds=(0,),
            probabilities=probabilities,
            labels=labels,
            fold_ids=folds,
            subtypes=np.asarray([f"class_{label}" for label in labels], dtype="U16"),
            source_paths={},
            source_hashes={},
            pointer_paths={},
            pointer_hashes={},
            protocol_paths={},
            protocol_hashes={},
            focal_best={},
        )
        with tempfile.TemporaryDirectory() as directory:
            task = stack.MetaTask(
                architecture="conv2d",
                stage="outer_0",
                outer_fold=0,
                seed=0,
                calibration="raw",
                feature_set="F0",
                fingerprint="synthetic-process-task",
                cache_path=str(Path(directory) / "task.npz"),
            )
            try:
                computed = stack.run_search_tasks(
                    architecture_input,
                    [task],
                    self.settings,
                    workers=1,
                    threads_per_worker=1,
                    resume=False,
                )
            except PermissionError as error:
                self.skipTest(f"Process semaphores are blocked by this sandbox: {error}")
            self.assertEqual(computed.loc[0, "status"], "computed")
            resumed = stack.run_search_tasks(
                architecture_input,
                [task],
                self.settings,
                workers=1,
                threads_per_worker=1,
                resume=True,
            )
            self.assertEqual(resumed.loc[0, "status"], "resumed")
            self.assertTrue(stack._task_cache_valid(task))

    def test_subtype_balancing_equalizes_rare_subtype_mass(self) -> None:
        labels = np.asarray([0] * 5 + [1] * 3 + [2] * 8 + [3] * 6 + [4] * 9)
        subtypes = np.asarray(
            ["dos"] * 5
            + ["probe"] * 3
            + ["r_a"] * 6
            + ["r_b"] * 2
            + ["u_a"] * 3
            + ["u_b"] * 2
            + ["u_c"]
            + ["normal"] * 9
        )
        weights, details = stack.subtype_balanced_weights(labels, subtypes)
        self.assertAlmostEqual(float(weights[labels == 2].sum()), 8.0)
        self.assertAlmostEqual(float(weights[labels == 3].sum()), 6.0)
        for class_id in (2, 3):
            masses = [
                float(weights[(labels == class_id) & (subtypes == subtype)].sum())
                for subtype in np.unique(subtypes[labels == class_id])
            ]
            np.testing.assert_allclose(masses, np.full(len(masses), masses[0]))
        self.assertEqual(details["R2L"]["rows"], 8)
        self.assertEqual(details["U2R"]["rows"], 6)

    def test_confusion_batch_matches_scalar_confusions(self) -> None:
        labels = np.asarray([0, 1, 2, 3, 4, 2, 3, 4])
        predictions = np.column_stack(
            [labels, np.asarray([1, 1, 2, 4, 4, 0, 3, 2])]
        )
        matrices = stack.confusion_batch(labels, predictions)
        for candidate in range(predictions.shape[1]):
            expected = np.bincount(
                labels * 5 + predictions[:, candidate], minlength=25
            ).reshape(5, 5)
            np.testing.assert_array_equal(matrices[candidate], expected)

    def test_canonical_candidate_counts(self) -> None:
        self.assertEqual(len(stack.stack_descriptor_frame("raw", "F0")), 11_340)
        self.assertEqual(6 * 11_340 + 84, 68_124)
        self.assertEqual(len(set(stack.stack_descriptor_frame("raw", "F0")["candidate_id"])), 11_340)

    def test_one_synthetic_meta_task_end_to_end(self) -> None:
        rng = np.random.default_rng(33)
        rows_per_fold_class = 3
        labels = np.tile(
            np.repeat(np.arange(5, dtype=np.int64), rows_per_fold_class), 4
        )
        fold_ids = np.repeat(np.arange(4, dtype=np.int64), 5 * rows_per_fold_class)
        row_count = len(labels)
        subtypes = np.asarray([f"class_{label}" for label in labels], dtype="U16")
        probabilities = np.empty((3, 3, row_count, 5), dtype=np.float32)
        for seed_index in range(3):
            for expert_index in range(3):
                logits = rng.normal(scale=0.7, size=(row_count, 5))
                logits[np.arange(row_count), labels] += 1.0 + 0.1 * expert_index
                probabilities[seed_index, expert_index] = np.exp(
                    logits - stack.logsumexp(logits, axis=1, keepdims=True)
                )
        stack._WORKER_CONTEXT = {
            "architecture": "conv2d",
            "seeds": (0, 1, 2),
            "probabilities": probabilities,
            "labels": labels,
            "fold_ids": fold_ids,
            "subtypes": subtypes,
            "settings": self.settings,
        }
        with tempfile.TemporaryDirectory() as directory:
            task = stack.MetaTask(
                architecture="conv2d",
                stage="outer_0",
                outer_fold=0,
                seed=0,
                calibration="raw",
                feature_set="F0",
                fingerprint="synthetic-task",
                cache_path=str(Path(directory) / "task.npz"),
            )
            result = stack._run_meta_task(task)
            self.assertTrue(stack._task_cache_valid(task))
            self.assertEqual(result["total_configurations"], 35)
            arrays = stack.load_task_arrays(task)
            self.assertEqual(
                arrays["natural_metrics"].shape,
                (5, 7, 4, 81, len(stack.METRICS)),
            )
            self.assertTrue(np.isfinite(arrays["natural_metrics"]).all())

    def test_synthetic_final_artifact_path_for_fixed_winner(self) -> None:
        rng = np.random.default_rng(44)
        labels = np.tile(np.repeat(np.arange(5), 3), 4).astype(np.int64)
        folds = np.repeat(np.arange(4), 15).astype(np.int64)
        subtypes = np.asarray([f"class_{label}" for label in labels], dtype="U16")
        probabilities = np.empty((3, 3, len(labels), 5), dtype=np.float32)
        for seed_index in range(3):
            for expert_index in range(3):
                logits = rng.normal(size=(len(labels), 5))
                logits[np.arange(len(labels)), labels] += 1.2
                probabilities[seed_index, expert_index] = np.exp(
                    logits - stack.logsumexp(logits, axis=1, keepdims=True)
                )
        architecture_input = stack.ArchitectureInput(
            architecture="conv2d",
            seeds=(0, 1, 2),
            probabilities=probabilities,
            labels=labels,
            fold_ids=folds,
            subtypes=subtypes,
            source_paths={},
            source_hashes={},
            pointer_paths={},
            pointer_hashes={},
            protocol_paths={},
            protocol_hashes={},
            focal_best={},
        )
        balanced_weights, details = stack.subtype_balanced_weights(labels, subtypes)
        seed_values = []
        for seed_index in range(3):
            average = probabilities[seed_index].transpose(1, 0, 2).mean(axis=1)
            predicted = np.argmax(average, axis=1)
            seed_values.append(
                stack.metric_values_for_predictions(labels, predicted, balanced_weights)
            )
        natural = np.stack([item[0] for item in seed_values])
        balanced_rare = np.asarray([item[1] for item in seed_values])
        robust = np.minimum(natural[:, stack.METRIC_INDEX["rare_f1"]], balanced_rare)
        row = {
            "rank": 1,
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
            "robust_rare_f1_mean": float(robust.mean()),
            "robust_rare_f1_std": float(robust.std(ddof=1)),
            "rare_f1_mean": float(natural[:, stack.METRIC_INDEX["rare_f1"]].mean()),
            "rare_f1_std": float(natural[:, stack.METRIC_INDEX["rare_f1"]].std(ddof=1)),
            "balanced_rare_f1_mean": float(balanced_rare.mean()),
            "minimum_rare_f1_mean": float(
                np.minimum(
                    natural[:, stack.METRIC_INDEX["r2l_f1"]],
                    natural[:, stack.METRIC_INDEX["u2r_f1"]],
                ).mean()
            ),
            "macro_f1_mean": float(natural[:, stack.METRIC_INDEX["macro_f1"]].mean()),
            "mcc_mean": float(natural[:, stack.METRIC_INDEX["mcc"]].mean()),
            "eligible": True,
        }

        def fake_ranking(*_args, **_kwargs):
            return pd.DataFrame([row]), {
                "rows": len(labels),
                "eligible_candidates": 1,
                "invalid_candidates": 0,
                "subtype_balance": details,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            train_path = root / "KDDTrain+.txt"
            protocol_path.write_text("{}\n", encoding="utf-8")
            train_path.write_text("synthetic\n", encoding="utf-8")
            with mock.patch.object(stack, "build_stage_ranking", side_effect=fake_ranking), mock.patch.object(
                stack, "CANONICAL_LIBRARY_COUNT", 1
            ):
                result = stack.evaluate_architecture(
                    architecture_input,
                    [],
                    self.settings,
                    "synthetic",
                    root,
                    1,
                    "experiment",
                    protocol_path,
                    train_path,
                )
            self.assertTrue(Path(result["best_config"]).is_file())
            self.assertTrue(Path(result["final_seed_models"]).is_file())
            with np.load(result["nested_predictions"], allow_pickle=False) as artifact:
                np.testing.assert_array_equal(
                    artifact["predictions"],
                    np.argmax(artifact["decision_probabilities"], axis=2),
                )


if __name__ == "__main__":
    unittest.main()
