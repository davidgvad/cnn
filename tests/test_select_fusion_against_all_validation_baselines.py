from __future__ import annotations

import sys
from pathlib import Path
import unittest
from unittest import mock

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import select_fusion_against_all_validation_baselines as selector  # noqa: E402
import tune_robust_calibrated_super_stack_all as stack  # noqa: E402


def standalone(best: float = 0.60) -> dict[str, dict[str, float]]:
    return {
        "general": {"rare_f1_mean": best - 0.03},
        "focal": {"rare_f1_mean": best - 0.02},
        "batching": {"rare_f1_mean": best - 0.01},
        "scaling": {"rare_f1_mean": best},
    }


class ValidationOnlyFusionSelectorTests(unittest.TestCase):
    @staticmethod
    def synthetic_input(row_count: int = 20) -> stack.ArchitectureInput:
        labels = np.tile(np.arange(5, dtype=np.int64), row_count // 5)
        probabilities = np.empty((3, 3, row_count, 5), dtype=np.float32)
        for seed in range(3):
            for expert in range(3):
                probabilities[seed, expert] = 0.05
                probabilities[seed, expert, np.arange(row_count), labels] = (
                    0.80 - 0.05 * expert
                )
                probabilities[seed, expert] /= probabilities[
                    seed, expert
                ].sum(axis=1, keepdims=True)
        return stack.ArchitectureInput(
            architecture="conv2d",
            seeds=(0, 1, 2),
            probabilities=probabilities,
            labels=labels,
            fold_ids=np.tile(np.arange(4, dtype=np.int64), row_count // 4),
            subtypes=np.asarray(["x"] * row_count),
            source_paths={},
            source_hashes={},
            pointer_paths={},
            pointer_hashes={},
            protocol_paths={},
            protocol_hashes={},
            focal_best={},
        )

    def test_standalone_average_preserves_requested_row_axis(self) -> None:
        architecture_input = self.synthetic_input()
        indices = np.asarray([1, 3, 4, 8, 12, 17], dtype=np.int64)
        metrics = selector.standalone_metrics(architecture_input, indices)
        self.assertIn("average", metrics)
        expected = {
            seed: np.argmax(
                architecture_input.probabilities[seed_index][:, indices, :].mean(
                    axis=0
                ),
                axis=1,
            )
            for seed_index, seed in enumerate(architecture_input.seeds)
        }
        observed = selector.aggregate_prediction_metrics(
            architecture_input.labels[indices],
            expected,
            architecture_input.seeds,
        )
        self.assertAlmostEqual(
            metrics["average"]["rare_f1_mean"], observed["rare_f1_mean"]
        )

    def test_ignores_legacy_macro_and_mcc_guards(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "fixed_average",
                    "family": "fixed",
                    "valid_all_seeds": True,
                    "eligible": True,
                    "meets_macro_guard": True,
                    "meets_mcc_guard": True,
                    "rare_f1_mean": 0.606,
                },
                {
                    "candidate_id": "stack_true_winner",
                    "family": "stack",
                    "valid_all_seeds": True,
                    "eligible": False,
                    "meets_macro_guard": False,
                    "meets_mcc_guard": False,
                    "rare_f1_mean": 0.620,
                },
                {
                    "candidate_id": "stack_old_policy_winner",
                    "family": "stack",
                    "valid_all_seeds": True,
                    "eligible": True,
                    "meets_macro_guard": True,
                    "meets_mcc_guard": True,
                    "rare_f1_mean": 0.615,
                },
            ]
        )
        result = selector.select_best_fusion(frame, standalone(), 0.005)
        self.assertEqual(result["candidate"]["candidate_id"], "stack_true_winner")
        self.assertTrue(result["passes"])

    def test_filters_only_technically_invalid_candidates(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "fixed_average",
                    "family": "fixed",
                    "valid_all_seeds": "true",
                    "eligible": False,
                    "rare_f1_mean": 0.606,
                },
                {
                    "candidate_id": "invalid_extreme",
                    "family": "stack",
                    "valid_all_seeds": "false",
                    "eligible": True,
                    "rare_f1_mean": 0.999,
                },
                {
                    "candidate_id": "valid_stack",
                    "family": "stack",
                    "valid_all_seeds": "true",
                    "eligible": False,
                    "rare_f1_mean": 0.610,
                },
                {
                    "candidate_id": "fixed_general",
                    "family": "fixed",
                    "valid_all_seeds": "true",
                    "eligible": True,
                    "rare_f1_mean": 0.950,
                },
            ]
        )
        result = selector.select_best_fusion(frame, standalone(), 0.005)
        self.assertEqual(result["candidate"]["candidate_id"], "valid_stack")
        self.assertEqual(result["candidate_count"], 2)

    def test_minimum_gain_boundary_is_absolute_and_inclusive(self) -> None:
        for rare_f1, expected in ((0.605, True), (0.604999, False)):
            with self.subTest(rare_f1=rare_f1):
                frame = pd.DataFrame(
                    [
                        {
                            "candidate_id": "stack_candidate",
                            "family": "stack",
                            "valid_all_seeds": True,
                            "rare_f1_mean": rare_f1,
                        }
                    ]
                )
                result = selector.select_best_fusion(frame, standalone(), 0.005)
                self.assertEqual(result["passes"], expected)
                self.assertAlmostEqual(result["required_rare_f1"], 0.605)

    def test_only_real_fusion_families_and_average_are_considered(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "fixed_general",
                    "family": "fixed",
                    "valid_all_seeds": True,
                    "rare_f1_mean": 0.99,
                },
                {
                    "candidate_id": "average_offset_01",
                    "family": "average_offset",
                    "valid_all_seeds": True,
                    "rare_f1_mean": 0.611,
                },
                {
                    "candidate_id": "fixed_average",
                    "family": "fixed",
                    "valid_all_seeds": True,
                    "rare_f1_mean": 0.606,
                },
            ]
        )
        result = selector.select_best_fusion(frame, standalone(), 0.005)
        self.assertEqual(result["candidate"]["candidate_id"], "average_offset_01")
        self.assertEqual(result["candidate_count"], 2)

    def test_nested_acceptance_requires_all_three_conditions(self) -> None:
        passing = selector.nested_acceptance(True, 4, 4, 0.605, 0.600, 0.005)
        self.assertTrue(passing["pooled_performance_pass"])
        self.assertTrue(passing["all_outer_inner_selections_pass"])
        self.assertTrue(passing["nested_audit_pass"])
        self.assertTrue(passing["advance_to_test"])

        below_boundary = selector.nested_acceptance(
            True, 4, 4, 0.604999, 0.600, 0.005
        )
        self.assertFalse(below_boundary["pooled_performance_pass"])
        self.assertFalse(below_boundary["advance_to_test"])

        missing_outer_pass = selector.nested_acceptance(
            True, 3, 4, 0.620, 0.600, 0.005
        )
        self.assertTrue(missing_outer_pass["pooled_performance_pass"])
        self.assertFalse(missing_outer_pass["all_outer_inner_selections_pass"])
        self.assertFalse(missing_outer_pass["advance_to_test"])

        failed_final = selector.nested_acceptance(
            False, 4, 4, 0.620, 0.600, 0.005
        )
        self.assertTrue(failed_final["nested_audit_pass"])
        self.assertFalse(failed_final["advance_to_test"])

    def test_outer_refits_use_exact_complements_and_cover_every_row(self) -> None:
        architecture_input = self.synthetic_input()
        row_count = len(architecture_input.labels)
        fold_ids = architecture_input.fold_ids
        calls: list[tuple[int, str, np.ndarray, np.ndarray]] = []

        def fake_fit(
            _architecture_input: stack.ArchitectureInput,
            seed: int,
            config: dict[str, object],
            train_indices: np.ndarray,
            prediction_indices: np.ndarray,
            _settings: object,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
            calls.append(
                (
                    int(seed),
                    str(config["candidate_id"]),
                    train_indices.copy(),
                    prediction_indices.copy(),
                )
            )
            predictions = np.full(len(prediction_indices), seed, dtype=np.int64)
            probabilities = np.zeros((len(prediction_indices), 5), dtype=np.float64)
            probabilities[:, seed] = 1.0
            return predictions, probabilities, {}

        assembled = {
            seed: np.full(row_count, -1, dtype=np.int64)
            for seed in architecture_input.seeds
        }
        with mock.patch.object(
            selector.stack,
            "fit_selected_candidate",
            side_effect=fake_fit,
        ):
            for outer_fold in range(4):
                config = {"candidate_id": f"outer_{outer_fold}"}
                train, heldout, predictions = selector.refit_outer_candidate(
                    architecture_input,
                    config,
                    outer_fold,
                    object(),  # type: ignore[arg-type]
                )
                np.testing.assert_array_equal(
                    train, np.flatnonzero(fold_ids != outer_fold)
                )
                np.testing.assert_array_equal(
                    heldout, np.flatnonzero(fold_ids == outer_fold)
                )
                self.assertEqual(np.intersect1d(train, heldout).size, 0)
                for seed in architecture_input.seeds:
                    assembled[seed][heldout] = predictions[seed]

        self.assertEqual(len(calls), 12)
        for seed, candidate_id, train, heldout in calls:
            outer_fold = int(candidate_id.rsplit("_", 1)[1])
            np.testing.assert_array_equal(train, np.flatnonzero(fold_ids != outer_fold))
            np.testing.assert_array_equal(
                heldout, np.flatnonzero(fold_ids == outer_fold)
            )
            self.assertEqual(seed, int(assembled[seed][heldout[0]]))
        for seed in architecture_input.seeds:
            np.testing.assert_array_equal(
                assembled[seed], np.full(row_count, seed, dtype=np.int64)
            )


if __name__ == "__main__":
    unittest.main()
