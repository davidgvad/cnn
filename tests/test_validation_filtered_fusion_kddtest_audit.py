from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import audit_validation_filtered_fusion_kddtest as audit  # noqa: E402
import run_no_ctgan_model_ablation_4gpu as core  # noqa: E402
import tune_robust_calibrated_super_stack_all as stack  # noqa: E402


class ValidationFilteredFusionAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = stack.SearchSettings(
            epsilon=1e-12,
            max_iter=100,
            tolerance=1e-6,
            temperature_min=0.01,
            temperature_max=100.0,
            temperature_xatol=1e-7,
            temperature_maxiter=100,
            candidate_chunk_size=2,
            macro_guard=0.005,
            mcc_guard=0.005,
        )

    @staticmethod
    def standalone_summary() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"method": "general", "rare_f1_mean": 0.60},
                {"method": "focal", "rare_f1_mean": 0.62},
                {"method": "batching", "rare_f1_mean": 0.64},
                {"method": "scaling", "rare_f1_mean": 0.63},
            ]
        )

    def test_validation_filter_is_inclusive_and_ignores_legacy_guards(self) -> None:
        ranking = pd.DataFrame(
            [
                {
                    "candidate_id": "fixed_general",
                    "family": "fixed",
                    "valid_all_seeds": True,
                    "eligible": True,
                    "rare_f1_mean": 0.99,
                    "rare_f1_std": 0.01,
                },
                {
                    "candidate_id": "stack_boundary",
                    "family": "stack",
                    "valid_all_seeds": "true",
                    "eligible": False,
                    "rare_f1_mean": 0.635,
                    "rare_f1_std": 0.02,
                },
                {
                    "candidate_id": "average_offset_below",
                    "family": "average_offset",
                    "valid_all_seeds": True,
                    "eligible": True,
                    "rare_f1_mean": 0.6349,
                    "rare_f1_std": 0.01,
                },
                {
                    "candidate_id": "invalid_high",
                    "family": "stack",
                    "valid_all_seeds": False,
                    "eligible": True,
                    "rare_f1_mean": 0.90,
                    "rare_f1_std": 0.01,
                },
            ]
        )
        survivors, details = audit.validation_survivors(
            ranking, self.standalone_summary(), 0.005
        )
        self.assertEqual(survivors["candidate_id"].tolist(), ["stack_boundary"])
        self.assertEqual(details["best_method"], "batching")
        self.assertAlmostEqual(details["threshold"], 0.635)

    def test_validation_survivors_preserve_ranking_architecture_column(self) -> None:
        ranking = pd.DataFrame(
            [
                {
                    "architecture": "conv2d",
                    "candidate_id": "fixed_average",
                    "family": "fixed",
                    "valid_all_seeds": True,
                    "rare_f1_mean": 0.64,
                    "rare_f1_std": 0.01,
                }
            ]
        )
        survivors, _ = audit.validation_survivors(
            ranking, self.standalone_summary(), 0.005
        )
        self.assertEqual(survivors["architecture"].tolist(), ["conv2d"])

    def test_frozen_scaling_is_applied_to_general_probabilities(self) -> None:
        labels = np.asarray([0, 2, 3, 4, 1], dtype=np.int64)
        probabilities_by_seed = {}
        for seed in (0, 1, 2):
            raw = np.full((len(labels), 3, 5), 0.01, dtype=np.float64)
            for expert in range(3):
                raw[np.arange(len(labels)), expert, labels] = 0.80
                raw[:, expert, :] /= raw[:, expert, :].sum(axis=1, keepdims=True)
            probabilities_by_seed[seed] = raw
        per_seed, summary = audit.standalone_metrics_from_probabilities(
            labels, probabilities_by_seed, (0, 1, 2), 1.0, 4.0
        )
        self.assertEqual(set(summary["method"]), set(audit.STANDALONES))
        observed = per_seed[
            (per_seed["method"] == "scaling") & (per_seed["seed"] == 0)
        ].iloc[0]
        expected_predictions = core.apply_class_score_scaling(
            probabilities_by_seed[0][:, 0, :], {2: 1.0, 3: 4.0}
        )
        expected = core.calculate_metrics(labels, expected_predictions)
        self.assertAlmostEqual(observed["rare_f1"], expected["rare_f1"])

    def test_vectorized_candidate_scoring_matches_direct_predictions(self) -> None:
        labels = np.asarray([0, 1, 2, 3, 4, 2, 3], dtype=np.int64)
        rng = np.random.default_rng(14)
        average = rng.dirichlet(np.ones(5), size=len(labels))
        fitted = rng.dirichlet(np.ones(5), size=len(labels))
        candidates = pd.DataFrame(
            [
                {
                    "candidate_id": "a",
                    "family": "stack",
                    "rho": 0.25,
                    "delta_r2l": -0.25,
                    "delta_u2r": 0.50,
                },
                {
                    "candidate_id": "b",
                    "family": "stack",
                    "rho": 0.75,
                    "delta_r2l": 0.25,
                    "delta_u2r": -0.50,
                },
            ]
        )
        scored = audit.score_probability_candidates(
            labels, average, candidates, self.settings, fitted
        ).set_index("candidate_id")
        for row in candidates.to_dict(orient="records"):
            blended = (1.0 - row["rho"]) * average + row["rho"] * fitted
            predictions = stack.apply_offsets(
                blended,
                row["delta_r2l"],
                row["delta_u2r"],
                self.settings.epsilon,
            )
            expected = core.calculate_metrics(labels, predictions)
            for metric in audit.METRICS:
                self.assertAlmostEqual(
                    scored.loc[row["candidate_id"], metric], expected[metric]
                )

    def test_test_guard_reports_and_ranks_every_passing_candidate(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "strong_test",
                    "validation_pass": True,
                    "validation_gap": -0.004,
                    "rare_f1_mean": 0.636,
                    "test_rare_f1_mean": 0.650,
                    "test_rare_f1_std": 0.02,
                },
                {
                    "candidate_id": "balanced",
                    "validation_pass": True,
                    "validation_gap": 0.002,
                    "rare_f1_mean": 0.642,
                    "test_rare_f1_mean": 0.638,
                    "test_rare_f1_std": 0.01,
                },
                {
                    "candidate_id": "test_failure",
                    "validation_pass": True,
                    "validation_gap": 0.010,
                    "rare_f1_mean": 0.650,
                    "test_rare_f1_mean": 0.634,
                    "test_rare_f1_std": 0.01,
                },
            ]
        )
        guarded, details = audit.apply_test_guard(
            frame, self.standalone_summary(), 0.005
        )
        passing = guarded[guarded["pass_both"]]
        self.assertEqual(set(passing["candidate_id"]), {"strong_test", "balanced"})
        self.assertEqual(guarded.iloc[0]["candidate_id"], "balanced")
        self.assertEqual(details["pass_both_count"], 2)

    def test_stack_tasks_share_all_q_c_candidates_by_calibration_features(self) -> None:
        survivors = pd.DataFrame(
            [
                {
                    "candidate_id": "s0",
                    "family": "stack",
                    "calibration": "raw",
                    "feature_set": "F0",
                    "q": 0.0,
                    "C": 1.0,
                    "rho": 0.25,
                    "delta_r2l": 0.0,
                    "delta_u2r": 0.0,
                },
                {
                    "candidate_id": "s1",
                    "family": "stack",
                    "calibration": "raw",
                    "feature_set": "F0",
                    "q": 0.25,
                    "C": 10.0,
                    "rho": 0.50,
                    "delta_r2l": 0.25,
                    "delta_u2r": -0.25,
                },
                {
                    "candidate_id": "average_offset_01",
                    "family": "average_offset",
                    "calibration": "none",
                    "feature_set": "none",
                    "q": np.nan,
                    "C": np.nan,
                },
            ]
        )
        tasks = audit.stack_tasks(survivors, (0, 1, 2))
        self.assertEqual(len(tasks), 3)
        self.assertEqual({task["seed"] for task in tasks}, {0, 1, 2})
        self.assertTrue(all(len(task["candidates"]) == 2 for task in tasks))
        self.assertTrue(all(task["planned_fit_count"] == 2 for task in tasks))

    def test_progress_message_reports_architecture_and_overall_percentages(self) -> None:
        message = audit.progress_message("Conv2D", 100, 400, 250, 1000)
        self.assertIn("Conv2D refit progress: 100/400 (25.0%)", message)
        self.assertIn("overall: 250/1,000 (25.0%)", message)

    def test_direct_stack_worker_refits_once_per_q_c_and_scores_all_rules(self) -> None:
        rng = np.random.default_rng(33)
        labels = np.tile(np.arange(5, dtype=np.int64), 8)
        probabilities = np.empty((3, 3, len(labels), 5), dtype=np.float32)
        for seed_index in range(3):
            for expert_index in range(3):
                logits = rng.normal(scale=0.5, size=(len(labels), 5))
                logits[np.arange(len(labels)), labels] += 1.0 + 0.1 * expert_index
                probabilities[seed_index, expert_index] = np.exp(
                    logits - stack.logsumexp(logits, axis=1, keepdims=True)
                )
        architecture_input = stack.ArchitectureInput(
            architecture="conv2d",
            seeds=(0, 1, 2),
            probabilities=probabilities,
            labels=labels,
            fold_ids=np.tile(np.arange(4, dtype=np.int64), len(labels) // 4),
            subtypes=np.asarray(["x"] * len(labels)),
            source_paths={},
            source_hashes={},
            pointer_paths={},
            pointer_hashes={},
            protocol_paths={},
            protocol_hashes={},
            focal_best={},
        )
        test_raw = probabilities[0].transpose(1, 0, 2).astype(np.float64)
        candidates = [
            {
                "candidate_id": "rule_0",
                "family": "stack",
                "calibration": "raw",
                "feature_set": "F0",
                "q": 0.0,
                "C": 1.0,
                "rho": 0.25,
                "delta_r2l": 0.0,
                "delta_u2r": 0.0,
            },
            {
                "candidate_id": "rule_1",
                "family": "stack",
                "calibration": "raw",
                "feature_set": "F0",
                "q": 0.0,
                "C": 1.0,
                "rho": 0.75,
                "delta_r2l": 0.25,
                "delta_u2r": -0.25,
            },
        ]
        audit._AUDIT_CONTEXT = {
            "architecture_input": architecture_input,
            "test_probabilities": {0: test_raw},
            "test_labels": labels,
            "settings": self.settings,
        }
        try:
            result = audit._score_stack_task(
                {
                    "seed": 0,
                    "calibration": "raw",
                    "feature_set": "F0",
                    "candidates": candidates,
                }
            )
        finally:
            audit._AUDIT_CONTEXT = None
        self.assertEqual(result["fit_count"], 1)
        self.assertEqual(result["failures"], [])
        self.assertEqual(
            {row["candidate_id"] for row in result["rows"]},
            {"rule_0", "rule_1"},
        )

    def test_task_cache_round_trip_and_fingerprint_rejection(self) -> None:
        result = {
            "seed": 1,
            "calibration": "raw",
            "feature_set": "F0",
            "rows": [
                {
                    "candidate_id": "candidate",
                    **{metric: 0.1 + index / 100.0 for index, metric in enumerate(audit.METRICS)},
                }
            ],
            "failures": [],
            "fit_count": 1,
            "temperatures": [
                {"expert": "general", "temperature": 1.0, "boundary": False}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.npz"
            audit.cache_task_result(path, "fingerprint", result)
            loaded = audit.load_cached_task(path, "fingerprint")
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded["seed"], 1)
            self.assertEqual(loaded["rows"][0]["candidate_id"], "candidate")
            self.assertEqual(loaded["fit_count"], 1)
            self.assertIsNone(audit.load_cached_task(path, "different"))


if __name__ == "__main__":
    unittest.main()
