from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import evaluate_final_natural_rare_super_stack_kddtest as final_stack  # noqa: E402
import tune_robust_calibrated_super_stack_all as stack  # noqa: E402


class FrozenNaturalRareKDDTestUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = stack.SearchSettings(
            epsilon=1e-12,
            max_iter=500,
            tolerance=1e-6,
            temperature_min=0.01,
            temperature_max=100.0,
            temperature_xatol=1e-7,
            temperature_maxiter=300,
            candidate_chunk_size=17,
            macro_guard=0.005,
            mcc_guard=0.005,
        )

    def synthetic_input(self) -> stack.ArchitectureInput:
        rng = np.random.default_rng(71)
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
        return stack.ArchitectureInput(
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

    def test_frozen_candidate_identities(self) -> None:
        expected = {
            "conv2d": "stack_c0_f0_q1_C5_r1_o30",
            "conv1d": "stack_c0_f0_q0_C3_r2_o41",
            "transformer": "stack_c0_f0_q4_C1_r1_o30",
            "mlp": "stack_c1_f0_q1_C4_r4_o11",
        }
        self.assertEqual(set(final_stack.FROZEN_NATURAL_CONFIGS), set(expected))
        for architecture, candidate_id in expected.items():
            self.assertEqual(
                final_stack.FROZEN_NATURAL_CONFIGS[architecture]["candidate_id"],
                candidate_id,
            )
            self.assertEqual(
                final_stack.FROZEN_NATURAL_CONFIGS[architecture]["feature_set"],
                "F0",
            )

    def test_natural_ranking_filters_guards_and_uses_declared_order(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": "ineligible_highest",
                    "eligible": False,
                    "family": "stack",
                    "calibration": "raw",
                    "feature_set": "F0",
                    "q": 0.0,
                    "C": 1.0,
                    "rho": 0.5,
                    "delta_r2l": 0.0,
                    "delta_u2r": 0.0,
                    "rare_f1_mean": 0.90,
                    "minimum_rare_f1_mean": 0.70,
                    "macro_f1_mean": 0.88,
                    "mcc_mean": 0.98,
                    "rare_f1_std": 0.01,
                },
                {
                    "candidate_id": "eligible_second",
                    "eligible": "true",
                    "family": "stack",
                    "calibration": "raw",
                    "feature_set": "F0",
                    "q": 0.0,
                    "C": 1.0,
                    "rho": 0.5,
                    "delta_r2l": 0.0,
                    "delta_u2r": 0.0,
                    "rare_f1_mean": 0.75,
                    "minimum_rare_f1_mean": 0.60,
                    "macro_f1_mean": 0.89,
                    "mcc_mean": 0.99,
                    "rare_f1_std": 0.02,
                },
                {
                    "candidate_id": "eligible_winner",
                    "eligible": "TRUE",
                    "family": "stack",
                    "calibration": "raw",
                    "feature_set": "F0",
                    "q": 0.0,
                    "C": 1.0,
                    "rho": 0.5,
                    "delta_r2l": 0.0,
                    "delta_u2r": 0.0,
                    "rare_f1_mean": 0.76,
                    "minimum_rare_f1_mean": 0.59,
                    "macro_f1_mean": 0.88,
                    "mcc_mean": 0.98,
                    "rare_f1_std": 0.03,
                },
            ]
        )
        ordered = final_stack.natural_ranking(frame)
        self.assertEqual(ordered["candidate_id"].tolist(), [
            "eligible_winner",
            "eligible_second",
        ])

    def test_saved_state_inference_matches_training_implementation(self) -> None:
        architecture_input = self.synthetic_input()
        all_indices = np.arange(len(architecture_input.labels), dtype=np.int64)
        configs = (
            {
                "candidate_id": "synthetic_raw",
                "family": "stack",
                "fixed_expert": "none",
                "calibration": "raw",
                "feature_set": "F0",
                "q": 0.25,
                "C": 1.0,
                "rho": 0.25,
                "delta_r2l": -0.25,
                "delta_u2r": 0.25,
            },
            {
                "candidate_id": "synthetic_temperature",
                "family": "stack",
                "fixed_expert": "none",
                "calibration": "temperature",
                "feature_set": "F0",
                "q": 0.25,
                "C": 1.0,
                "rho": 1.0,
                "delta_r2l": -0.75,
                "delta_u2r": -0.50,
            },
        )
        for config in configs:
            with self.subTest(calibration=config["calibration"]):
                expected_predictions, expected_probabilities, state = (
                    stack.fit_selected_candidate(
                        architecture_input,
                        0,
                        config,
                        all_indices,
                        all_indices,
                        self.settings,
                    )
                )
                raw = architecture_input.probabilities[0].transpose(1, 0, 2)
                observed = final_stack.predict_with_fitted_state(
                    raw, config, state, self.settings
                )
                np.testing.assert_array_equal(
                    observed["predictions"], expected_predictions
                )
                np.testing.assert_allclose(
                    observed["decision_probabilities"],
                    expected_probabilities,
                    atol=1e-12,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    observed["decision_probabilities"].sum(axis=1),
                    1.0,
                    atol=1e-12,
                    rtol=0.0,
                )


if __name__ == "__main__":
    unittest.main()
