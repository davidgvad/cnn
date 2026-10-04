from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import experiment_runtime
from paper_config import load_config
import paper_metrics

spec = importlib.util.spec_from_file_location("reproduce", ROOT / "reproduction/reproduce.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class PaperReproductionTests(unittest.TestCase):
    def test_metrics_match_independent_sklearn_for_missing_and_rare_classes(self):
        labels = np.array([0, 0, 1, 2, 2, 3, 3, 4, 4])
        predicted = np.array([0, 1, 1, 3, 4, 3, 4, 4, 4])
        m = paper_metrics.metrics_from_confusion(paper_metrics.confusion(labels, predicted))
        per_class = f1_score(labels, predicted, labels=np.arange(5), average=None, zero_division=0)
        self.assertAlmostEqual(m["rare_f1"], per_class[2:4].mean())
        self.assertAlmostEqual(m["macro_f1"], per_class.mean())
        self.assertAlmostEqual(m["mcc"], matthews_corrcoef(labels, predicted))

    def test_frozen_commands_keep_all_folds_seeds_and_batch_size_on_any_hardware(self):
        config = load_config()
        for count in (1, 2, 8):
            parser = argparse.ArgumentParser();experiment_runtime.add_arguments(parser)
            execution = experiment_runtime.select_runtime(parser.parse_args([]), gpu_count=count)
            oof, final = entry.build_commands(config, Path("external data"), Path("external results"), execution)
            self.assertEqual(len(oof), 16)
            self.assertEqual(len(final), 1)
            for command in oof + final:
                self.assertEqual(command[command.index("--epochs") + 1], "25")
                self.assertEqual(command[command.index("--batch-size") + 1], "256")
                self.assertEqual(command[command.index("--seeds") + 1:command.index("--seeds") + 4], ["0", "1", "2"])
            self.assertEqual(sum("--betas" in command for command in oof), 4)
            self.assertEqual(sum("--coefficient-values" in command for command in oof), 12)

    def test_score_scaling_divides_scores_and_has_stable_tie_rule(self):
        p = np.array([[.1, .1, .3, .3, .2], [.2, .2, .2, .2, .2]], dtype=np.float32)
        np.testing.assert_array_equal(paper_metrics.scaled_predictions(p, 2., .5), [3, 3])
        np.testing.assert_array_equal(paper_metrics.scaled_predictions(p, 1., 1.), [2, 0])

    def test_interactions_of_additive_factorial_are_zero(self):
        values = {name: 10 + 3*bits[0] + 5*bits[1] + 7*bits[2] for name, (_, _, bits) in paper_metrics.CONFIGS.items()}
        result = paper_metrics.contrasts(values)
        self.assertEqual(result["marginal_pp"], {"focal": 3., "batching": 5., "scaling": 7.})
        self.assertTrue(all(v == 0 for v in result["pairwise_pp"].values()))
        self.assertEqual(result["three_way_pp"], 0)

    def test_paper_configuration_is_free_of_machine_paths(self):
        text = (ROOT / "reproduction/paper_config.json").read_text()
        self.assertNotIn("/home/", text)
        self.assertNotIn("/Users/", text)
        self.assertEqual(len(load_config()["architectures"]), 4)


if __name__ == "__main__":
    unittest.main()
