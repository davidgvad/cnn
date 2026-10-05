from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
import search
import run_final_baseline_vs_full_kddtest_4gpu as final_runner


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

    def make_search_records(self, directory):
        config = load_config()
        for architecture in config["architectures"]:
            stem = directory / f"{architecture}_focal_stage1_test"
            protocol = {"settings": {"model": search.LABELS[architecture],
                "betas": list(search.BETAS), "focal_gammas": list(search.GAMMAS),
                "training_seeds": [0, 1, 2], "fold_count": 4, "fold_seed": 0,
                "epochs": 25, "batch_size": 256, "batching": "ordinary_shuffled",
                "validation_used_during_training": False, "deterministic_ops": False},
                "expected_configurations": 18, "expected_fits": 216, "kddtest_accessed": False,
                "kddtrain_sha256": config["datasets"]["KDDTrain+.txt"]["sha256"]}
            rows = []
            for index, (beta, gamma) in enumerate((b, g) for b in search.BETAS for g in search.GAMMAS):
                rows.append({"beta": beta, "focal_gamma": gamma, "config_key": str(index),
                    "config_id": f"pair_{index}", "rare_f1_mean": 1 - index * .01,
                    "macro_f1_mean": .8, "total_fits": 12})
            best = {"beta": .99, "focal_gamma": .25, "config_id": "pair_0", "kddtest_accessed": False}
            protocol_path, best_path = Path(str(stem) + "_protocol.json"), Path(str(stem) + "_best_config.json")
            summary_path = Path(str(stem) + "_summary.csv")
            protocol_path.write_text(json.dumps(protocol))
            best_path.write_text(json.dumps(best))
            with summary_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            oof = Path(str(stem) + "_oof")
            oof.mkdir()
            for seed in (0, 1, 2):
                (oof / f"pair_0_s{seed}_oof_predictions.npz").write_bytes(f"OOF {seed}".encode())
            pointer = {"protocol": str(protocol_path), "best_config": str(best_path),
                       "summary": str(summary_path), "oof_directory": str(oof)}
            (directory / f"{architecture}_focal_stage1_latest.json").write_text(json.dumps(pointer))
        return config

    def test_focal_handoff_uses_ranked_winners_and_preserves_input_config(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = self.make_search_records(directory)
            original = copy.deepcopy(config)
            chosen = search.focal_selection(config, directory)
            self.assertEqual(config, original)
            self.assertEqual(chosen["architectures"]["conv2d"]["focal_gamma"], .25)
            self.assertFalse(chosen["parameter_selection_source"]["parameters_fixed"])
            oof, _ = entry.build_commands(chosen, directory, directory, experiment_runtime.Runtime(
                "cpu", ("cpu0",), {"cpu0": ""}, False), include_focal=False)
            self.assertEqual(len(oof), 12)
            self.assertFalse(any("--betas" in command for command in oof))
            self.assertEqual(sum("--cb-beta" in command for command in oof), 4)

    def test_focal_handoff_rejects_missing_candidate_and_false_winner(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = self.make_search_records(directory)
            best_path = directory / "mlp_focal_stage1_test_best_config.json"
            best = json.loads(best_path.read_text())
            best["focal_gamma"] = .5
            best_path.write_text(json.dumps(best))
            with self.assertRaisesRegex(ValueError, "disagrees with OOF ranking"):
                search.focal_selection(config, directory)
            best["focal_gamma"] = .25
            best_path.write_text(json.dumps(best))
            summary = directory / "mlp_focal_stage1_test_summary.csv"
            summary.write_text("\n".join(summary.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "all 18 pairs"):
                search.focal_selection(config, directory)

    def test_focal_reuse_requires_local_unchanged_search_predictions(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            chosen = search.focal_selection(self.make_search_records(directory), directory)
            self.assertTrue(search.can_reuse_focal(chosen, directory))
            self.assertFalse(search.can_reuse_focal(chosen, directory / "fresh_results"))
            execution = experiment_runtime.Runtime("cpu", ("cpu0",), {"cpu0": ""}, False)
            oof, _ = entry.build_commands(chosen, directory, directory / "fresh_results", execution,
                include_focal=not search.can_reuse_focal(chosen, directory / "fresh_results"))
            self.assertEqual(len(oof), 16)
            prediction = directory / "mlp_focal_stage1_test_oof/pair_0_s0_oof_predictions.npz"
            prediction.write_bytes(b"changed predictions")
            with self.assertRaisesRegex(ValueError, "Selected focal search changed"):
                search.can_reuse_focal(chosen, directory)

    def test_focal_reuse_keeps_fixed_parameter_reruns_in_the_plan(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            chosen = search.focal_selection(self.make_search_records(directory), directory)
            protocol_path = directory / "mlp_focal_stage1_test_protocol.json"
            protocol = json.loads(protocol_path.read_text())
            protocol["expected_configurations"] = 1
            protocol_path.write_text(json.dumps(protocol))
            self.assertFalse(search.can_reuse_focal(chosen, directory))

    def test_selected_config_resume_rejects_changed_parameters(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "selected_config.json"
            config = load_config()
            search.save_config(path, config)
            search.save_config(path, config)
            changed = copy.deepcopy(config)
            changed["architectures"]["mlp"]["focal_gamma"] = .5
            with self.assertRaisesRegex(ValueError, "Selection changed"):
                search.save_config(path, changed)
            self.assertEqual(json.loads(path.read_text()), config)

    def test_scaling_handoff_requires_all_pairs_and_same_focal_predictions(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = search.focal_selection(self.make_search_records(directory), directory)
            manifest = {"kddtest_accessed": False, "architectures": list(config["architectures"]),
                "seeds": [0, 1, 2], "coefficient_values": list(search.COEFFICIENTS),
                "selection_id": "fresh_selection", "selected_coefficients": [], "source_metadata": {}}
            for architecture in config["architectures"]:
                evidence = config["parameter_selection_source"]["focal_searches"][architecture]
                manifest["source_metadata"][architecture] = {"focal_only": {
                    "protocol": evidence["protocol"], "protocol_sha256": evidence["protocol_sha256"],
                    "best_config_sha256": evidence["best_config_sha256"], "oof_files": evidence["oof_files"]}}
                for regime in ("baseline", "focal_only", "batch_only", "focal_batch"):
                    manifest["selected_coefficients"].append({"architecture": architecture,
                        "base_training": regime, "r2l_score_coefficient": 1., "u2r_score_coefficient": 2.5})
                    if regime != "focal_only":
                        protocol_path = directory / f"{architecture}_{regime}_protocol.json"
                        protocol_path.write_text(json.dumps({"kddtest_accessed": False,
                            "kddtrain_sha256": config["datasets"]["KDDTrain+.txt"]["sha256"],
                            "training_settings": {"training_seeds": [0, 1, 2], "fold_count": 4,
                                "fold_seed": 0, "epochs": 25, "batch_size": 256,
                                "validation_used_during_training": False, "deterministic_ops": False,
                                "minority_per_batch": 0 if regime == "baseline" else 1,
                                "cb_beta": .99, "focal_gamma": .25}}))
                        manifest["source_metadata"][architecture][regime] = {
                            "protocol": str(protocol_path), "protocol_sha256": search.digest(protocol_path),
                            "oof_files": evidence["oof_files"]}
            path = directory / "selection.json"
            pointer = directory / "variant_specific_scaling_selection_latest.json"
            def save_manifest():
                path.write_text(json.dumps(manifest))
                pointer.write_text(json.dumps({"selection_manifest": str(path),
                    "selection_manifest_sha256": search.digest(path)}))
            save_manifest()
            chosen = search.scaling_selection(config, directory)
            self.assertTrue(chosen["parameter_selection_source"]["parameters_fixed"])
            self.assertEqual(chosen["architectures"]["mlp"]["score_scaling"]["baseline"]["u2r"], 2.5)
            batch_protocol = directory / "mlp_focal_batch_protocol.json"
            batch_record = json.loads(batch_protocol.read_text())
            batch_record["training_settings"]["focal_gamma"] = .5
            batch_protocol.write_text(json.dumps(batch_record))
            manifest["source_metadata"]["mlp"]["focal_batch"]["protocol_sha256"] = search.digest(batch_protocol)
            save_manifest()
            with self.assertRaisesRegex(ValueError, "focal_batch/focal_gamma"):
                search.scaling_selection(config, directory)
            batch_record["training_settings"]["focal_gamma"] = .25
            batch_protocol.write_text(json.dumps(batch_record))
            manifest["source_metadata"]["mlp"]["focal_batch"]["protocol_sha256"] = search.digest(batch_protocol)
            manifest["source_metadata"]["mlp"]["focal_only"]["best_config_sha256"] = "another_choice"
            save_manifest()
            with self.assertRaisesRegex(ValueError, "another focal choice"):
                search.scaling_selection(config, directory)
            manifest["source_metadata"]["mlp"]["focal_only"]["best_config_sha256"] = config[
                "parameter_selection_source"]["focal_searches"]["mlp"]["best_config_sha256"]
            manifest["selected_coefficients"].pop()
            save_manifest()
            with self.assertRaisesRegex(ValueError, "one score pair"):
                search.scaling_selection(config, directory)

    def test_native_scaling_manifest_hands_off_to_config_without_test_data(self):
        import tune_variant_specific_score_scaling as selector
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            config = self.make_search_records(directory)
            data_dir = directory / "data"
            data_dir.mkdir()
            train = data_dir / "KDDTrain+.txt"
            train.write_text("synthetic training data for the artifact hash check\n")
            config["datasets"]["KDDTrain+.txt"]["sha256"] = search.digest(train)
            labels = np.tile(np.arange(5), 4)
            probabilities = np.eye(5, dtype=np.float32)[labels] * .8 + .04
            arrays = dict(labels=labels, probabilities=probabilities, raw_predictions=labels,
                          fold_ids=np.repeat(np.arange(4), 5), row_indices=np.arange(20))
            for architecture in config["architectures"]:
                focal_protocol = directory / f"{architecture}_focal_stage1_test_protocol.json"
                record = json.loads(focal_protocol.read_text())
                record["kddtrain_sha256"] = search.digest(train)
                focal_protocol.write_text(json.dumps(record))
                oof = directory / f"{architecture}_focal_stage1_test_oof"
                for seed in (0, 1, 2):
                    np.savez_compressed(oof / f"pair_0_s{seed}_oof_predictions.npz", **arrays)
                for regime, mode, suffix in (("baseline", "baseline_ce", "baseline_cv"),
                        ("batch_only", "baseline_batch", "batch_baseline_cv"),
                        ("focal_batch", "focal_balanced", "focal_batch_cv")):
                    regime_oof = directory / f"{architecture}_{regime}_oof"
                    regime_oof.mkdir()
                    for seed in (0, 1, 2):
                        np.savez_compressed(regime_oof / f"seed_{seed}_oof_probabilities.npz", **arrays)
                    protocol_path = directory / f"{architecture}_{regime}_protocol.json"
                    protocol_path.write_text(json.dumps({"kddtest_accessed": False,
                        "kddtrain_sha256": search.digest(train), "training_settings": {
                            "architecture": architecture, "model": search.LABELS[architecture], "training_mode": mode,
                            "training_seeds": [0, 1, 2], "fold_count": 4, "fold_seed": 0, "epochs": 25,
                            "batch_size": 256, "validation_used_during_training": False, "deterministic_ops": False,
                            "minority_per_batch": 0 if regime == "baseline" else 1, "cb_beta": .99, "focal_gamma": .25}}))
                    (directory / f"{architecture}_{suffix}_latest.json").write_text(json.dumps({
                        "protocol": str(protocol_path), "oof_directory": str(regime_oof)}))
            focal = search.focal_selection(config, directory)
            arguments = selector.build_parser().parse_args(["--results-dir", str(directory),
                "--data-dir", str(data_dir), "select", "--coefficient-values", *map(str, search.COEFFICIENTS)])
            with redirect_stdout(io.StringIO()):
                selector.select_coefficients(arguments)
            frozen = search.scaling_selection(focal, directory)
            self.assertTrue(frozen["parameter_selection_source"]["parameters_fixed"])
            self.assertFalse((data_dir / "KDDTest+.txt").exists())
            for settings in frozen["architectures"].values():
                for pair in settings["score_scaling"].values():
                    self.assertEqual(pair, {"r2l": 1., "u2r": 1.})

    def test_final_runner_uses_new_focal_choices_and_requires_frozen_config(self):
        config = load_config()
        config["architectures"]["mlp"].update(focal_beta=.999, focal_gamma=.5)
        chosen = final_runner.frozen_settings(config, "mlp")
        self.assertEqual((chosen["beta"], chosen["focal_gamma"]), (.999, .5))
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "unfrozen.json"
            config["parameter_selection_source"]["parameters_fixed"] = False
            path.write_text(json.dumps(config))
            with patch.object(sys, "argv", ["final", "--paper-config", str(path)]):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    final_runner.parse_arguments()


if __name__ == "__main__":
    unittest.main()
