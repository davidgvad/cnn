from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
import run_final_baseline_vs_full_kddtest_4gpu as final
import run_no_ctgan_model_ablation_4gpu as core
import tune_variant_specific_score_scaling as scaling


def selection_fixture(root):
    data = root / "data"
    results = root / "results"
    data.mkdir()
    results.mkdir()
    (data / "KDDTrain+.txt").write_text("training fixture\n")
    train_hash = core.sha256_file(data / "KDDTrain+.txt")
    labels = np.tile(np.arange(5, dtype=np.int64), 4)
    probabilities = np.eye(5, dtype=np.float32)[labels]
    sources = {}
    rows = []
    for arch in scaling.ARCHITECTURES:
        sources[arch] = {}
        for regime in scaling.BASE_TRAINING_ORDER:
            stage = scaling.BASE_TRAINING[regime]
            training = {
                "model": scaling.ARCHITECTURE_LABELS[arch],
                "backbone": final.FROZEN_CONFIG[arch]["backbone"],
                "training_seeds": [0, 1, 2], "fold_count": 4, "fold_seed": 0,
                "epochs": 7, "batch_size": 64, "deterministic_ops": False,
                "batching": "minority_guaranteed_with_replacement" if stage["batching"] else "ordinary_shuffled",
                "validation_used_during_training": False, "ctgan": False,
            }
            protocol = {"kddtest_accessed": False, "kddtrain_sha256": train_hash}
            best_path = None
            if regime == "focal_only":
                training.update(betas=[0.99, 0.999], focal_gammas=[0.25, 0.75])
                protocol["settings"] = training
                best_path = results / f"{arch}_best.json"
                core.atomic_json(best_path, {
                    "beta": 0.999, "focal_gamma": 0.75, "config_id": "winner",
                    "training_seeds": [0, 1, 2], "fold_count": 4,
                    "kddtest_accessed": False,
                })
            else:
                training.update(
                    architecture=arch, training_mode=scaling.SHARED_TRAINING_MODE[regime],
                    cb_beta=0.999 if stage["focal"] else None,
                    focal_gamma=0.75 if stage["focal"] else None,
                    minority_per_batch=2 if stage["batching"] else 0,
                )
                protocol["training_settings"] = training
            protocol_path = results / f"{arch}_{regime}_protocol.json"
            core.atomic_json(protocol_path, protocol)
            oof = results / f"{arch}_{regime}_oof"
            oof.mkdir()
            files = {}
            for seed in (0, 1, 2):
                path = oof / f"seed_{seed}.npz"
                core.atomic_npz(path, labels=labels, probabilities=probabilities)
                files[str(seed)] = {"path": str(path), "sha256": core.sha256_file(path)}
            pointer_name = stage["latest_names"][0].format(architecture=arch)
            sources[arch][regime] = {
                "protocol": str(protocol_path), "protocol_sha256": core.sha256_file(protocol_path),
                "kddtrain_sha256": train_hash, "latest_pointer": str(results / pointer_name),
                "oof_directory": str(oof), "oof_files": files,
                "best_config": str(best_path) if best_path else None,
                "best_config_sha256": core.sha256_file(best_path) if best_path else None,
                "config_id": "winner" if best_path else None,
            }
            rows.append({
                "architecture": arch, "base_training": regime,
                "r2l_score_coefficient": 0.7, "u2r_score_coefficient": 2.2,
            })
    manifest = {
        "selection_id": "native_fixture", "kddtest_accessed": False,
        "architectures": list(scaling.ARCHITECTURES), "seeds": [0, 1, 2],
        "coefficient_values": [0.7, 1.0, 2.2],
        "base_training_regimes": list(scaling.BASE_TRAINING_ORDER),
        "source_metadata": sources, "selected_coefficients": rows,
    }
    path = results / "selection.json"
    core.atomic_json(path, manifest)
    core.atomic_json(results / "variant_specific_scaling_selection_latest.json", {
        "selection_manifest": str(path), "selection_manifest_sha256": core.sha256_file(path),
    })
    args = argparse.Namespace(
        selection=str(path), results_dir=results, data_dir=data,
        seeds=[0, 1, 2], architectures=list(scaling.ARCHITECTURES),
        epochs=7, batch_size=64, minority_per_batch=2, deterministic_ops=False,
    )
    return args, path, manifest


class FinalExperimentTests(unittest.TestCase):
    def test_native_handoff_uses_actual_choices_without_test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            args, path, manifest = selection_fixture(Path(directory))
            selected_path, _, settings = final.load_frozen_selection(args, REPO_ROOT)
            self.assertEqual(selected_path, path.resolve())
            self.assertFalse((args.data_dir / "KDDTest+.txt").exists())
            for arch in scaling.ARCHITECTURES:
                self.assertEqual(settings[arch]["beta"], 0.999)
                self.assertEqual(settings[arch]["focal_gamma"], 0.75)
                self.assertEqual(settings[arch]["u2r_score_coefficient"], 2.2)
                self.assertEqual(settings[arch]["backbone"], final.FROZEN_CONFIG[arch]["backbone"])

    def test_selection_rejects_stale_protocol_winner_and_oof_files(self):
        for kind in ("protocol", "winner", "oof"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                args, path, manifest = selection_fixture(Path(directory))
                metadata = manifest["source_metadata"]["mlp"]["focal_only"]
                target = metadata["protocol"] if kind == "protocol" else metadata["best_config"]
                if kind == "oof":
                    target = metadata["oof_files"]["0"]["path"]
                with Path(target).open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    final.load_frozen_selection(args, REPO_ROOT)

    def test_selection_rejects_incomplete_invalid_or_incoherent_choices(self):
        for kind in ("missing", "duplicate", "coefficient", "focal", "quota", "test", "dataset"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                args, path, manifest = selection_fixture(Path(directory))
                if kind == "missing":
                    manifest["selected_coefficients"].pop()
                elif kind == "duplicate":
                    manifest["selected_coefficients"][-1] = manifest["selected_coefficients"][0]
                elif kind == "coefficient":
                    manifest["selected_coefficients"][0]["u2r_score_coefficient"] = 0
                elif kind == "test":
                    manifest["kddtest_accessed"] = True
                elif kind == "dataset":
                    (args.data_dir / "KDDTrain+.txt").write_text("different dataset\n")
                else:
                    metadata = manifest["source_metadata"]["mlp"]["focal_batch"]
                    protocol_path = Path(metadata["protocol"])
                    protocol = core.read_json(protocol_path)
                    key = "focal_gamma" if kind == "focal" else "minority_per_batch"
                    protocol["training_settings"][key] = 0.5 if kind == "focal" else 1
                    core.atomic_json(protocol_path, protocol)
                    metadata["protocol_sha256"] = core.sha256_file(protocol_path)
                core.atomic_json(path, manifest)
                with self.assertRaises(ValueError):
                    final.load_frozen_selection(args, REPO_ROOT)

    def test_final_budget_must_match_selected_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = selection_fixture(Path(directory))
            for key, value in (("epochs", 25), ("batch_size", 256), ("minority_per_batch", 1), ("seeds", [0, 1])):
                changed = copy.copy(args)
                setattr(changed, key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    final.load_frozen_selection(changed, REPO_ROOT)

    def test_final_record_rejects_different_focal_choice(self):
        chosen = {"beta": 0.999, "focal_gamma": 0.75, "epochs": 7, "batch_size": 64,
                  "minority_per_batch": 2, "deterministic_ops": False,
                  "backbone": final.FROZEN_CONFIG["mlp"]["backbone"]}
        run = {"cb_beta": 0.999, "focal_gamma": 0.75, "epochs_requested": 7, "batch_size": 64,
               "batching": "ordinary_shuffled", "minority_per_batch_per_class": 0,
               "validation_used_during_training": False, "ctgan_used": False,
               "deterministic_ops_requested": False,
               "backbone": chosen["backbone"], "model_parameters": 99845, "epochs_completed": 7}
        scaling.validate_final_training(run, chosen, "focal_only")
        run["focal_gamma"] = 0.25
        with self.assertRaisesRegex(ValueError, "focal_gamma"):
            scaling.validate_final_training(run, chosen, "focal_only")

    def test_worker_reads_frozen_protocol_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            args, _, _ = selection_fixture(Path(directory))
            _, _, settings = final.load_frozen_selection(args, REPO_ROOT)
            path = Path(directory) / "final_protocol.json"
            core.atomic_json(path, {"experiment_key": "frozen", "frozen_config": settings})
            args.worker_protocol_path = str(path)
            args.worker_protocol_sha256 = core.sha256_file(path)
            args.experiment_key = "frozen"
            args.worker_architecture = "mlp"
            self.assertEqual(final.load_worker_settings(args)["focal_gamma"], 0.75)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                final.load_worker_settings(args)

    def test_one_and_two_gpu_dry_plans_cover_48_fits(self):
        with tempfile.TemporaryDirectory() as directory:
            args, path, _ = selection_fixture(Path(directory))
            (args.data_dir / "KDDTest+.txt").write_text("test fixture\n")
            command = [sys.executable, str(REPO_ROOT / "src/run_final_baseline_vs_full_kddtest_4gpu.py"),
                       "--selection", str(path), "--results-dir", str(args.results_dir),
                       "--data-dir", str(args.data_dir), "--epochs", "7", "--batch-size", "64",
                       "--minority-per-batch", "2", "--dry-run"]
            for runtime in (["--gpus", "0"], ["--device", "gpu", "--gpus", "0", "1"]):
                result = subprocess.run(command + runtime, capture_output=True, text=True,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Fits: 48 (4 architectures x 4 configurations x 3 seeds)", result.stdout)

    def test_final_command_rejects_cpu_auto_and_allow_cpu(self):
        command = [sys.executable, str(REPO_ROOT / "src/run_final_baseline_vs_full_kddtest_4gpu.py")]
        for runtime in (["--device", "cpu"], ["--device", "auto"], ["--allow-cpu"]):
            with self.subTest(runtime=runtime):
                result = subprocess.run(command + runtime, capture_output=True, text=True,
                                        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, timeout=60)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("error:", result.stderr)


if __name__ == "__main__":
    unittest.main()
