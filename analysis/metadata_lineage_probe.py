"""Reproduce the final evaluator's semantic metadata gap without ML dependencies.

This extracts and runs only load_test_run_metadata from the reviewed source.
The prediction placeholder is NOT a probability artifact; probability validation
is intentionally outside the scope of this metadata-only probe.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path = ROOT / "src/tune_variant_specific_score_scaling.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_test_run_metadata")
    isolated = ast.Module(body=[function], type_ignores=[])
    scope = {"Path": Path, "Dict": Dict, "Any": Any,
             "core": SimpleNamespace(read_json=lambda p: json.loads(Path(p).read_text()), sha256_file=sha256_file)}
    exec(compile(isolated, str(path), "exec"), scope)
    with tempfile.TemporaryDirectory(prefix="cnn_metadata_probe_") as directory:
        root = Path(directory)
        (root / "data").mkdir()
        (root / "data/KDDTrain+.txt").write_text("training fixture\n")
        (root / "data/KDDTest+.txt").write_text("test fixture\n")
        results = root / "results"
        pred_dir = results / "probe_predictions"
        run_dir = results / "probe_runs"
        pred_dir.mkdir(parents=True)
        run_dir.mkdir()
        prediction = pred_dir / "model.npz"
        prediction.write_bytes(b"metadata-only placeholder; not valid NPZ")
        incompatible = {"loss": "categorical_crossentropy", "cb_beta": 0.9,
                        "focal_gamma": 99.0, "epochs_requested": 1,
                        "batch_size": 8, "ctgan_used": True,
                        "model_parameters": 1, "batching": "wrong_policy"}
        run = {"architecture": "conv2d", "variant": "focal_only", "seed": 0,
               "kddtest_used_for_selection": False,
               "prediction_sha256": sha256_file(prediction),
               "feature_cache_sha256": "linked-cache", **incompatible}
        (run_dir / "model.json").write_text(json.dumps(run))
        cache = {"cache_sha256": "linked-cache",
                 "train_sha256": sha256_file(root / "data/KDDTrain+.txt"),
                 "test_sha256": sha256_file(root / "data/KDDTest+.txt")}
        (results / "probe_feature_cache.json").write_text(json.dumps(cache))
        scope["load_test_run_metadata"](prediction, root, "conv2d", "focal_only", 0)
    evidence = {"metadata_check_accepted_incompatible_configuration": True,
                "incompatible_fields": incompatible,
                "scope": "metadata check only; array checker not invoked; no real experimental run asserted invalid",
                "source": str(path.relative_to(ROOT)),
                "source_sha256": sha256_file(path)}
    (ROOT / "analysis/metadata_lineage_probe.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
