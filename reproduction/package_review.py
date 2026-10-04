#!/usr/bin/env python3
"""Create a self-contained paper-code archive without local audit/server files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CORE = ["experiment_runtime.py", "paper_config.py", "paper_metrics.py", "export_paper_tables.py",
        "run_no_ctgan_model_ablation_4gpu.py", "cnn_opt.py", "cnn_opt_1d_4gpu.py", "cnn_gan_foc.py",
        "run_final_baseline_vs_full_kddtest_4gpu.py", "tune_variant_specific_score_scaling.py",
        "tune_conv2d_score_scaling_cv_4gpu.py"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/cnn_paper_reproduction.zip")
    parser.add_argument("--include-data", action="store_true")
    args = parser.parse_args()
    sources = [ROOT / "src" / name for name in CORE]
    for arch in ("conv2d", "conv1d", "transformer", "mlp"):
        sources += [ROOT / "src" / f"tune_{arch}_focal_cv_4gpu.py"]
        sources += [ROOT / "src" / f"run_{arch}_{regime}_cv_4gpu.py" for regime in ("baseline", "batch_baseline", "focal_batch")]
    sources += [ROOT / "README.md", ROOT / ".gitignore"]
    sources += [p for p in (ROOT / "reproduction").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    sources += [ROOT / "tests" / n for n in ("test_experiment_runtime.py", "test_paper_reproduction.py", "test_variant_specific_score_scaling.py")]
    if args.include_data:
        config = json.loads((ROOT / "reproduction/paper_config.json").read_text())
        for name, record in config["datasets"].items():
            path = ROOT / "data" / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Dataset differs from the checked paper input: {name}")
            sources.append(path)
    if (ROOT / "LICENSE").is_file():
        sources.append(ROOT / "LICENSE")
    sources = sorted(set(sources))
    manifest = {"schema_version": 1, "includes_data": args.include_data,
                "saved_predictions_included": False, "verified_reference_tables_included": True,
                "files": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Choose a new archive filename: {args.output}")
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            relative = path.relative_to(ROOT).as_posix()
            data = path.read_bytes()
            manifest["files"][relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            archive.writestr("cnn_paper_reproduction/" + relative, data)
        archive.writestr("cnn_paper_reproduction/MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.output) as archive:
        for name, record in manifest["files"].items():
            assert hashlib.sha256(archive.read("cnn_paper_reproduction/" + name)).hexdigest() == record["sha256"]
    print(f"Created {args.output}: {len(sources)} files, dataset included={args.include_data}")


if __name__ == "__main__":
    main()
