#!/usr/bin/env python3
"""Run the paper's parameter searches or its fixed-parameter experiments."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiment_runtime as runtime
from paper_config import DEFAULT_CONFIG, load_config


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_commands(config, data_dir, results_dir, execution, *, smoke=False, config_path=DEFAULT_CONFIG, include_focal=True):
    training = config["training"]
    seeds = [0] if smoke else training["seeds"]
    epochs = 1 if smoke else training["epochs"]
    common = ["--seeds", *map(str, seeds), "--epochs", str(epochs),
              "--batch-size", str(training["batch_size"]), "--data-dir", str(data_dir),
              "--results-dir", str(results_dir), "--device", execution.device]
    if execution.device == "gpu":
        common += ["--gpus", *execution.workers]
    else:
        common += ["--workers", str(len(execution.workers))]
    if training["deterministic_ops"]:
        common += ["--deterministic-ops"]
    oof = []
    for arch, settings in config["architectures"].items():
        if smoke and arch != "conv2d":
            continue
        beta, gamma = str(settings["focal_beta"]), str(settings["focal_gamma"])
        choices = [(f"tune_{arch}_focal_cv_4gpu.py", ["--betas", beta, "--focal-gammas", gamma])] if include_focal else []
        if not smoke:
            choices += [(f"run_{arch}_baseline_cv_4gpu.py", ["--coefficient-values", "1.0"]),
                        (f"run_{arch}_batch_baseline_cv_4gpu.py", ["--coefficient-values", "1.0", "--minority-per-batch", str(training["minority_per_batch_per_class"])])]
        choices += [(f"run_{arch}_focal_batch_cv_4gpu.py", ["--cb-beta", beta, "--focal-gamma", gamma,
                     "--minority-per-batch", str(training["minority_per_batch_per_class"]), "--coefficient-values", "1.0"])]
        for script, options in choices:
            oof.append([sys.executable, "-u", str(REPO / "src" / script), *options,
                        "--fold-seed", str(training["fold_seed"]), *common])
    variants = ["baseline", "full"] if smoke else ["baseline", "focal_only", "batch_only", "full"]
    final = [[sys.executable, "-u", str(REPO / "src/run_final_baseline_vs_full_kddtest_4gpu.py"),
              "--architectures", *config["architectures"], "--variants", *variants,
              "--paper-config", str(config_path),
              "--minority-per-batch", str(training["minority_per_batch_per_class"]), *common]]
    return oof, final


def make_smoke_data(data_dir, destination):
    from paper_metrics import TAXONOMY
    destination.mkdir(parents=True, exist_ok=True)
    for name, per_class in (("KDDTrain+.txt", 20), ("KDDTest+.txt", 5)):
        counts = [0] * 5
        selected = []
        with (data_dir / name).open(newline="") as handle:
            for row in csv.reader(handle):
                label = TAXONOMY[row[41]]
                if counts[label] < per_class:
                    selected.append(row)
                    counts[label] += 1
                if min(counts) == per_class:
                    break
        if min(counts) != per_class:
            raise ValueError(f"Insufficient examples for the smoke check: {name}")
        with (destination / name).open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerows(selected)


def check_smoke_results(results_dir, output, config):
    records = []
    for path in results_dir.glob("*_runs/*.json"):
        data = json.loads(path.read_text())
        if "model_parameters" not in data:
            continue
        if data["epochs_requested"] != 1 or data["epochs_completed"] != 1:
            raise ValueError(f"Incomplete smoke fit: {path}")
        arch = data.get("architecture", "conv2d")
        if data["model_parameters"] != config["architectures"][arch]["parameters"]:
            raise ValueError(f"Wrong model parameter count: {path}")
        records.append({"run": path.name, "parameters": data["model_parameters"],
                        "epochs": data["epochs_completed"], "device": data["runtime_device"]})
    if len(records) != 16:
        raise ValueError(f"Expected sixteen small smoke fits; found {len(records)}")
    report = {"smoke_check_passed": True, "paper_results_reproduced": False,
              "training_rows": 100, "test_rows": 25, "fits": records}
    (output / "smoke_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Smoke check passed: {len(records)} one-epoch fits. These are not paper results.", flush=True)


def compare_reference(output):
    with (REPO / "reproduction/reference_tables/summary_unrounded.csv").open(newline="") as f:
        reference = list(csv.DictReader(f))
    with (output / "paper_tables/summary_unrounded.csv").open(newline="") as f:
        observed = list(csv.DictReader(f))
    keys = ("partition", "architecture", "configuration")
    lookup = {tuple(r[k] for k in keys): r for r in observed}
    if len(lookup) != len(reference):
        raise ValueError("Reproduction and reference summaries cover different configurations.")
    rows = []
    for ref in reference:
        run = lookup[tuple(ref[k] for k in keys)]
        for metric, value in ref.items():
            if metric in keys:
                continue
            rows.append({**{k: ref[k] for k in keys}, "metric": metric,
                         "reference": float(value), "reproduction": float(run[metric]),
                         "difference": float(run[metric]) - float(value)})
    with (output / "reference_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} comparisons to {output / 'reference_comparison.csv'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "oof", "final", "tables", "compare", "all", "smoke",
                        "search-focal", "search-oof", "search-scaling", "search-all"), default="plan")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs/paper")
    parser.add_argument("--dry-run", action="store_true")
    runtime.add_arguments(parser, paths=False)
    args = parser.parse_args()
    args.dry_run = args.dry_run or args.stage == "plan"
    config = load_config(args.config)
    if args.stage in ("final", "all", "tables") and config.get("parameter_selection_source", {}).get("parameters_fixed") is not True:
        parser.error("Freeze focal and score-scaling parameters before final evaluation or table export.")
    data_dir, output = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if output == REPO or output == data_dir or output.is_relative_to(data_dir):
        parser.error("Choose an output directory separate from the source data.")
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "compare":
        compare_reference(output); return
    searching = args.stage.startswith("search-")
    datasets = {"KDDTrain+.txt": config["datasets"]["KDDTrain+.txt"]} if searching else config["datasets"]
    for name, record in datasets.items():
        if sha256(data_dir / name) != record["sha256"]:
            parser.error(f"Dataset SHA-256 differs from the audited paper input: {name}")
    config_copy = output / ("search_base_config.json" if searching else "paper_config.json")
    selected_path = output / "selected_config.json"
    if not searching and selected_path.exists() and load_config(selected_path) != config:
        parser.error(f"This directory has new selected parameters. Use --config {selected_path}.")
    if config_copy.exists() and load_config(config_copy) != config:
        parser.error("This output directory belongs to a different paper configuration; choose a new directory.")
    if not config_copy.exists():
        temporary = config_copy.with_name(config_copy.name + ".tmp")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(config_copy)
    results_dir = output / "results"
    if args.stage == "tables":
        subprocess.run([sys.executable, str(REPO / "src/export_paper_tables.py"), "--repo", str(REPO),
                        "--config", str(config_copy), "--data-dir", str(data_dir), "--results-dir", str(results_dir),
                        "--output-dir", str(output / "paper_tables")], check=True)
        return
    execution = runtime.configure_controller(args, parser)
    if searching:
        from search import run_search
        packages = {}
        for name in ("tensorflow", "keras", "numpy", "pandas", "scikit-learn"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "not-installed"
        manifest = {"stage": args.stage, "dry_run": args.dry_run, "runtime": execution.metadata(),
                    "python": sys.version, "platform": platform.platform(), "packages": packages,
                    "search_base_config_sha256": sha256(config_copy), "kddtest_accessed": False,
                    "launcher_sha256": {str(p.relative_to(REPO)): sha256(p) for p in
                                          (Path(__file__).resolve(), REPO / "reproduction/search.py")},
                    "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in (REPO / "src").glob("*.py")},
                    "datasets": {name: sha256(data_dir / name) for name in datasets}, "commands": []}
        manifest_path = output / f"invocation_{time.time_ns()}.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print("Full search budget: 864 focal fits + 144 other OOF fits. Final fitting adds 48 fits.", flush=True)
        run_search(args, config, data_dir, output, execution, build_commands, REPO, manifest_path=manifest_path)
        return
    smoke = args.stage == "smoke"
    if smoke:
        data_dir = output / "smoke_data"
        make_smoke_data(args.data_dir.expanduser().resolve(), data_dir)
        results_dir = output / "smoke_results"
    from search import can_reuse_focal
    reuse_focal = not smoke and can_reuse_focal(config, results_dir)
    oof, final = build_commands(config, data_dir, results_dir, execution, smoke=smoke, config_path=config_copy,
                                include_focal=not reuse_focal)
    commands = []
    if args.stage in ("oof", "all", "plan", "smoke"):
        commands += oof
    if args.stage in ("final", "all", "plan", "smoke"):
        commands += final
    if args.dry_run:
        commands = [c + ["--dry-run"] for c in commands]
    packages = {}
    for name in ("tensorflow", "keras", "numpy", "pandas", "scikit-learn", "matplotlib", "seaborn"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "not-installed"
    manifest = {"stage": args.stage, "dry_run": args.dry_run, "smoke": smoke, "runtime": execution.metadata(),
                "python": sys.version, "platform": platform.platform(), "packages": packages,
                "paper_config_sha256": sha256(config_copy), "commands": commands,
                "source_sha256": {str(p.relative_to(REPO)): sha256(p) for p in (REPO / "src").glob("*.py")},
                "datasets": {name: sha256(data_dir / name) for name in config["datasets"]}}
    manifest_path = output / f"invocation_{time.time_ns()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Runtime: {execution.metadata()}; invocation: {manifest_path}", flush=True)
    budget = "144 OOF fits + 48 final fits, reusing the selected focal OOF scores" if reuse_focal else "192 OOF fits + 48 final fits"
    print("Planned fits: 16 small smoke fits" if smoke else f"Paper budget: {budget}. Parameters are fixed.", flush=True)
    env = os.environ.copy()
    if execution.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for command in commands:
        subprocess.run(command, cwd=REPO, env=env, check=True)
    if smoke and not args.dry_run:
        check_smoke_results(results_dir, output, config)
    if args.stage == "all" and not args.dry_run:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", "tables", "--config", str(config_copy),
                        "--data-dir", str(data_dir), "--output-dir", str(output)], check=True)
        compare_reference(output)


if __name__ == "__main__":
    main()
