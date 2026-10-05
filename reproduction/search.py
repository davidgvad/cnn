"""Select focal settings and score coefficients from KDDTrain+ OOF scores."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from paper_config import ARCHITECTURES, REGIMES, load_config

BETAS = (0.99, 0.999, 0.9999)
GAMMAS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
COEFFICIENTS = (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.15,
                1.30, 1.45, 1.60, 1.75, 1.90, 2.20, 2.50, 3.00,
                3.50, 4.00, 4.50, 5.00, 6.00, 7.00, 8.00, 10.00)
LABELS = dict(zip(ARCHITECTURES, ("Conv2D", "Conv1D", "Transformer", "MLP")))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def recorded_path(value, recording):
    path = Path(value).expanduser()
    candidates = [recording.parent / path.name, path] if path.is_absolute() else [recording.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing search artifact: {value}")


def save_config(path, config):
    """Keep a completed selection unchanged when resuming a run."""
    path = Path(path)
    if path.exists():
        if read_json(path) != config:
            raise ValueError(f"Selection changed. Use a new output directory: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)


def focal_selection(config, results):
    chosen = copy.deepcopy(config)
    evidence = {}
    training = config["training"]
    expected_pairs = {(b, g) for b in BETAS for g in GAMMAS}
    for architecture in ARCHITECTURES:
        pointer_path = results / f"{architecture}_focal_stage1_latest.json"
        pointer = read_json(pointer_path)
        protocol_path = recorded_path(pointer["protocol"], pointer_path)
        best_path = recorded_path(pointer["best_config"], pointer_path)
        summary_path = recorded_path(pointer["summary"], pointer_path)
        protocol, best = read_json(protocol_path), read_json(best_path)
        settings = protocol["settings"]
        expected = {"model": LABELS[architecture], "betas": list(BETAS),
                    "focal_gammas": list(GAMMAS), "training_seeds": training["seeds"],
                    "fold_count": training["fold_count"], "fold_seed": training["fold_seed"],
                    "epochs": training["epochs"], "batch_size": training["batch_size"],
                    "batching": "ordinary_shuffled", "validation_used_during_training": False,
                    "deterministic_ops": training["deterministic_ops"]}
        for name, value in expected.items():
            if settings.get(name) != value:
                raise ValueError(f"Focal search setting mismatch: {architecture}/{name}")
        fits = len(expected_pairs) * len(training["seeds"]) * training["fold_count"]
        if (protocol.get("expected_configurations"), protocol.get("expected_fits")) != (18, fits):
            raise ValueError(f"Incomplete focal search budget: {architecture}")
        if protocol.get("kddtest_accessed") is not False or best.get("kddtest_accessed") is not False:
            raise ValueError(f"Focal choice must exclude KDDTest+: {architecture}")
        if protocol.get("kddtrain_sha256") != config["datasets"]["KDDTrain+.txt"]["sha256"]:
            raise ValueError(f"Focal search dataset mismatch: {architecture}")
        with summary_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 18 or {(float(r["beta"]), float(r["focal_gamma"])) for r in rows} != expected_pairs:
            raise ValueError(f"Focal ranking must contain all 18 pairs: {architecture}")
        if any(int(r["total_fits"]) != fits // 18 for r in rows):
            raise ValueError(f"Incomplete focal seed/fold coverage: {architecture}")
        ranked = sorted(rows, key=lambda r: (-float(r["rare_f1_mean"]),
                        -float(r["macro_f1_mean"]), float(r["focal_gamma"]),
                        float(r["beta"]), r["config_key"]))
        winner = ranked[0]
        if best["config_id"] != winner["config_id"] or (best["beta"], best["focal_gamma"]) != (
                float(winner["beta"]), float(winner["focal_gamma"])):
            raise ValueError(f"Focal choice disagrees with OOF ranking: {architecture}")
        oof_dir = recorded_path(pointer["oof_directory"], pointer_path)
        paths = {str(seed): oof_dir / f"{best['config_id']}_s{seed}_oof_predictions.npz"
                 for seed in training["seeds"]}
        evidence[architecture] = {"protocol": str(protocol_path), "protocol_sha256": digest(protocol_path),
            "best_config": str(best_path), "best_config_sha256": digest(best_path),
            "summary": str(summary_path), "summary_sha256": digest(summary_path),
            "oof_files": {seed: {"path": str(path), "sha256": digest(path)} for seed, path in paths.items()}}
        chosen["architectures"][architecture].update(focal_beta=best["beta"], focal_gamma=best["focal_gamma"])
    chosen["parameter_selection_source"] = {"parameters_fixed": False, "selection_search_repeated": True,
        "selection_partition": "KDDTrain+ pooled four-fold OOF only", "kddtest_accessed": False,
        "focal_searches": evidence, "score_scaling_selected": False}
    return chosen


def can_reuse_focal(config, results):
    """Reuse a search winner only when its verified files are in this run."""
    source = config["parameter_selection_source"]
    if not source.get("selection_search_repeated") or not source.get("focal_searches"):
        return False
    results = Path(results).resolve()
    for architecture in ARCHITECTURES:
        pointer_path = results / f"{architecture}_focal_stage1_latest.json"
        if not pointer_path.exists():
            return False
        try:
            protocol_path = recorded_path(read_json(pointer_path)["protocol"], pointer_path)
        except FileNotFoundError:
            return False
        if read_json(protocol_path).get("expected_configurations") != 18:
            return False
    try:
        current = focal_selection(config, results)
    except FileNotFoundError:
        return False
    for architecture, evidence in current["parameter_selection_source"]["focal_searches"].items():
        paths = [evidence[name] for name in ("protocol", "best_config", "summary")]
        paths += [record["path"] for record in evidence["oof_files"].values()]
        if any(not Path(path).resolve().is_relative_to(results) for path in paths):
            return False
        expected = source["focal_searches"][architecture]
        hashes = ("protocol_sha256", "best_config_sha256", "summary_sha256")
        if (any(evidence[name] != expected[name] for name in hashes)
                or {seed: record["sha256"] for seed, record in evidence["oof_files"].items()}
                != {seed: record["sha256"] for seed, record in expected["oof_files"].items()}
                or any(current["architectures"][architecture][name] != config["architectures"][architecture][name]
                       for name in ("focal_beta", "focal_gamma"))):
            raise ValueError(f"Selected focal search changed: {architecture}. Use a new output directory.")
    return True


def scaling_selection(config, results):
    pointer_path = results / "variant_specific_scaling_selection_latest.json"
    pointer = read_json(pointer_path)
    manifest_path = recorded_path(pointer["selection_manifest"], pointer_path)
    if digest(manifest_path) != pointer["selection_manifest_sha256"]:
        raise ValueError("Score-selection manifest hash mismatch.")
    manifest = read_json(manifest_path)
    if manifest.get("kddtest_accessed") is not False:
        raise ValueError("Score selection must exclude KDDTest+.")
    if (set(manifest["architectures"]) != set(ARCHITECTURES)
            or manifest["seeds"] != config["training"]["seeds"]
            or manifest["coefficient_values"] != list(COEFFICIENTS)):
        raise ValueError("Score-search grid, architectures or seeds differ from the paper protocol.")
    rows = manifest["selected_coefficients"]
    expected = {(a, r) for a in ARCHITECTURES for r in REGIMES}
    if len(rows) != 16 or {(r["architecture"], r["base_training"]) for r in rows} != expected:
        raise ValueError("Expected one score pair for every architecture and training regime.")
    chosen = copy.deepcopy(config)
    focal_evidence = config["parameter_selection_source"]["focal_searches"]
    for architecture in ARCHITECTURES:
        for regime in REGIMES:
            source = manifest["source_metadata"][architecture][regime]
            protocol_path = Path(source["protocol"])
            if digest(protocol_path) != source["protocol_sha256"]:
                raise ValueError(f"OOF protocol changed: {architecture}/{regime}")
            protocol = read_json(protocol_path)
            if protocol.get("kddtest_accessed") is not False or protocol.get("kddtrain_sha256") != config["datasets"]["KDDTrain+.txt"]["sha256"]:
                raise ValueError(f"OOF selection dataset mismatch: {architecture}/{regime}")
            if set(source["oof_files"]) != set(map(str, config["training"]["seeds"])):
                raise ValueError(f"OOF selection seed coverage mismatch: {architecture}/{regime}")
            for record in source["oof_files"].values():
                if digest(record["path"]) != record["sha256"]:
                    raise ValueError(f"OOF predictions changed: {architecture}/{regime}")
            if regime == "focal_only":
                if source["protocol_sha256"] != focal_evidence[architecture]["protocol_sha256"]:
                    raise ValueError(f"Score search used another focal protocol: {architecture}")
                continue
            settings = protocol["training_settings"]
            training = config["training"]
            expected = {"training_seeds": training["seeds"], "fold_count": training["fold_count"],
                "fold_seed": training["fold_seed"], "epochs": training["epochs"],
                "batch_size": training["batch_size"], "validation_used_during_training": False,
                "deterministic_ops": training["deterministic_ops"],
                "minority_per_batch": training["minority_per_batch_per_class"] if regime != "baseline" else 0}
            if regime == "focal_batch":
                expected.update(cb_beta=config["architectures"][architecture]["focal_beta"],
                                focal_gamma=config["architectures"][architecture]["focal_gamma"])
            for name, value in expected.items():
                if settings.get(name) != value:
                    raise ValueError(f"Score-search OOF setting mismatch: {architecture}/{regime}/{name}")
        source = manifest["source_metadata"][architecture]["focal_only"]
        if source["best_config_sha256"] != focal_evidence[architecture]["best_config_sha256"]:
            raise ValueError(f"Score search used another focal choice: {architecture}")
        if source["oof_files"] != focal_evidence[architecture]["oof_files"]:
            raise ValueError(f"Score search used other focal predictions: {architecture}")
    for row in rows:
        r2l, u2r = row["r2l_score_coefficient"], row["u2r_score_coefficient"]
        if r2l not in COEFFICIENTS or u2r not in COEFFICIENTS:
            raise ValueError("Selected coefficient is outside the declared grid.")
        chosen["architectures"][row["architecture"]]["score_scaling"][row["base_training"]] = {"r2l": r2l, "u2r": u2r}
    chosen["parameter_selection_source"].update(parameters_fixed=True, score_scaling_selected=True,
        selection_id=manifest["selection_id"], selection_manifest=str(manifest_path),
        selection_manifest_sha256=digest(manifest_path))
    return chosen


def run_search(args, config, data_dir, output, execution, build_commands, repo, *, manifest_path=None):
    results = output / "results"
    focal_path, selected_path = output / "focal_config.json", output / "selected_config.json"
    stages = ("search-focal", "search-oof", "search-scaling") if args.stage == "search-all" else (args.stage,)
    env = os.environ.copy()
    if execution.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for stage in stages:
        if stage == "search-focal":
            commands, _ = build_commands(config, data_dir, results, execution)
            commands = [command for command in commands if "--betas" in command]
            for command in commands:
                start, end = command.index("--betas"), command.index("--fold-seed")
                command[start:end] = ["--betas", *map(str, BETAS), "--focal-gammas", *map(str, GAMMAS)]
        elif stage == "search-oof":
            if args.dry_run and not focal_path.exists():
                chosen = config
                print("OOF plan uses reference focal values until the search chooses new values.", flush=True)
            else:
                chosen = focal_selection(config, results)
                save_config(focal_path, chosen)
            commands, _ = build_commands(chosen, data_dir, results, execution, include_focal=False)
        else:
            if not args.dry_run:
                chosen = focal_selection(config, results)
                if not focal_path.exists() or load_config(focal_path) != chosen:
                    raise ValueError("Focal search records changed. Resume with the original run directory.")
            commands = [[sys.executable, "-u", str(repo / "src/tune_variant_specific_score_scaling.py"),
                "--results-dir", str(results), "--data-dir", str(data_dir), "select", "--seeds",
                *map(str, config["training"]["seeds"]), "--coefficient-values", *map(str, COEFFICIENTS)]]
        for command in commands:
            print(shlex.join(command), flush=True)
            actual_command = command + (["--dry-run"] if args.dry_run and stage != "search-scaling" else [])
            if manifest_path is not None:
                manifest = read_json(manifest_path)
                manifest["commands"].append({"stage": stage, "command": actual_command,
                    "executed": not (args.dry_run and stage == "search-scaling")})
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            if args.dry_run and stage == "search-scaling":
                continue
            subprocess.run(actual_command, cwd=repo, env=env, check=True)
        if args.dry_run:
            continue
        if stage == "search-focal":
            save_config(focal_path, focal_selection(config, results))
        elif stage == "search-scaling":
            save_config(selected_path, scaling_selection(chosen, results))
            print(f"Frozen choices: {selected_path}. Use this --config for final fitting and tables.", flush=True)
