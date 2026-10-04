"""Independently recompute paper results from frozen server prediction files.

Requires NumPy; does not import project modules, TensorFlow, or train models.
Reads explicit frozen manifests rather than mutable latest pointers. Original
results are read only. The output directory must be new.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ARCHITECTURES = ("conv2d", "conv1d", "transformer", "mlp")
REGIMES = ("baseline", "focal_only", "batch_only", "focal_batch")
CONFIGS = {
    "Baseline": ("baseline", False, (0, 0, 0)),
    "Focal only": ("focal_only", False, (1, 0, 0)),
    "Batching only": ("batch_only", False, (0, 1, 0)),
    "Scaling only": ("baseline", True, (0, 0, 1)),
    "Focal + batching": ("focal_batch", False, (1, 1, 0)),
    "Focal + scaling": ("focal_only", True, (1, 0, 1)),
    "Batching + scaling": ("batch_only", True, (0, 1, 1)),
    "All three controls": ("focal_batch", True, (1, 1, 1)),
}
STORED_CONFIGS = {
    "Baseline": "baseline", "Focal only": "focal_only",
    "Batching only": "batch_only", "Scaling only": "scaling_only_tuned",
    "Focal + batching": "focal_batch", "Focal + scaling": "focal_scaling_tuned",
    "Batching + scaling": "batch_scaling_tuned", "All three controls": "full_retuned",
}
METRICS = (
    "accuracy", "mcc", "macro_f1", "macro_recall", "rare_f1",
    "minimum_minority_recall", "r2l_precision", "r2l_recall", "r2l_f1",
    "u2r_precision", "u2r_recall", "u2r_f1",
)
TAXONOMY = {
    "neptune": 0, "smurf": 0, "back": 0, "teardrop": 0, "pod": 0, "land": 0,
    "apache2": 0, "processtable": 0, "mailbomb": 0, "udpstorm": 0,
    "satan": 1, "ipsweep": 1, "portsweep": 1, "nmap": 1, "mscan": 1, "saint": 1,
    "warezclient": 2, "guess_passwd": 2, "warezmaster": 2, "imap": 2,
    "ftp_write": 2, "multihop": 2, "phf": 2, "spy": 2, "snmpguess": 2,
    "snmpgetattack": 2, "httptunnel": 2, "named": 2, "sendmail": 2,
    "xlock": 2, "xsnoop": 2, "worm": 2,
    "buffer_overflow": 3, "rootkit": 3, "loadmodule": 3, "perl": 3,
    "ps": 3, "xterm": 3, "sqlattack": 3, "normal": 4,
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_from_confusion(confusion):
    """Five-class metrics calculated directly from counts, independently of src/."""
    c = np.asarray(confusion, dtype=np.float64)
    support, predicted, tp = c.sum(axis=1), c.sum(axis=0), c.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(5), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(5), where=(support + predicted) != 0)
    n = float(c.sum())
    denom = (n * n - float(predicted @ predicted)) * (n * n - float(support @ support))
    mcc = (float(tp.sum()) * n - float(support @ predicted)) / math.sqrt(denom) if denom > 0 else 0.0
    return {
        "accuracy": float(tp.sum() / n) if n else 0.0, "mcc": mcc,
        "macro_f1": float(f1.mean()), "macro_recall": float(recall.mean()),
        "rare_f1": float(f1[2:4].mean()), "minimum_minority_recall": float(recall[2:4].min()),
        "r2l_precision": float(precision[2]), "r2l_recall": float(recall[2]), "r2l_f1": float(f1[2]),
        "u2r_precision": float(precision[3]), "u2r_recall": float(recall[3]), "u2r_f1": float(f1[3]),
    }


def confusion(labels, predictions):
    return np.bincount(5 * labels + predictions, minlength=25).reshape(5, 5)


def scaled_predictions(probabilities, kr, ku):
    scores = probabilities.astype(np.float64, copy=True)
    scores[:, 2] /= kr
    scores[:, 3] /= ku
    return scores.argmax(axis=1)


def grid_confusions(labels, probabilities, values, chunk_size=20000):
    """One R2L coefficient at a time; NumPy argmax's lowest-class tie rule."""
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros((len(values), len(values), 5, 5), dtype=np.int64)
    offsets = (25 * np.arange(len(values)))[:, None]
    majority_classes = np.array([0, 1, 4])
    for start in range(0, len(labels), chunk_size):
        stop = start + chunk_size
        p = probabilities[start:stop].astype(np.float64)
        y = labels[start:stop]
        majority = p[:, [0, 1, 4]]
        choice = majority.argmax(axis=1)
        base_ids = majority_classes[choice]
        base_scores = majority[np.arange(len(y)), choice]
        u_scores = p[:, 3][None, :] / values[:, None]
        for i, kr in enumerate(values):
            r_scores = p[:, 2] / kr
            r_wins = (r_scores > base_scores) | ((r_scores == base_scores) & (base_ids > 2))
            middle_ids = np.where(r_wins, 2, base_ids)
            middle_scores = np.maximum(r_scores, base_scores)
            u_wins = (u_scores > middle_scores) | ((u_scores == middle_scores) & (middle_ids > 3))
            predictions = np.broadcast_to(middle_ids, u_scores.shape).copy()
            predictions[u_wins] = 3
            codes = offsets + 5 * y[None, :] + predictions
            result[i] += np.bincount(codes.ravel(), minlength=25 * len(values)).reshape(len(values), 5, 5)
    return result.reshape(len(values) ** 2, 5, 5)


def self_test():
    perfect = metrics_from_confusion(np.eye(5, dtype=int))
    assert perfect["mcc"] == perfect["rare_f1"] == perfect["macro_f1"] == 1
    y, pred = np.array([2, 2, 3, 3]), np.array([2, 3, 3, 4])
    m = metrics_from_confusion(confusion(y, pred))
    assert math.isclose(m["r2l_f1"], 2 / 3) and math.isclose(m["u2r_f1"], 0.5)
    assert math.isclose(m["rare_f1"], 7 / 12) and math.isclose(m["macro_f1"], 7 / 30)
    assert metrics_from_confusion(confusion(np.array([4, 4]), np.array([4, 4])))["mcc"] == 0
    p = np.array([[.1, .1, .3, .3, .2], [.1, .1, .2, .2, .4], [.2, .2, .2, .2, .2]], dtype=np.float32)
    labels = np.array([2, 4, 0])
    values = [.5, 1., 2.]
    grid = grid_confusions(labels, p, values, chunk_size=2)
    for c, (kr, ku) in zip(grid, itertools.product(values, repeat=2)):
        assert np.array_equal(c, confusion(labels, scaled_predictions(p, kr, ku)))


def contrasts(values):
    bits = {CONFIGS[name][2]: v for name, v in values.items()}
    marginal = {}
    for axis, name in enumerate(("focal", "batching", "scaling")):
        differences = []
        for state, value in bits.items():
            if state[axis] == 0:
                active = list(state)
                active[axis] = 1
                differences.append(bits[tuple(active)] - value)
        marginal[name] = statistics.mean(differences)
    pairs = {}
    for a, b, name in ((0, 1, "focal_batching"), (0, 2, "focal_scaling"), (1, 2, "batching_scaling")):
        differences = []
        for other in (0, 1):
            state = [0, 0, 0]
            state[3 - a - b] = other
            total = 0
            for x, y, sign in ((1, 1, 1), (1, 0, -1), (0, 1, -1), (0, 0, 1)):
                state[a], state[b] = x, y
                total += sign * bits[tuple(state)]
            differences.append(total)
        pairs[name] = statistics.mean(differences)
    return {"marginal_pp": marginal, "pairwise_pp": pairs,
            "three_way_pp": sum((1 if sum(b) % 2 else -1) * v for b, v in bits.items())}


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    started = time.monotonic()
    repo = args.repo.resolve()
    claims = json.loads(args.claims.read_text())
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    counts, failures = {}, []

    def check(kind, detail, expected, observed, *, rounded=False):
        counts[kind] = counts.get(kind, 0) + 1
        if rounded:
            match = f"{float(expected):.2f}" == f"{float(observed):.2f}"
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            match = math.isclose(float(expected), float(observed), rel_tol=1e-12, abs_tol=1e-12)
        else:
            match = expected == observed
        if not match:
            failures.append({"kind": kind, "detail": detail, "expected": expected, "observed": observed})

    def path(recorded):
        p = Path(recorded)
        original = Path(claims["original_server_repo"])
        if p.is_absolute():
            p = repo / p.relative_to(original)
        else:
            p = repo / p
        p = p.resolve()
        if not p.is_relative_to(repo):
            raise ValueError(f"Recorded input escapes repository: {recorded}")
        return p

    def verified_json(recorded, expected_hash, detail):
        p = path(recorded)
        check("metadata_sha256", detail, expected_hash, digest(p))
        return json.loads(p.read_text())

    selection = verified_json(claims["selection_manifest"], claims["selection_sha256"], "selection manifest")
    evaluation = verified_json(claims["evaluation_protocol"], claims["evaluation_sha256"], "evaluation protocol")
    check("frozen_selection_link", "evaluation selection ID", selection["selection_id"], evaluation["selection_id"])
    check("frozen_selection_link", "evaluation selection SHA", claims["selection_sha256"], evaluation["selection_manifest_sha256"])
    check("selection_protocol", "selection did not access test", False, selection["kddtest_accessed"])
    check("selection_protocol", "test coefficient selection", False, evaluation["coefficient_selection_on_kddtest"])
    check("selection_protocol", "test evaluator network retraining", False, evaluation["network_retraining"])
    check("selection_protocol", "architectures", list(ARCHITECTURES), selection["architectures"])
    check("selection_protocol", "training seeds", [0, 1, 2], selection["seeds"])
    selected = {(r["architecture"], r["base_training"]): r for r in selection["selected_coefficients"]}
    for r in claims["coefficients"]:
        actual = selected[r["architecture"], r["base_training"]]
        for key in ("r2l_score_coefficient", "u2r_score_coefficient"):
            check("paper_coefficients", f'{r["architecture"]}/{r["base_training"]}/{key}', r[key], actual[key])

    labels_by_partition, dataset_counts = {}, {}
    for partition, filename in (("oof", "KDDTrain+.txt"), ("kddtest", "KDDTest+.txt")):
        p = repo / "data" / filename
        check("dataset_sha256", filename, claims["dataset_sha256"][partition], digest(p))
        with p.open(newline="") as f:
            labels = np.array([TAXONOMY[row[41]] for row in csv.reader(f)], dtype=np.int64)
        labels_by_partition[partition] = labels
        dataset_counts[partition] = np.bincount(labels, minlength=5).tolist()
        check("paper_class_counts", partition, claims["class_counts"][partition], dataset_counts[partition])

    seed_rows, file_checks, grid_rows, method_records = [], [], [], []
    common_folds = None
    for partition in ("oof", "kddtest"):
        for architecture in ARCHITECTURES:
            for regime in REGIMES:
                arrays = []
                source = (selection["source_metadata"] if partition == "oof" else evaluation["test_source_metadata"])[architecture][regime]
                if partition == "oof":
                    protocol = verified_json(source["protocol"], source["protocol_sha256"], f"{architecture}/{regime} OOF protocol")
                    settings = protocol.get("settings", protocol.get("training_settings"))
                    if not isinstance(settings, dict):
                        raise ValueError(f"No recognized training settings in {source['protocol']}")
                    for k, v in {"epochs": 25, "batch_size": 256, "fold_count": 4, "fold_seed": 0, "training_seeds": [0, 1, 2], "validation_used_during_training": False, "ctgan": False}.items():
                        check("oof_settings", f"{architecture}/{regime}/{k}", v, settings[k])
                    if regime == "focal_only":
                        candidate_count = protocol["expected_configurations"]
                        fits = protocol["expected_fits"]
                        method_records.append({"architecture": architecture, "protocol": source["protocol"], "candidate_count": candidate_count, "expected_fits": fits})
                        check("paper_focal_search_budget", architecture, 18, candidate_count)
                for seed in (0, 1, 2):
                    ref = source["oof_files"][str(seed)] if partition == "oof" else source[str(seed)]
                    p = path(ref["path"])
                    actual_hash = digest(p)
                    check("prediction_sha256", f"{partition}/{architecture}/{regime}/{seed}", ref["sha256"], actual_hash)
                    with np.load(p, allow_pickle=False) as z:
                        labels = z["labels"].astype(np.int64)
                        probabilities = z["probabilities"].astype(np.float32)
                        saved_raw = z["raw_predictions"].astype(np.int64)
                        if partition == "oof":
                            rows, folds = z["row_indices"], z["fold_ids"]
                            check("oof_row_order", str(p), True, bool(np.array_equal(rows, np.arange(len(labels)))))
                            check("oof_fold_ids", str(p), [0, 1, 2, 3], np.unique(folds).tolist())
                            if common_folds is None:
                                common_folds = folds.copy()
                            check("oof_common_folds", str(p), True, bool(np.array_equal(folds, common_folds)))
                    if probabilities.shape != (len(labels), 5) or saved_raw.shape != labels.shape:
                        raise ValueError(f"Invalid prediction array shape: {p}")
                    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-6) or np.any(probabilities > 1 + 1e-6):
                        raise ValueError(f"Invalid probabilities: {p}")
                    check("probability_row_sums", str(p), True, bool(np.allclose(probabilities.sum(axis=1), 1, atol=2e-4, rtol=0)))
                    check("labels_match_dataset", str(p), True, bool(np.array_equal(labels, labels_by_partition[partition])))
                    raw = probabilities.argmax(axis=1)
                    check("raw_argmax_matches", str(p), True, bool(np.array_equal(raw, saved_raw)))
                    file_checks.append({"partition": partition, "architecture": architecture, "base_training": regime, "seed": seed, "path": str(p), "expected_sha256": ref["sha256"], "actual_sha256": actual_hash, "row_count": len(labels)})
                    arrays.append((labels, probabilities))
                    chosen = selected[architecture, regime]
                    for name, (underlying, scaled, _) in CONFIGS.items():
                        if underlying != regime:
                            continue
                        kr = chosen["r2l_score_coefficient"] if scaled else 1.0
                        ku = chosen["u2r_score_coefficient"] if scaled else 1.0
                        pred = scaled_predictions(probabilities, kr, ku) if scaled else raw
                        cm = confusion(labels, pred)
                        metrics = metrics_from_confusion(cm)
                        seed_rows.append({"partition": partition, "architecture": architecture, "configuration": name, "seed": seed, "r2l_score_coefficient": kr, "u2r_score_coefficient": ku, **metrics, "confusion_matrix": json.dumps(cm.tolist())})
                    if partition == "kddtest":
                        run_record = verified_json(ref["run_metadata_path"], ref["run_metadata_sha256"], str(p))
                        focal = regime in ("focal_only", "focal_batch")
                        for key, value in {"epochs_requested": 25, "epochs_completed": 25, "batch_size": 256, "validation_used_during_training": False, "kddtest_used_for_selection": False, "ctgan_used": False, "synthetic_rows": 0, "model_parameters": claims["parameters"][architecture], "cb_beta": claims["focal_parameters"][architecture]["beta"] if focal else None, "focal_gamma": claims["focal_parameters"][architecture]["gamma"] if focal else None, "minority_per_batch_per_class": 1 if regime in ("batch_only", "focal_batch") else 0}.items():
                            check("final_fit_settings", f"{architecture}/{regime}/{seed}/{key}", value, run_record[key])
                        raw_metrics = metrics_from_confusion(confusion(labels, raw))
                        for metric in METRICS:
                            check("raw_run_record_metric", f"{architecture}/{regime}/{seed}/{metric}", run_record["raw_argmax_metrics"][metric], raw_metrics[metric])
                if partition == "oof" and not args.skip_coefficient_grid:
                    values = selection["coefficient_values"]
                    pairs = list(itertools.product(values, repeat=2))
                    per_seed = [[metrics_from_confusion(c) for c in grid_confusions(y, p, values)] for y, p in arrays]
                    ranking = []
                    for i, (kr, ku) in enumerate(pairs):
                        rare = [part[i]["rare_f1"] for part in per_seed]
                        macro = [part[i]["macro_f1"] for part in per_seed]
                        ranking.append({"architecture": architecture, "base_training": regime, "r2l_score_coefficient": kr, "u2r_score_coefficient": ku, "rare_f1_mean": float(np.mean(rare)), "macro_f1_mean": float(np.mean(macro)), "scaling_log_distance": float(abs(np.log(kr)) + abs(np.log(ku))), "rare_f1_std": float(np.std(rare, ddof=1))})
                    ranking.sort(key=lambda r: (-r["rare_f1_mean"], -r["macro_f1_mean"], r["scaling_log_distance"], r["rare_f1_std"], r["r2l_score_coefficient"], r["u2r_score_coefficient"]))
                    for rank, row in enumerate(ranking, 1):
                        row["rank"] = rank
                    grid_rows.extend(ranking)
                    chosen = selected[architecture, regime]
                    for k in ("r2l_score_coefficient", "u2r_score_coefficient"):
                        check("independent_coefficient_grid_winner", f"{architecture}/{regime}/{k}", chosen[k], ranking[0][k])
                print(f"Verified {partition}: {architecture}/{regime}", flush=True)

    for partition, outputs in (("oof", selection["outputs"]), ("kddtest", evaluation["outputs"])):
        key = "validation_seed_metrics" if partition == "oof" else "seed_metrics"
        with path(outputs[key]).open(newline="") as f:
            recorded = {(r["architecture"], r["configuration"], int(r["seed"])): r for r in csv.DictReader(f)}
        check("seed_csv_rows", partition, 96, len(recorded))
        for row in [r for r in seed_rows if r["partition"] == partition]:
            identity = (row["architecture"], STORED_CONFIGS[row["configuration"]], row["seed"])
            for metric in METRICS:
                check("seed_csv_metric", f"{partition}/{identity}/{metric}", float(recorded[identity][metric]), row[metric])

    summaries, aggregate, contrast_results = {}, {}, {}
    summary_rows, architecture_rows = [], []
    for partition in ("oof", "kddtest"):
        for architecture in ARCHITECTURES:
            for config in CONFIGS:
                group = [r for r in seed_rows if (r["partition"], r["architecture"], r["configuration"]) == (partition, architecture, config)]
                assert sorted(r["seed"] for r in group) == [0, 1, 2]
                row = {"partition": partition, "architecture": architecture, "configuration": config}
                for metric in METRICS:
                    v = [r[metric] for r in group]
                    row[metric + "_mean"] = statistics.mean(v)
                    row[metric + "_std"] = statistics.stdev(v)
                summaries[partition, architecture, config] = row
                summary_rows.append(row)
        for config in CONFIGS:
            v = [100 * summaries[partition, a, config]["rare_f1_mean"] for a in ARCHITECTURES]
            row = {"partition": partition, "configuration": config, "mean": statistics.mean(v), "minimum": min(v), "range": max(v) - min(v), "sd": statistics.stdev(v)}
            aggregate[partition, config] = row
            architecture_rows.append(row)
        contrast_results[partition] = {
            "architecture_mean": contrasts({c: aggregate[partition, c]["mean"] for c in CONFIGS}),
            "by_architecture": {a: contrasts({c: 100 * summaries[partition, a, c]["rare_f1_mean"] for c in CONFIGS}) for a in ARCHITECTURES},
        }
    for row in claims["primary"]:
        actual = summaries[row["partition"], row["architecture"], row["configuration"]]
        for name, metric in (("mean", "rare_f1_mean"), ("sd", "rare_f1_std")):
            check("paper_primary", f'{row["partition"]}/{row["architecture"]}/{row["configuration"]}/{name}', row[name], 100 * actual[metric], rounded=True)
    for row in claims["primary_aggregates"]:
        for metric in ("mean", "sd"):
            check("paper_primary_aggregate", f'{row["partition"]}/{row["configuration"]}/{metric}', row[metric], aggregate[row["partition"], row["configuration"]][metric], rounded=True)
    for row in claims["architecture_summary"]:
        for metric in ("mean", "minimum", "range", "sd"):
            check("paper_architecture_summary", f'{row["configuration"]}/{metric}', row[metric], aggregate["kddtest", row["configuration"]][metric], rounded=True)
    for row in claims["secondary"]:
        actual = 100 * statistics.mean(summaries["kddtest", a, row["configuration"]][row["metric"] + "_mean"] for a in ARCHITECTURES)
        check("paper_secondary", f'{row["configuration"]}/{row["metric"]}', row["value"], actual, rounded=True)
    for row in claims["interactions"]:
        c = contrast_results[row["partition"]]["architecture_mean"]
        actual = c["three_way_pp"] if row["interaction"] == "three_way_pp" else c["pairwise_pp"][row["interaction"]]
        check("paper_interaction", f'{row["partition"]}/{row["interaction"]}', row["value"], actual, rounded=True)
    for a, expected in claims["batching_marginal_pp"].items():
        check("paper_batching_marginal", a, expected, contrast_results["kddtest"]["by_architecture"][a]["marginal_pp"]["batching"], rounded=True)
    winners = {p: {a: max(CONFIGS, key=lambda c: summaries[p, a, c]["rare_f1_mean"]) for a in ARCHITECTURES} for p in ("oof", "kddtest")}
    for partition in winners:
        check("paper_backbone_winners", partition, claims["backbone_winners"][partition], winners[partition])
    all_batch_positive = all(c["marginal_pp"]["batching"] > 0 for c in contrast_results["kddtest"]["by_architecture"].values())
    check("paper_qualitative_claim", "positive test batching contrast in every backbone", True, all_batch_positive)
    check("paper_qualitative_claim", "all four test winners contain batching", True, all(CONFIGS[c][2][1] for c in winners["kddtest"].values()))
    test_order = sorted(CONFIGS, key=lambda c: aggregate["kddtest", c]["mean"], reverse=True)
    check("paper_qualitative_claim", "four highest architecture means contain batching", True, all(CONFIGS[c][2][1] for c in test_order[:4]))
    check("paper_qualitative_claim", "negative architecture-averaged test pairwise interactions", True, all(v < 0 for v in contrast_results["kddtest"]["architecture_mean"]["pairwise_pp"].values()))
    check("paper_qualitative_claim", "focal plus batching highest test architecture mean", "Focal + batching", test_order[0])
    check("paper_qualitative_claim", "focal plus batching smallest architecture SD", "Focal + batching", min(CONFIGS, key=lambda c: aggregate["kddtest", c]["sd"]))
    check("paper_qualitative_claim", "focal plus batching smallest architecture range", "Focal + batching", min(CONFIGS, key=lambda c: aggregate["kddtest", c]["range"]))
    check("paper_qualitative_claim", "batching plus scaling highest minimum backbone", "Batching + scaling", max(CONFIGS, key=lambda c: aggregate["kddtest", c]["minimum"]))
    check("paper_qualitative_claim", "batching plus scaling highest OOF mean", "Batching + scaling", max(CONFIGS, key=lambda c: aggregate["oof", c]["mean"]))
    check("paper_qualitative_claim", "all three controls smallest OOF architecture SD", "All three controls", min(CONFIGS, key=lambda c: aggregate["oof", c]["sd"]))

    known_locations = {
        ("paper_focal_search_budget", "mlp"),
        ("paper_architecture_summary", "Focal + scaling/range"),
        ("paper_architecture_summary", "All three controls/range"),
        ("paper_batching_marginal", "transformer"),
    }
    unexpected_failures = [f for f in failures if (f["kind"], f["detail"]) not in known_locations]
    integrity_kinds = {"metadata_sha256", "frozen_selection_link", "dataset_sha256", "prediction_sha256", "labels_match_dataset", "raw_argmax_matches", "probability_row_sums", "oof_row_order", "oof_fold_ids", "oof_common_folds", "raw_run_record_metric", "seed_csv_rows", "seed_csv_metric"}
    main_result_kinds = {"paper_primary", "paper_primary_aggregate", "paper_secondary", "paper_interaction", "paper_backbone_winners", "paper_qualitative_claim", "independent_coefficient_grid_winner"}
    report = {
        "paper_sha256": claims["paper_sha256"], "repository": str(repo), "output_directory": str(out),
        "numpy_version": np.__version__, "python_version": sys.version, "runtime_seconds": time.monotonic() - started,
        "training_run": False, "model_inference_run": False, "experiment_modules_imported": False,
        "prediction_files_decoded": len(file_checks), "seed_evaluations_recomputed": len(seed_rows),
        "metric_values_compared_to_seed_csv": counts.get("seed_csv_metric", 0),
        "independent_coefficient_grid_checked": not args.skip_coefficient_grid,
        "coefficient_pairs_per_family": len(selection["coefficient_values"]) ** 2,
        "check_counts": counts, "failures": failures,
        "saved_prediction_metrics_and_main_result_checks_pass": not any(f["kind"] in integrity_kinds | main_result_kinds for f in failures),
        "unexpected_failures": unexpected_failures,
        "prediction_file_checks": file_checks, "focal_search_protocols": method_records,
        "planned_focal_fits_in_referenced_protocols": sum(r["expected_fits"] for r in method_records),
        "contrasts": contrast_results, "backbone_winners": winners, "test_configuration_order": test_order,
        "headline": {"baseline_test_mean_percent": aggregate["kddtest", "Baseline"]["mean"], "focal_batching_test_mean_percent": aggregate["kddtest", "Focal + batching"]["mean"], "improvement_pp": aggregate["kddtest", "Focal + batching"]["mean"] - aggregate["kddtest", "Baseline"]["mean"]},
        "wording_note": "Negative test pairwise interactions hold after averaging architectures; all three MLP pairwise contrasts are positive.",
        "scope": "Metrics and scaling selection are independently recomputed from saved prediction arrays. Training history and focal search completeness can only be assessed from recorded metadata. Fresh model training reproducibility is not tested.",
    }
    write_csv(out / "recomputed_seed_metrics.csv", seed_rows)
    write_csv(out / "recomputed_summary.csv", summary_rows)
    write_csv(out / "recomputed_architecture_summary.csv", architecture_rows)
    if grid_rows:
        write_csv(out / "recomputed_coefficient_rankings.csv", grid_rows)
    (out / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("output_directory", "prediction_files_decoded", "seed_evaluations_recomputed", "metric_values_compared_to_seed_csv", "independent_coefficient_grid_checked", "saved_prediction_metrics_and_main_result_checks_pass", "headline", "failures", "runtime_seconds")}, indent=2))
    return 1 if unexpected_failures else (2 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--claims", type=Path, default=Path(__file__).with_name("paper_verification_claims.json"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-coefficient-grid", action="store_true", help="Only apply frozen coefficients; skip independent 576-pair OOF reselection")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("Independent metric and score-scaling tie-rule self-tests passed.")
        return 0
    if args.output_dir is None:
        parser.error("--output-dir must name a new directory")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
