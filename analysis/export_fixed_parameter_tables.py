"""Export all nine paper tables using fixed focal and score-scaling parameters.

Consumes newly trained OOF/test probabilities. No fitting or parameter search.
Also accepts explicit original frozen artifacts for validating this exporter.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np

from reverify_paper_results import (
    ARCHITECTURES, CONFIGS, METRICS, REGIMES, TAXONOMY, confusion, contrasts,
    digest, metrics_from_confusion, scaled_predictions, write_csv,
)

LABELS = {"conv2d": "Conv2D", "conv1d": "Conv1D", "transformer": "Transformer", "mlp": "MLP"}
FOCAL = {"conv2d": (0.99, 0.50), "conv1d": (0.99, 0.25), "transformer": (0.99, 0.75), "mlp": (0.99, 0.25)}
PARAMETERS = {"conv2d": 109381, "conv1d": 109797, "transformer": 110661, "mlp": 99845}
BIAS = {"conv2d": "Engineered 2-D locality", "conv1d": "Ordered 1-D locality", "transformer": "Global self-attention", "mlp": "Global dense mixing"}
REGIME_LABEL = {"baseline": "Baseline", "focal_only": "Focal only", "batch_only": "Batching only", "focal_batch": "Focal + batching"}
SELECTION_SHA256 = "0c73f26079edf9f7cdd1d643551e63bcbb22d9f2d23b76e62aa18500e70da08f"


def export_table(out, name, label, rows):
    write_csv(out / (name + ".csv"), rows)
    keys = list(rows[0])
    def tex(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value).replace("_", r"\_").replace("%", r"\%")
    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + name.replace("_", " ") + "}",
             r"\label{" + label + "}", r"\begin{tabular}{" + "l" * len(keys) + "}", r"\hline",
             " & ".join(map(tex, keys)) + r" \\", r"\hline"]
    lines += [" & ".join(tex(r[k]) for k in keys) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    (out / (name + ".tex")).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True, help="Original selection manifest; only its fixed coefficients are used")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-evaluation", type=Path, help="Validation only: read the original frozen OOF/test artifact paths instead of new latest pointers")
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.output_dir.resolve()
    if out.is_relative_to(repo / "results"):
        parser.error("Write derived paper tables outside the experiment results directory")
    if digest(args.selection) != SELECTION_SHA256:
        raise ValueError("Frozen selection does not match the paper's audited selection manifest")
    selection = json.loads(args.selection.read_text())
    original = json.loads(args.original_evaluation.read_text()) if args.original_evaluation else None
    selected = {(r["architecture"], r["base_training"]): r for r in selection["selected_coefficients"]}
    assert len(selected) == 16
    sys.path.insert(0, str(repo / "src"))
    native = importlib.import_module("tune_variant_specific_score_scaling")
    expected_labels, class_counts = {}, {}
    for partition, filename in (("oof", "KDDTrain+.txt"), ("kddtest", "KDDTest+.txt")):
        with (repo / "data" / filename).open(newline="") as f:
            y = np.array([TAXONOMY[r[41]] for r in csv.reader(f)], dtype=np.int64)
        expected_labels[partition] = y
        class_counts[partition] = np.bincount(y, minlength=5).tolist()
    seed_rows, provenance, all_fold_ids = [], [], None
    for partition in ("oof", "kddtest"):
        for arch in ARCHITECTURES:
            for regime in REGIMES:
                if partition == "oof":
                    if original:
                        refs = selection["source_metadata"][arch][regime]["oof_files"]
                        paths = {s: Path(refs[str(s)]["path"]) for s in (0, 1, 2)}
                    else:
                        paths, _ = native.load_oof_paths(repo, repo / "results", arch, regime, (0, 1, 2))
                else:
                    variant = "full" if regime == "focal_batch" else regime
                    if original:
                        refs = original["test_source_metadata"][arch][regime]
                        paths = {s: Path(refs[str(s)]["path"]) for s in (0, 1, 2)}
                    else:
                        paths = {s: native.find_test_prediction_path(repo / "results", arch, variant, s) for s in (0, 1, 2)}
                for seed, path in paths.items():
                    y, p, raw, folds = native.read_probability_artifact(path, require_oof_fields=partition == "oof")
                    if not np.array_equal(y, expected_labels[partition]):
                        raise ValueError(f"Labels do not match original dataset order: {path}")
                    if folds is not None:
                        if all_fold_ids is None:
                            all_fold_ids = folds.copy()
                        if not np.array_equal(folds, all_fold_ids):
                            raise ValueError(f"Different OOF partitions: {path}")
                    if partition == "kddtest":
                        _, record, _, _ = native.load_test_run_metadata(path, repo, arch, variant, seed)
                        focal = regime in ("focal_only", "focal_batch")
                        expected = {"model_parameters": PARAMETERS[arch], "epochs_requested": 25,
                                    "epochs_completed": 25, "batch_size": 256,
                                    "cb_beta": FOCAL[arch][0] if focal else None,
                                    "focal_gamma": FOCAL[arch][1] if focal else None,
                                    "minority_per_batch_per_class": 1 if regime in ("batch_only", "focal_batch") else 0}
                        for k, v in expected.items():
                            if record[k] != v:
                                raise ValueError(f"Frozen fit setting mismatch: {path}: {k}")
                    provenance.append({"partition": partition, "architecture": arch, "regime": regime, "seed": seed, "path": str(path), "sha256": digest(path)})
                    chosen = selected[arch, regime]
                    for name, (base, scaling, _) in CONFIGS.items():
                        if base != regime:
                            continue
                        kr = float(chosen["r2l_score_coefficient"]) if scaling else 1.
                        ku = float(chosen["u2r_score_coefficient"]) if scaling else 1.
                        predictions = scaled_predictions(p, kr, ku) if scaling else raw
                        seed_rows.append({"partition": partition, "architecture": arch, "configuration": name,
                                          "seed": seed, "r2l_score_coefficient": kr, "u2r_score_coefficient": ku,
                                          **metrics_from_confusion(confusion(y, predictions))})
    summary, raw_summary = {}, []
    for partition in ("oof", "kddtest"):
        for arch in ARCHITECTURES:
            for name in CONFIGS:
                group = [r for r in seed_rows if (r["partition"], r["architecture"], r["configuration"]) == (partition, arch, name)]
                assert sorted(r["seed"] for r in group) == [0, 1, 2]
                row = {"partition": partition, "architecture": arch, "configuration": name}
                for metric in METRICS:
                    values = [r[metric] for r in group]
                    row[metric + "_mean"] = statistics.mean(values)
                    row[metric + "_std"] = statistics.stdev(values)
                summary[partition, arch, name] = row
                raw_summary.append(row)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "seed_metrics.csv", seed_rows)
    write_csv(out / "summary_unrounded.csv", raw_summary)
    export_table(out, "table_01_class_counts", "tab:class_counts", [
        {"Class": name, "KDDTrain+": class_counts["oof"][i], "KDDTest+": class_counts["kddtest"][i]}
        for i, name in ((4, "Normal"), (0, "DoS"), (1, "Probe"), (2, "R2L"), (3, "U2R"))] +
        [{"Class": "Total", "KDDTrain+": sum(class_counts["oof"]), "KDDTest+": sum(class_counts["kddtest"])}])
    export_table(out, "table_02_architectures", "tab:architectures", [
        {"Backbone": LABELS[a], "Total parameters": PARAMETERS[a], "Inductive bias": BIAS[a]} for a in ARCHITECTURES])
    export_table(out, "table_03_focal_parameters", "tab:focal_parameters", [
        {"Backbone": LABELS[a], "Beta": FOCAL[a][0], "Gamma": FOCAL[a][1]} for a in ARCHITECTURES])
    export_table(out, "table_04_score_parameters", "tab:score_parameters", [
        {"Backbone": LABELS[a], "Underlying training": REGIME_LABEL[g],
         "k_R2L": float(selected[a, g]["r2l_score_coefficient"]), "k_U2R": float(selected[a, g]["u2r_score_coefficient"])}
        for a in ARCHITECTURES for g in REGIMES])
    aggregates = {}
    for partition, number, label in (("oof", "05", "tab:oof_results"), ("kddtest", "06", "tab:test_results")):
        rows = []
        for name in CONFIGS:
            means = [100 * summary[partition, a, name]["rare_f1_mean"] for a in ARCHITECTURES]
            aggregates[partition, name] = {"mean": statistics.mean(means), "minimum": min(means), "range": max(means) - min(means), "sd": statistics.stdev(means)}
            row = {"Configuration": name}
            for a in ARCHITECTURES:
                r = summary[partition, a, name]
                row[LABELS[a]] = f"${100*r['rare_f1_mean']:.2f}\\pm{100*r['rare_f1_std']:.2f}$"
            row.update({"Arch. Mean": aggregates[partition, name]["mean"], "Arch. SD": aggregates[partition, name]["sd"]})
            rows.append(row)
        export_table(out, "table_" + number + "_" + partition + "_results", label, rows)
    export_table(out, "table_07_architecture_summary", "tab:architecture_summary", [
        {"Configuration": name, "Mean": aggregates["kddtest", name]["mean"], "Minimum": aggregates["kddtest", name]["minimum"],
         "Range": aggregates["kddtest", name]["range"], "SD": aggregates["kddtest", name]["sd"]} for name in CONFIGS])
    secondary = []
    for name in CONFIGS:
        row = {"Configuration": name}
        for label, metric in (("MCC", "mcc"), ("Macro-F1", "macro_f1"), ("Rare Macro-F1", "rare_f1"), ("R2L F1", "r2l_f1"), ("U2R F1", "u2r_f1"), ("R2L Recall", "r2l_recall"), ("U2R Recall", "u2r_recall")):
            row[label] = 100 * statistics.mean(summary["kddtest", a, name][metric + "_mean"] for a in ARCHITECTURES)
        secondary.append(row)
    export_table(out, "table_08_secondary_test", "tab:secondary_test", secondary)
    c = {p: contrasts({n: aggregates[p, n]["mean"] for n in CONFIGS}) for p in ("oof", "kddtest")}
    interactions = []
    for label, key in (("Focal x batching", "focal_batching"), ("Focal x scaling", "focal_scaling"), ("Batching x scaling", "batching_scaling"), ("Focal x batching x scaling", "three_way_pp")):
        interactions.append({"Interaction": label, **{p: c[p]["three_way_pp"] if key == "three_way_pp" else c[p]["pairwise_pp"][key] for p in c}})
    export_table(out, "table_09_factorial_interactions", "tab:factorial_interactions", interactions)
    (out / "provenance.json").write_text(json.dumps({"fixed_selection_sha256": digest(args.selection), "focal_parameters": FOCAL,
        "coefficient_search_performed": False, "model_training_performed_by_exporter": False,
        "probability_files": provenance, "seed_evaluations": len(seed_rows), "table_count": 9,
        "metric_scale": "percentages or 100 x MCC; sample SD uses n-1", "dataset_sha256": {
            name: digest(repo / "data" / name) for name in ("KDDTrain+.txt", "KDDTest+.txt")}}, indent=2) + "\n")
    print(f"Exported all 9 paper tables (CSV and LaTeX), 192 seed evaluations, and provenance to {out}")


if __name__ == "__main__":
    main()
