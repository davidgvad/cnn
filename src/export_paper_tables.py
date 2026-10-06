"""format experiment summaries and calculate architecture and factorial statistics"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import run_no_ctgan_model_ablation_4gpu as core
import tune_variant_specific_score_scaling as scoring

REPO = Path(__file__).resolve().parents[1]
CONFIGURATIONS = {
    "baseline": ("Baseline", (0, 0, 0)),
    "focal_only": ("Focal only", (1, 0, 0)),
    "batch_only": ("Batching only", (0, 1, 0)),
    "scaling_only_tuned": ("Scaling only", (0, 0, 1)),
    "focal_batch": ("Focal + batching", (1, 1, 0)),
    "focal_scaling_tuned": ("Focal + scaling", (1, 0, 1)),
    "batch_scaling_tuned": ("Batching + scaling", (0, 1, 1)),
    "full_retuned": ("All three controls", (1, 1, 1)),
}
REGIME_LABELS = {"baseline": "Baseline", "focal_only": "Focal only",
                 "batch_only": "Batching only", "focal_batch": "Focal + batching"}
INDUCTIVE_BIAS = {"conv2d": "Engineered 2-D locality", "conv1d": "Ordered 1-D locality",
                  "transformer": "Global self-attention", "mlp": "Global dense mixing"}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def export_table(directory, name, label, rows):
    write_csv(directory / f"{name}.csv", rows)
    columns = list(rows[0])

    def tex(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value).replace("_", r"\_").replace("%", r"\%")

    lines = [r"\begin{table*}[t]", r"\centering", r"\caption{" + name.replace("_", " ") + "}",
             r"\label{" + label + "}", r"\begin{tabular}{" + "l" * len(columns) + "}", r"\hline",
             " & ".join(map(tex, columns)) + r" \\", r"\hline"]
    lines += [" & ".join(tex(row[column]) for column in columns) + r" \\" for row in rows]
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    (directory / f"{name}.tex").write_text("\n".join(lines) + "\n")


def read_summary(path, seeds):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(architecture, configuration) for architecture in scoring.ARCHITECTURES
                for configuration in CONFIGURATIONS}
    keys = [(row["architecture"], row["configuration"]) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError(f"Expected all four backbones and eight configurations: {path}")
    summary = {}
    for row, key in zip(rows, keys):
        if int(row["runs"]) != len(seeds) or sorted(map(int, row["seeds"].split(","))) != sorted(seeds):
            raise ValueError(f"Incomplete seed summary: {path}: {key}")
        values = {f"{metric}_{stat}": float(row[f"{metric}_{stat}"])
                  for metric in scoring.METRICS for stat in ("mean", "std")}
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite summary value: {path}: {key}")
        if any(values[f"{metric}_std"] < 0 for metric in scoring.METRICS):
            raise ValueError(f"Negative standard deviation: {path}: {key}")
        summary[key] = values
    return summary


def contrasts(values):
    """Factors are focal loss, batching, and scaling, in that order."""
    cells = {CONFIGURATIONS[name][1]: value for name, value in values.items()}
    marginal = {}
    for axis, name in enumerate(("focal", "batching", "scaling")):
        differences = []
        for state, value in cells.items():
            if state[axis] == 0:
                active = list(state)
                active[axis] = 1
                differences.append(cells[tuple(active)] - value)
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
                total += sign * cells[tuple(state)]
            differences.append(total)
        pairs[name] = statistics.mean(differences)
    return {"marginal_pp": marginal, "pairwise_pp": pairs,
            "three_way_pp": sum((1 if sum(state) % 2 else -1) * value for state, value in cells.items())}


def analyze(results_dir, output_dir, data_dir, selection=None, evaluation=None):
    results_dir, output_dir, data_dir = map(lambda path: Path(path).resolve(),
                                         (results_dir, output_dir, data_dir))
    if output_dir == results_dir or output_dir.is_relative_to(data_dir):
        raise ValueError("Write tables in a separate directory from results and source data.")
    evaluation_path = Path(evaluation).resolve() if evaluation else results_dir / "variant_specific_scaling_kddtest_latest.json"
    protocol = core.read_json(evaluation_path)
    if "protocol" in protocol:
        protocol_path = scoring.resolve_recorded_path(protocol["protocol"], REPO, evaluation_path)
        if core.sha256_file(protocol_path) != protocol["protocol_sha256"]:
            raise ValueError("Evaluation protocol changed.")
        protocol = core.read_json(protocol_path)
    else:
        protocol_path = evaluation_path
    selected_path = selection or scoring.resolve_recorded_path(protocol["selection_manifest"], REPO, protocol_path)
    manifest_path, manifest = scoring.load_selection_manifest(str(selected_path), REPO, results_dir)
    if core.sha256_file(manifest_path) != protocol["selection_manifest_sha256"]:
        raise ValueError("OOF selection and test evaluation come from different experiments.")
    seeds = tuple(manifest["seeds"])
    if len(seeds) < 2 or set(manifest["architectures"]) != set(scoring.ARCHITECTURES):
        raise ValueError("The analysis needs all four backbones and at least two seeds.")
    choices = scoring.validate_selection_lineage(manifest_path, manifest, REPO, data_dir, verify_arrays=False)
    summary_paths = {"oof": scoring.resolve_recorded_path(manifest["outputs"]["validation_summary"], REPO, manifest_path),
                     "kddtest": scoring.resolve_recorded_path(protocol["outputs"]["summary"], REPO, protocol_path)}
    summaries = {partition: read_summary(path, seeds) for partition, path in summary_paths.items()}
    parameters = {}
    for architecture in scoring.ARCHITECTURES:
        record = protocol["test_source_metadata"][architecture]["baseline"][str(seeds[0])]
        run_path = scoring.resolve_recorded_path(record["run_metadata_path"], REPO, protocol_path)
        parameters[architecture] = int(core.read_json(run_path)["model_parameters"])
    counts = {}
    for partition, filename in (("oof", "KDDTrain+.txt"), ("kddtest", "KDDTest+.txt")):
        frame = core.load_collapsed_nsl_kdd(data_dir / filename, is_train=partition == "oof")
        counts[partition] = frame["class"].value_counts().reindex(range(5), fill_value=0).astype(int).tolist()
    output_dir.mkdir(parents=True, exist_ok=True)
    export_table(output_dir, "table_01_class_counts", "tab:class_counts", [
        {"Class": name, "KDDTrain+": counts["oof"][i], "KDDTest+": counts["kddtest"][i]}
        for i, name in ((4, "Normal"), (0, "DoS"), (1, "Probe"), (2, "R2L"), (3, "U2R"))] +
        [{"Class": "Total", "KDDTrain+": sum(counts["oof"]), "KDDTest+": sum(counts["kddtest"])}])
    labels = scoring.ARCHITECTURE_LABELS
    export_table(output_dir, "table_02_architectures", "tab:architectures", [
        {"Backbone": labels[architecture], "Total parameters": parameters[architecture],
         "Inductive bias": INDUCTIVE_BIAS[architecture]} for architecture in scoring.ARCHITECTURES])
    export_table(output_dir, "table_03_focal_parameters", "tab:focal_parameters", [
        {"Backbone": labels[architecture], "Beta": choices[architecture]["beta"],
         "Gamma": choices[architecture]["focal_gamma"]} for architecture in scoring.ARCHITECTURES])
    selected = {(row["architecture"], row["base_training"]): row for row in manifest["selected_coefficients"]}
    export_table(output_dir, "table_04_score_parameters", "tab:score_parameters", [
        {"Backbone": labels[architecture], "Underlying training": REGIME_LABELS[regime],
         "k_R2L": float(selected[architecture, regime]["r2l_score_coefficient"]),
         "k_U2R": float(selected[architecture, regime]["u2r_score_coefficient"])}
        for architecture in scoring.ARCHITECTURES for regime in scoring.BASE_TRAINING_ORDER])
    aggregates = {}
    for partition, number, label in (("oof", "05", "tab:oof_results"), ("kddtest", "06", "tab:test_results")):
        rows = []
        for configuration, (display, _) in CONFIGURATIONS.items():
            means = [100 * summaries[partition][architecture, configuration]["rare_f1_mean"]
                     for architecture in scoring.ARCHITECTURES]
            aggregate = {"mean": statistics.mean(means), "minimum": min(means),
                         "range": max(means) - min(means), "sd": statistics.stdev(means)}
            aggregates[partition, configuration] = aggregate
            row = {"Configuration": display}
            for architecture in scoring.ARCHITECTURES:
                observed = summaries[partition][architecture, configuration]
                row[labels[architecture]] = f"${100 * observed['rare_f1_mean']:.2f}\\pm{100 * observed['rare_f1_std']:.2f}$"
            row.update({"Arch. Mean": aggregate["mean"], "Arch. SD": aggregate["sd"]})
            rows.append(row)
        export_table(output_dir, f"table_{number}_{partition}_results", label, rows)
    export_table(output_dir, "table_07_architecture_summary", "tab:architecture_summary", [
        {"Configuration": display, "Mean": aggregates["kddtest", configuration]["mean"],
         "Minimum": aggregates["kddtest", configuration]["minimum"],
         "Range": aggregates["kddtest", configuration]["range"],
         "SD": aggregates["kddtest", configuration]["sd"]} for configuration, (display, _) in CONFIGURATIONS.items()])
    secondary = []
    for configuration, (display, _) in CONFIGURATIONS.items():
        row = {"Configuration": display}
        for label, metric in (("MCC", "mcc"), ("Macro-F1", "macro_f1"), ("Rare Macro-F1", "rare_f1"),
                             ("R2L F1", "r2l_f1"), ("U2R F1", "u2r_f1"),
                             ("R2L Recall", "r2l_recall"), ("U2R Recall", "u2r_recall")):
            row[label] = 100 * statistics.mean(summaries["kddtest"][architecture, configuration][metric + "_mean"]
                                             for architecture in scoring.ARCHITECTURES)
        secondary.append(row)
    export_table(output_dir, "table_08_secondary_test", "tab:secondary_test", secondary)
    effects, contrast_rows = {}, []
    for partition in summaries:
        for architecture in (*scoring.ARCHITECTURES, "architecture_mean"):
            values = {configuration: aggregates[partition, configuration]["mean"] if architecture == "architecture_mean"
                      else 100 * summaries[partition][architecture, configuration]["rare_f1_mean"]
                      for configuration in CONFIGURATIONS}
            effect = contrasts(values)
            if architecture == "architecture_mean":
                effects[partition] = effect
            contrast_rows.append({"partition": partition, "architecture": architecture,
                **{name + "_marginal_pp": value for name, value in effect["marginal_pp"].items()},
                **{name + "_interaction_pp": value for name, value in effect["pairwise_pp"].items()},
                "three_way_pp": effect["three_way_pp"]})
    interactions = [{"Interaction": label, **{partition: effect["three_way_pp"] if key == "three_way_pp"
                    else effect["pairwise_pp"][key] for partition, effect in effects.items()}}
                    for label, key in (("Focal x batching", "focal_batching"), ("Focal x scaling", "focal_scaling"),
                                       ("Batching x scaling", "batching_scaling"), ("Focal x batching x scaling", "three_way_pp"))]
    export_table(output_dir, "table_09_factorial_interactions", "tab:factorial_interactions", interactions)
    write_csv(output_dir / "factorial_contrasts.csv", contrast_rows)
    write_csv(output_dir / "summary_unrounded.csv", [
        {"partition": partition, "architecture": architecture,
         "configuration": CONFIGURATIONS[configuration][0], **values}
        for partition, summary in summaries.items() for (architecture, configuration), values in summary.items()])
    inputs = {str(path): core.sha256_file(path) for path in (manifest_path, protocol_path, *summary_paths.values())}
    (output_dir / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    print(f"Saved nine tables and factorial contrasts to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO / "results")
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO / "tables")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    analyze(args.results_dir, args.output_dir, args.data_dir, args.selection, args.evaluation)


if __name__ == "__main__":
    main()
