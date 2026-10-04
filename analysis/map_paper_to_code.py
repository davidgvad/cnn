"""Read the supplied LaTeX paper and map its experiment to local source files.

Usage: python3 analysis/map_paper_to_code.py /path/to/paper.txt
Uses the standard library only. Does not import or execute experiment code.
Paper table checks use printed rounded numbers, not missing raw predictions.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"
ARCHITECTURES = ("conv2d", "conv1d", "transformer", "mlp")
CELLS = {
    "Baseline": (0, 0, 0),
    "Focal only": (1, 0, 0),
    "Batching only": (0, 1, 0),
    "Scaling only": (0, 0, 1),
    "Focal + batching": (1, 1, 0),
    "Focal + scaling": (1, 0, 1),
    "Batching + scaling": (0, 1, 1),
    "All three controls": (1, 1, 1),
}
CORE = {
    "cnn_gan_foc.py": "Shared ClassBalancedFocalLoss; GAN generation/main is outside this paper.",
    "cnn_opt.py": "Shared Conv2D builder, BalancedBatchSequence and score helpers; legacy main is outside this paper.",
    "cnn_opt_1d_4gpu.py": "Shared Conv1D, feature Transformer and MLP builders; legacy main is outside this paper.",
    "run_no_ctgan_model_ablation_4gpu.py": "Shared raw mapping, fold preprocessor, class weights, metrics and artifact utilities; its holdout main is not the paper CV protocol.",
    "tune_conv2d_focal_cv_4gpu.py": "Conv2D Stage-1 18-candidate four-fold focal selection and selected focal-only OOF arrays.",
    "tune_conv1d_focal_cv_4gpu.py": "Conv1D Stage-1 focal selection and selected focal-only OOF arrays.",
    "tune_transformer_focal_cv_4gpu.py": "Transformer Stage-1 focal selection and selected focal-only OOF arrays.",
    "tune_mlp_focal_cv_4gpu.py": "MLP Stage-1 focal selection and selected focal-only OOF arrays.",
    "tune_conv2d_score_scaling_cv_4gpu.py": "Generic four-backbone OOF engine for CE, CE+B and F+B, plus shared 576-pair scoring machinery.",
    "run_final_baseline_vs_full_kddtest_4gpu.py": "Four-regime final network fits and saved raw test probabilities; own older scaled results do not match all paper coefficients.",
    "tune_variant_specific_score_scaling.py": "Independent per-regime scaling selection, frozen manifest, complete OOF/test eight-cell tables and secondary metrics.",
}


def strip_comments(text):
    lines = []
    for line in text.splitlines(keepends=True):
        cutoff = len(line)
        for i, char in enumerate(line):
            if char != "%":
                continue
            backslashes, j = 0, i - 1
            while j >= 0 and line[j] == "\\":
                backslashes, j = backslashes + 1, j - 1
            if backslashes % 2 == 0:
                cutoff = i
                break
        clean = line[:cutoff].rstrip("\r\n")
        lines.append(clean + ("\n" if line.endswith("\n") else ""))
    return "".join(lines)


def numeric(cell):
    found = re.findall(r"-?\d+(?:\.\d+)?", cell.replace(",", ""))
    if len(found) != 1:
        raise ValueError("Expected one printed number: " + cell)
    return float(found[0])


def table_rows(table):
    body = table.split("\\midrule", 1)[1].split("\\bottomrule", 1)[0]
    body = body.replace("\\midrule", "")
    return [[cell.strip() for cell in row.strip().split("&")]
            for row in re.split(r"\\\\", body) if "&" in row]


def contrasts(values):
    marginal = {}
    for axis, name in enumerate(("focal", "batching", "scaling")):
        differences = []
        for bits, value in values.items():
            if bits[axis] != 0:
                continue
            active = list(bits)
            active[axis] = 1
            differences.append(values[tuple(active)] - value)
        marginal[name] = statistics.mean(differences)
    pairwise = {}
    for a, b, name in ((0, 1, "focal_batching"), (0, 2, "focal_scaling"), (1, 2, "batching_scaling")):
        remaining = 3 - a - b
        diffs = []
        for state in (0, 1):
            base = [0, 0, 0]
            base[remaining] = state
            total = 0.0
            for x, y, sign in ((1, 1, 1), (1, 0, -1), (0, 1, -1), (0, 0, 1)):
                bits = base.copy()
                bits[a], bits[b] = x, y
                total += sign * values[tuple(bits)]
            diffs.append(total)
        pairwise[name] = statistics.mean(diffs)
    threeway = sum((1 if sum(bits) % 2 == 1 else -1) * value for bits, value in values.items())
    return {"marginal_pp": marginal, "pairwise_pp": pairwise, "three_way_pp": threeway}


def literal_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paper", type=Path)
    args = parser.parse_args()
    data = args.paper.read_bytes()
    original = data.decode("utf-8")
    active = strip_comments(original)
    tables = {}
    for match in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", active, re.S):
        label = re.search(r"\\label\{([^}]+)\}", match.group())
        if label:
            tables[label.group(1)] = {"line": active[:match.start()].count("\n") + 1,
                                      "latex": match.group()}
    required = {"tab:oof_results", "tab:test_results", "tab:score_parameters", "tab:focal_parameters"}
    assert required <= set(tables), set(tables)
    table_csv_rows = []
    parsed_results = {}
    checks = []
    for label, partition in (("tab:oof_results", "oof"), ("tab:test_results", "kddtest")):
        parsed_results[partition] = {}
        for cells in table_rows(tables[label]["latex"]):
            name = cells[0]
            assert name in CELLS and len(cells) == 7, cells
            values = []
            for architecture, cell in zip(ARCHITECTURES, cells[1:5]):
                mean, sd = re.search(r"(-?\d+(?:\.\d+)?)\s*\\pm\s*(-?\d+(?:\.\d+)?)", cell).groups()
                mean, sd = float(mean), float(sd)
                values.append(mean)
                f, b, s = CELLS[name]
                table_csv_rows.append({"partition": partition, "configuration": name,
                                       "architecture": architecture, "focal": f, "batching": b,
                                       "scaling": s, "rare_f1_mean_percent": mean,
                                       "rare_f1_seed_sample_sd_pp": sd})
            printed_mean, printed_sd = numeric(cells[5]), numeric(cells[6])
            reconstructed_mean, reconstructed_sd = statistics.mean(values), statistics.stdev(values)
            assert abs(printed_mean - reconstructed_mean) <= 0.011
            assert abs(printed_sd - reconstructed_sd) <= 0.011
            parsed_results[partition][name] = {"architecture_means": dict(zip(ARCHITECTURES, values)),
                                              "printed_architecture_mean": printed_mean,
                                              "printed_architecture_sd": printed_sd,
                                              "reconstructed_architecture_mean": reconstructed_mean,
                                              "reconstructed_architecture_sd": reconstructed_sd}
            checks.append({"partition": partition, "configuration": name,
                           "mean_and_sd_consistent_with_printed_cells": True})
    with (OUT / "paper_rare_f1_cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table_csv_rows[0]))
        writer.writeheader()
        writer.writerows(table_csv_rows)

    contrast_output = {}
    for partition, rows in parsed_results.items():
        architecture_mean = {CELLS[name]: row["reconstructed_architecture_mean"] for name, row in rows.items()}
        contrast_output[partition] = {"architecture_mean": contrasts(architecture_mean), "by_architecture": {}}
        for architecture in ARCHITECTURES:
            values = {CELLS[name]: row["architecture_means"][architecture] for name, row in rows.items()}
            contrast_output[partition]["by_architecture"][architecture] = contrasts(values)
    expected_interactions = {}
    for cells in table_rows(tables["tab:factorial_interactions"]["latex"]):
        name = cells[0].lower()
        if name.count("\\times") == 2:
            key = "three_way_pp"
        elif "batching" in name and "scaling" in name:
            key = "batching_scaling"
        elif "batching" in name:
            key = "focal_batching"
        else:
            key = "focal_scaling"
        expected_interactions[key] = {"oof": numeric(cells[1]), "kddtest": numeric(cells[2])}
        for partition in ("oof", "kddtest"):
            actual = contrast_output[partition]["architecture_mean"]
            value = actual["three_way_pp"] if key == "three_way_pp" else actual["pairwise_pp"][key]
            assert abs(value - expected_interactions[key][partition]) <= 0.05
    focal_pairs = {cells[0].lower(): {"beta": numeric(cells[1]), "gamma": numeric(cells[2])}
                   for cells in table_rows(tables["tab:focal_parameters"]["latex"])}
    coefficient_rows = [{"architecture": cells[0].lower(), "training_regime": cells[1],
                         "r2l_divisor": numeric(cells[2]), "u2r_divisor": numeric(cells[3])}
                        for cells in table_rows(tables["tab:score_parameters"]["latex"])]
    assert len(coefficient_rows) == 16
    with (OUT / "paper_scaling_coefficients.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coefficient_rows[0]))
        writer.writeheader()
        writer.writerows(coefficient_rows)

    source_files = sorted((ROOT / "src").glob("*.py"))
    stems = {p.stem: p.name for p in source_files}
    dependencies = {p.name: [] for p in source_files}
    edges = []
    trees = {}
    for path in source_files:
        tree = ast.parse(path.read_text())
        trees[path.name] = tree
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                candidates = [node.module.split(".")[0]] if node.module else []
                names = [name.name for name in node.names]
            elif isinstance(node, ast.Import):
                candidates = [alias.name.split(".")[0] for alias in node.names]
                names = candidates
            else:
                continue
            for candidate in candidates:
                if candidate in stems:
                    target = stems[candidate]
                    dependencies[path.name].append(target)
                    edges.append({"source": path.name, "target": target,
                                  "line": node.lineno, "imported_symbols": names})
    roots = {"tune_variant_specific_score_scaling.py", "run_final_baseline_vs_full_kddtest_4gpu.py",
             *[f"tune_{a}_focal_cv_4gpu.py" for a in ARCHITECTURES]}
    closure, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        closure.add(name)
        pending.extend(dependencies[name])
    assert closure == set(CORE), (closure - set(CORE), set(CORE) - closure)
    optional = {}
    for architecture in ARCHITECTURES:
        for kind, mode in (("baseline", "ordinary CE"), ("batch_baseline", "CE+B"), ("focal_batch", "F+B fallback")):
            optional[f"run_{architecture}_{kind}_cv_4gpu.py"] = f"Convenience OOF launcher for {architecture}/{mode}; equivalent generic-engine CLI exists."
        optional[f"tune_{architecture}_baseline_score_scaling_cv_4gpu.py"] = f"Alternative {architecture} ordinary-CE OOF launcher with baseline scaling grid; produces the same baseline pointer family."
        if architecture != "conv2d":
            optional[f"tune_{architecture}_score_scaling_cv_4gpu.py"] = f"Preferred focal+batching OOF source launcher for {architecture}; generic Conv2D-named engine handles this architecture."
    optional["run_final_single_enhancements_kddtest_4gpu.py"] = "Convenience final launcher for focal_only/batch_only/scaling_only; raw focal/batch arrays feed the paper, old scaling_only array is unnecessary."
    assert len(optional) == 20
    file_rows = []
    for path in source_files:
        if path.name in CORE:
            category, reason = "paper_core_or_shared_dependency", CORE[path.name]
        elif path.name in optional:
            category, reason = "paper_optional_or_alternative_launcher", optional[path.name]
        else:
            category = "outside_paper_numerical_pipeline"
            reason = "No import edge from the identified paper engine closure and no paper method needing this entry point. Historical development or separate research; this does not prove it was never run."
        file_rows.append({"path": str(path.relative_to(ROOT)), "category": category,
                          "role_or_reason": reason, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(file_rows) == 57
    with (OUT / "paper_source_classification.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(file_rows[0]))
        writer.writeheader()
        writer.writerows(file_rows)
    frozen = literal_assignment(trees["run_final_baseline_vs_full_kddtest_4gpu.py"], "FROZEN_CONFIG")
    coefficient_comparison = []
    for row in coefficient_rows:
        if row["training_regime"] != "Focal + batching":
            continue
        config = frozen[row["architecture"]]
        old_pair = (config["r2l_score_coefficient"], config["u2r_score_coefficient"])
        new_pair = (row["r2l_divisor"], row["u2r_divisor"])
        coefficient_comparison.append({"architecture": row["architecture"],
                                       "old_final_runner_coefficients": old_pair,
                                       "paper_focal_batch_coefficients": new_pair,
                                       "same": old_pair == new_pair})
        assert focal_pairs[row["architecture"]] == {"beta": config["beta"], "gamma": config["focal_gamma"]}
    outputs = [str(p.relative_to(ROOT)) for p in (ROOT / "results").rglob("*") if p.is_file()]
    modern_artifacts = [p for p in outputs if p.endswith((".npz", ".json")) or "variant_specific" in p or "focal_stage1" in p]
    external_figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", active)
    report = {"paper_path": str(args.paper.resolve()), "paper_sha256": hashlib.sha256(data).hexdigest(),
              "paper_lines": len(original.splitlines()), "paper_tables": {k: {"line": v["line"]} for k, v in tables.items()},
              "parsed_primary_results": parsed_results, "rounded_table_checks": checks,
              "derived_contrasts_from_rounded_cells": contrast_output,
              "printed_interaction_claims": expected_interactions,
              "focal_pairs_match_final_runner": True,
              "old_vs_paper_focal_batch_coefficients": coefficient_comparison,
              "core_source_files": sorted(closure), "optional_launcher_files": sorted(optional),
              "outside_pipeline_files": [Path(row["path"]).name for row in file_rows if row["category"] == "outside_paper_numerical_pipeline"],
              "local_import_edges": edges,
              "modern_result_artifacts_present_in_results": modern_artifacts,
              "external_paper_images": [{"path": p, "present_at_repo_root": (ROOT / p).is_file()} for p in external_figures],
              "scope": "Source dependency/protocol mapping and rounded manuscript arithmetic. Exact executed entry points and raw-prediction result reproduction are not established."}
    (OUT / "paper_code_lineage.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"paper_tables": len(tables), "parsed_primary_cells": len(table_csv_rows),
                      "selected_coefficient_pairs": len(coefficient_rows), "core_source_files": len(closure),
                      "optional_launchers": len(optional), "outside_pipeline": 57-len(closure)-len(optional),
                      "modern_result_artifacts_found": modern_artifacts,
                      "test_contrasts_from_rounded_manuscript": contrast_output["kddtest"]["architecture_mean"]}, indent=2))


if __name__ == "__main__":
    main()
