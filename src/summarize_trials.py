"""
Summarize many `results/*_results.txt` files from cnn_fin.py into a ranked table.

Typical usage (from repo root):
  python -u src/summarize_trials.py --glob "fin_g1_*_results.txt"

Outputs:
  - results/trials_raw.csv      (one row per run)
  - results/trials_summary.csv  (grouped by cb_beta/focal_gamma/groups by default)
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ParsedRun:
    path: Path
    data: Dict[str, Any]


CLASS_LABELS = ["DoS", "Probe", "R2L", "U2R", "normal", "macro avg", "weighted avg"]


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _safe_int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None


def _parse_key_value(lines: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for line in lines:
        # e.g. "cb_beta: 0.9999"
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue

        if key in {
            "focal_gamma",
            "cb_beta",
            "Test Loss",
            "Test Accuracy (keras)",
            "Test Accuracy (sklearn)",
            "MCC",
            "groups",
            "base_filters",
            "dense_units",
        }:
            # numeric fields
            f = _safe_float(val)
            out[key] = f if f is not None else val
        else:
            out[key] = val
    return out


def _parse_classification_report(lines: List[str]) -> Dict[str, Any]:
    """
    Parses sklearn's classification_report text block.
    Returns keys like:
      - DoS_precision, DoS_recall, DoS_f1, DoS_support
      - macro_avg_f1, macro_avg_recall, ...
      - accuracy
    """
    out: Dict[str, Any] = {}

    # Find the "Classification report:" marker if present
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("classification report"):
            start_idx = i
            break

    report_lines = lines[start_idx + 1 :] if start_idx is not None else lines

    for raw in report_lines:
        line = raw.rstrip("\n")
        if not line.strip():
            continue

        toks = line.split()
        if not toks:
            continue

        # accuracy line looks like: "accuracy 0.7588 22544"
        if toks[0] == "accuracy" and len(toks) >= 3:
            acc = _safe_float(toks[1])
            out["accuracy"] = acc
            out["support_total"] = _safe_int(toks[2])
            continue

        # macro avg / weighted avg: 6 tokens
        if len(toks) >= 6 and toks[0] in {"macro", "weighted"} and toks[1] == "avg":
            label = f"{toks[0]} avg"
            prec = _safe_float(toks[2])
            rec = _safe_float(toks[3])
            f1 = _safe_float(toks[4])
            sup = _safe_int(toks[5])

            key_prefix = label.replace(" ", "_")
            out[f"{key_prefix}_precision"] = prec
            out[f"{key_prefix}_recall"] = rec
            out[f"{key_prefix}_f1"] = f1
            out[f"{key_prefix}_support"] = sup
            continue

        # class rows: 5 tokens: label, prec, rec, f1, support
        if len(toks) >= 5 and toks[0] in {"DoS", "Probe", "R2L", "U2R", "normal"}:
            label = toks[0]
            prec = _safe_float(toks[1])
            rec = _safe_float(toks[2])
            f1 = _safe_float(toks[3])
            sup = _safe_int(toks[4])
            out[f"{label}_precision"] = prec
            out[f"{label}_recall"] = rec
            out[f"{label}_f1"] = f1
            out[f"{label}_support"] = sup
            continue

    return out


def _parse_run_name_from_filename(path: Path) -> Dict[str, Any]:
    """
    Best-effort: extract seed from filenames like "..._s7_results.txt" or "...seed7_results.txt".
    """
    stem = path.name
    out: Dict[str, Any] = {"run_name": path.stem.replace("_results", "")}

    m = re.search(r"(?:_s|seed)(\d+)\b", stem)
    if m:
        out["seed_from_name"] = int(m.group(1))
    return out


def parse_results_file(path: Path) -> ParsedRun:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    data: Dict[str, Any] = {}
    data.update(_parse_run_name_from_filename(path))
    data.update(_parse_key_value(lines))
    data.update(_parse_classification_report(lines))

    # Normalize some commonly-used keys for grouping/metrics
    if "cb_beta" in data:
        data["cb_beta"] = float(data["cb_beta"])
    if "focal_gamma" in data:
        data["focal_gamma"] = float(data["focal_gamma"])
    if "groups" in data:
        try:
            data["groups"] = int(float(data["groups"]))
        except Exception:
            pass

    # Convenience aliases
    if "macro_avg_f1" not in data and "macro_avg_f1" in data:
        pass
    # Macro-F1 / macro-recall are the key imbalance-aware metrics
    if "macro_avg_f1" not in data and "macro_avg_f1" in data:
        data["macro_f1"] = data["macro_avg_f1"]
    else:
        data["macro_f1"] = data.get("macro_avg_f1")
    data["macro_recall"] = data.get("macro_avg_recall")
    data["mcc"] = data.get("MCC")

    return ParsedRun(path=path, data=data)


def _mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_results = repo_root / "results"

    parser = argparse.ArgumentParser(description="Summarize many cnn_fin results files into a ranked table.")
    parser.add_argument("--results-dir", type=str, default=str(default_results))
    parser.add_argument("--glob", type=str, default="*_results.txt", help='e.g. "fin_g1_*_results.txt"')
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["cb_beta", "focal_gamma", "groups"],
        help="Fields to aggregate over (mean/std across runs).",
    )
    parser.add_argument("--metric", type=str, default="macro_f1", help="Metric name to rank configs by (mean).")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--out-raw", type=str, default=str(default_results / "trials_raw.csv"))
    parser.add_argument("--out-summary", type=str, default=str(default_results / "trials_summary.csv"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    files = sorted(results_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {results_dir}/{args.glob}")

    parsed = [parse_results_file(p) for p in files]
    raw_rows = [p.data | {"path": str(p.path)} for p in parsed]

    # Collect fieldnames
    fieldnames = sorted({k for r in raw_rows for k in r.keys()})
    _write_csv(Path(args.out_raw), raw_rows, fieldnames)

    # Group and aggregate
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        key = tuple(r.get(f) for f in args.group_by)
        groups.setdefault(key, []).append(r)

    summary_rows: List[Dict[str, Any]] = []
    for key, rows in groups.items():
        row: Dict[str, Any] = {f: v for f, v in zip(args.group_by, key)}
        row["n"] = len(rows)

        # Core metrics to summarize
        for m in [
            "macro_f1",
            "macro_recall",
            "mcc",
            "accuracy",
            "R2L_recall",
            "R2L_f1",
            "U2R_recall",
            "U2R_f1",
            "normal_recall",
        ]:
            vals = [r.get(m) for r in rows]
            vals_f = [v for v in vals if isinstance(v, (int, float))]
            mean, std = _mean_std([float(v) for v in vals_f])
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std

        summary_rows.append(row)

    metric_mean_key = f"{args.metric}_mean"
    summary_rows.sort(key=lambda r: (r.get(metric_mean_key) is not None, r.get(metric_mean_key, -1e9)), reverse=True)

    summary_fieldnames = sorted({k for r in summary_rows for k in r.keys()})
    _write_csv(Path(args.out_summary), summary_rows, summary_fieldnames)

    print(f"Wrote raw runs: {args.out_raw}")
    print(f"Wrote summary:  {args.out_summary}")
    print("")
    print(f"Top {min(args.top, len(summary_rows))} configs by {metric_mean_key}:")

    for i, r in enumerate(summary_rows[: args.top], start=1):
        cb_beta = r.get("cb_beta")
        gamma = r.get("focal_gamma")
        groups_v = r.get("groups")
        n = r.get("n")
        score = r.get(metric_mean_key)
        score_std = r.get(f"{args.metric}_std")
        r2l = r.get("R2L_f1_mean")
        u2r = r.get("U2R_f1_mean")
        print(
            f"{i:>2}. cb_beta={cb_beta} gamma={gamma} groups={groups_v} n={n} "
            f"{args.metric}={score:.4f}±{(score_std or 0.0):.4f}  "
            f"R2L_f1={None if r2l is None else f'{r2l:.3f}'}  U2R_f1={None if u2r is None else f'{u2r:.3f}'}"
        )


if __name__ == "__main__":
    main()


