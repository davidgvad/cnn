"""Read-only, standard-library checks for the reviewed research checkout.

Run: python3 analysis/audit_repository.py
Writes audit artifacts in analysis/; does not import training code or run models.
The inventory excludes .git and this analysis directory to preserve review scope.
"""
from __future__ import annotations

import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def png_chunks(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    while offset < len(data):
        size = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + size]
        crc = struct.unpack(">I", data[offset + 8 + size:offset + 12 + size])[0]
        assert zlib.crc32(kind + payload) & 0xffffffff == crc
        yield kind, payload
        offset += size + 12
    assert offset == len(data)


def decode_rgba(path):
    chunks = list(png_chunks(path.read_bytes()))
    header = next(payload for kind, payload in chunks if kind == b"IHDR")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", header)
    assert (depth, color, compression, filtering, interlace) == (8, 6, 0, 0, 0)
    raw = zlib.decompress(b"".join(payload for kind, payload in chunks if kind == b"IDAT"))
    stride = width * 4
    assert len(raw) == height * (stride + 1)
    previous = bytearray(stride)
    rows = []
    for y in range(height):
        start = y * (stride + 1)
        mode = raw[start]
        row = bytearray(raw[start + 1:start + 1 + stride])
        for x in range(stride):
            a = row[x - 4] if x >= 4 else 0
            b = previous[x]
            c = previous[x - 4] if x >= 4 else 0
            if mode == 1:
                predictor = a
            elif mode == 2:
                predictor = b
            elif mode == 3:
                predictor = (a + b) // 2
            elif mode == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            else:
                assert mode == 0
                predictor = 0
            row[x] = (row[x] + predictor) & 255
        rows.append(row)
        previous = row
    return width, height, rows


def write_png(path, width, height, rows):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff)
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(b"".join(b"\0" + bytes(row) for row in rows)))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts and "analysis" not in p.relative_to(ROOT).parts)
    inventory, source_map, csv_details, model_configs = [], [], {}, {}
    failures = []
    for path in files:
        rel = str(path.relative_to(ROOT))
        data = path.read_bytes()
        item = {"path": rel, "bytes": len(data), "sha256": sha(data), "kind": path.suffix, "details": ""}
        if path.suffix == ".py":
            source = data.decode("utf-8")
            try:
                compile(source, rel, "exec")
                tree = ast.parse(source)
                definitions = [{"name": node.name, "line": node.lineno, "end_line": node.end_lineno} for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))]
                tests = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
                imports = [ast.get_source_segment(source, n) for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
                source_map.append({"path": rel, "lines": len(source.splitlines()), "docstring": ast.get_docstring(tree), "definitions": definitions, "imports": imports, "tests": tests})
                item["details"] = "syntax PASS; %d lines; %d test methods" % (len(source.splitlines()), len(tests))
            except Exception as error:
                failures.append({"path": rel, "error": str(error)})
        elif path.suffix == ".png":
            chunks = list(png_chunks(data))
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", chunks[0][1])
            item["details"] = f"PNG CRC PASS; {width}x{height}; depth={depth}; color={color}; interlace={interlace}"
        elif path.suffix == ".pyc":
            item["details"] = "bytecode header only; magic=%s; flags=%d; not executed" % (data[:4].hex(), struct.unpack("<I", data[4:8])[0])
        elif path.suffix == ".csv":
            rows = list(csv.reader(data.decode("utf-8-sig").splitlines()))
            details = {"data_rows": len(rows) - 1, "header": rows[0], "width_counts": dict(collections.Counter(len(row) for row in rows))}
            if path.parent.name == "data":
                details["class_counts"] = dict(collections.Counter(row[-1] for row in rows[1:]))
            else:
                details["column_unique_counts"] = {name: len(set(row[i] for row in rows[1:])) for i, name in enumerate(rows[0])}
            csv_details[rel] = details
            item["details"] = f"all CSV rows read; {len(rows)-1} data rows; {len(rows[0])} columns"
        elif path.suffix == ".h5":
            assert data[:8] == b"\x89HDF\r\n\x1a\n"
            for key in (b'{"class_name"', b'{"loss"'):
                position = data.find(key)
                if position >= 0:
                    obj, _ = json.JSONDecoder().raw_decode(data[position:].decode("utf-8", "replace"))
                    model_configs[key.decode()] = obj
            item["details"] = "HDF5 signature and embedded model/training JSON inspected; weights not executed"
        else:
            try:
                text = data.decode("utf-8")
                item["details"] = f"text read; {len(text.splitlines())} lines"
            except UnicodeDecodeError:
                item["details"] = "binary hash and length inspected"
        inventory.append(item)
    with (OUT / "file_inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    (OUT / "source_map.json").write_text(json.dumps(source_map, indent=2) + "\n")
    (OUT / "csv_audit.json").write_text(json.dumps(csv_details, indent=2) + "\n")
    (OUT / "saved_model_config.json").write_text(json.dumps(model_configs, indent=2) + "\n")

    core_tree = ast.parse((ROOT / "src/run_no_ctgan_model_ablation_4gpu.py").read_text())
    loader = next(n for n in core_tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_collapsed_nsl_kdd")
    branch = next(n for n in loader.body if isinstance(n, ast.If))
    maps = [ast.literal_eval(branch.body[0].value), ast.literal_eval(branch.orelse[0].value)]
    dataset = {}
    feature_sets = []
    for name, mapping in zip(("KDDTrain+", "KDDTest+"), maps):
        rows = list(csv.reader((ROOT / "data" / (name + ".txt")).open()))
        labels = {raw: group for group, subtypes in mapping.items() for raw in subtypes}
        groups = collections.defaultdict(list)
        for row in rows:
            groups[tuple(row[:41])].append(row[41])
        dataset[name] = {"rows": len(rows), "width_counts": dict(collections.Counter(len(r) for r in rows)), "class_counts": dict(collections.Counter(labels.get(r[41], r[41]) for r in rows)), "subtype_counts": dict(collections.Counter(r[41] for r in rows)), "duplicate_feature_rows_beyond_first": sum(len(values)-1 for values in groups.values()), "conflicting_raw_label_feature_groups": sum(len(set(values)) > 1 for values in groups.values()), "sha256": sha((ROOT / "data" / (name + ".txt")).read_bytes())}
        dataset[name]["conflicting_collapsed_class_feature_groups"] = sum(len({labels.get(v, v) for v in values}) > 1 for values in groups.values())
        dataset[name]["categorical_cardinalities"] = {str(i): len({r[i] for r in rows}) for i in (1, 2, 3)}
        dataset[name]["nonfinite_numeric_values"] = sum(not math.isfinite(float(r[i])) for r in rows for i in range(41) if i not in (1, 2, 3))
        dataset[name]["constant_raw_numeric_features"] = [i for i in range(41) if i not in (1, 2, 3) and len({r[i] for r in rows}) == 1]
        feature_sets.append(set(groups))
        if name == "KDDTest+":
            overlap = [r for r in rows if tuple(r[:41]) in feature_sets[0]]
            dataset["train_test_feature_overlap"] = {"unique_vectors": len(feature_sets[0] & feature_sets[1]), "test_rows": len(overlap), "test_subtypes": dict(collections.Counter(r[41] for r in overlap))}
    dataset["test_only_attack_subtypes"] = sorted(set(dataset["KDDTest+"]["subtype_counts"]) - set(dataset["KDDTrain+"]["subtype_counts"]))
    (OUT / "dataset_audit.json").write_text(json.dumps(dataset, indent=2) + "\n")

    sample_paths = []
    for group in ("train", "test", "synth"):
        for i in range(21):
            name = f"synth_{i}.png" if group == "synth" else f"{i}.png"
            sample_paths.append(ROOT / "images" / group / name)
    tile, gap, columns = 110, 5, 7
    width, height = columns*(tile+gap)+gap, 9*(tile+gap)+gap
    canvas = [bytearray([230, 234, 239, 255]*width) for _ in range(height)]
    for index, path in enumerate(sample_paths):
        w, h, pixels = decode_rgba(path)
        x0, y0 = gap+(index % columns)*(tile+gap), gap+(index//columns)*(tile+gap)
        for y in range(tile):
            source_row = pixels[min(h-1, y*h//tile)]
            for x in range(tile):
                source_x = min(w-1, x*w//tile)*4
                canvas[y0+y][(x0+x)*4:(x0+x+1)*4] = source_row[source_x:source_x+4]
    write_png(OUT / "sample_images_contact_sheet.png", width, height, canvas)
    (OUT / "sample_images_contact_sheet_order.txt").write_text("Read left to right, top to bottom: train 0..20, test 0..20, synth 0..20. Each group occupies three rows.\n")
    summary = {"review_scope_files": len(files), "bytes": sum(item["bytes"] for item in inventory), "extension_counts": dict(collections.Counter(p.suffix for p in files)), "python_source_lines": sum(item["lines"] for item in source_map), "python_modules": len(source_map), "test_methods": sum(len(item["tests"]) for item in source_map), "syntax_failures": failures, "png_crc_checked": sum(p.suffix == ".png" for p in files), "sample_images_decoded": len(sample_paths), "runtime_training": "not run; numerical dependencies absent in available interpreter"}
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
