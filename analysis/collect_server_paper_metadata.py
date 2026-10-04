"""Collect metadata for the identified paper runs, without training or inference.

Run from the server's cnn repository:
  python3 analysis/collect_server_paper_metadata.py --hash-predictions

Creates one archive containing metadata, result CSVs and current source files.
Prediction arrays are optionally checksummed but are not put in the archive.
Existing experiment files are only read. Uses the Python standard library.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import time


SELECTION = "variant_specific_scaling_selection_ea63ca7e718d.json"
EVALUATION = "variant_specific_scaling_kddtest_d5c1c2bb3050_protocol.json"
EXPECTED = {
    SELECTION: "0c73f26079edf9f7cdd1d643551e63bcbb22d9f2d23b76e62aa18500e70da08f",
    EVALUATION: "cb292ebeece383a6d07c1896c9c10d2377b2ba02d5a80a0ec693f62cc5781778",
}
RECORDED_ROOT = Path("/home/gvadzabd/cnn")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--hash-predictions", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output = args.output or Path(tempfile.gettempdir()) / (
        "paper_provenance_ea63ca7e718d_d5c1c2bb3050_"
        + time.strftime("%Y%m%dT%H%M%S")
        + ".tar.gz"
    )
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("Archive already exists: " + str(output))

    files = {}
    predictions = {}

    def resolve_recorded(value):
        path = Path(value)
        if path.is_absolute():
            path = root / path.relative_to(RECORDED_ROOT)
        else:
            path = root / path
        path = path.resolve()
        path.relative_to(root)  # Refuse paths that escape the repository.
        return path

    def add(value, expected=None):
        if not value:
            return
        path = resolve_recorded(value)
        if path in files and files[path] and expected:
            if files[path] != expected:
                raise ValueError("Conflicting recorded hashes: " + str(path))
        files[path] = expected or files.get(path)

    documents = []
    for filename in (SELECTION, EVALUATION):
        path = root / "results" / filename
        actual = sha256(path)
        if actual != EXPECTED[filename]:
            raise ValueError("Frozen manifest hash mismatch: " + str(path))
        documents.append(json.loads(path.read_text()))
        add("results/" + filename, EXPECTED[filename])
    selection, evaluation = documents
    if evaluation["selection_manifest_sha256"] != EXPECTED[SELECTION]:
        raise ValueError("Evaluation references a different frozen selection.")

    for regimes in selection["source_metadata"].values():
        for record in regimes.values():
            for field in ("protocol", "best_config", "latest_pointer"):
                add(record[field], record[field + "_sha256"])
            for source in record["oof_files"].values():
                predictions[resolve_recorded(source["path"])] = source["sha256"]
    for regimes in evaluation["test_source_metadata"].values():
        for seeds in regimes.values():
            for record in seeds.values():
                for field in ("run_metadata_path", "feature_cache_metadata_path"):
                    add(record[field], record[field.replace("_path", "_sha256")])
                add(record["feature_cache_metadata_path"].replace(
                    "_feature_cache.json", "_protocol.json"
                ))
                predictions[resolve_recorded(record["path"])] = record["sha256"]
    for document in documents:
        for name, path in document["outputs"].items():
            if name != "rankings":
                add(path)
    for path in sorted((root / "src").glob("*.py")):
        add(str(path.relative_to(root)))
    for filename in ("file.sh", "file.sh.save", "file.sh.save.1", "gan.sh"):
        if (root / filename).is_file():
            add(filename)

    report = {
        "repository_root": str(root),
        "selection_id": selection["selection_id"],
        "evaluation_id": evaluation["evaluation_id"],
        "collection_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "existing_files_modified": False,
        "training_or_inference_run": False,
        "prediction_arrays_included": False,
        "metadata_files": [],
        "prediction_checks": [],
    }
    for path, expected in sorted(predictions.items()):
        exists = path.is_file()
        actual = sha256(path) if args.hash_predictions and exists else None
        report["prediction_checks"].append({
            "path": str(path.relative_to(root)),
            "exists": exists,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": actual == expected if actual is not None else None,
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for path, expected in sorted(files.items()):
            exists = path.is_file()
            actual = sha256(path) if exists else None
            report["metadata_files"].append({
                "path": str(path.relative_to(root)),
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected if expected and exists else None,
            })
            if exists:
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
        data = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
        member = tarfile.TarInfo("collection_report.json")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    missing = [r["path"] for r in report["metadata_files"] if not r["exists"]]
    mismatched = [r["path"] for r in report["metadata_files"] if r["matches"] is False]
    prediction_mismatches = [r["path"] for r in report["prediction_checks"] if r["matches"] is False]
    print("Archive:", output)
    print("Metadata/source/table files included:", len(files) - len(missing))
    print("Prediction files referenced:", len(predictions))
    print("Missing metadata:", missing)
    print("Metadata hash mismatches:", mismatched)
    print("Prediction hash mismatches:", prediction_mismatches)


if __name__ == "__main__":
    main()
