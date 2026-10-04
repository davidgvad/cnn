"""Adapt isolated reproduction copies for 1–4 GPU workers or one CPU worker.

The original repository is never edited. Fitting, preprocessing, predictions,
metrics and worker-command functions must retain identical source text.
CPU mode additionally relaxes the device check, without changing fitting code.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

MARKER = "resolve_reproduction_cuda_tokens"
RESOLVER = '''def resolve_reproduction_cuda_tokens(gpus: Sequence[str]) -> Dict[str, str]:
    """Resolve logical worker IDs within a parent CUDA allocation."""
    allocated = [token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()]
    try:
        indices = [int(gpu) for gpu in gpus]
    except ValueError:
        return {gpu: gpu for gpu in gpus}
    if allocated and all(0 <= i < len(allocated) for i in indices):
        return {gpu: allocated[i] for gpu, i in zip(gpus, indices, strict=True)}
    return {gpu: gpu for gpu in gpus}


'''
CPU_PREFIX = '''    if os.environ.get("REPRO_ALLOW_CPU") == "1":
        return {gpu: "" for gpu in gpus}
'''
GPU_CHECK = '    if len(visible_gpus) != 1:\n'
CPU_CHECK = '''    if len(visible_gpus) != 1 and not (
        os.environ.get("REPRO_ALLOW_CPU") == "1" and len(visible_gpus) == 0
    ):
'''
CPU_FILES = [
    "tune_conv2d_focal_cv_4gpu.py", "tune_conv1d_focal_cv_4gpu.py",
    "tune_transformer_focal_cv_4gpu.py", "tune_mlp_focal_cv_4gpu.py",
    "tune_conv2d_score_scaling_cv_4gpu.py",
]


def function_sources(source):
    tree = ast.parse(source)
    return {n.name: ast.get_source_segment(source, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def adapt(source):
    if MARKER in function_sources(source):
        return source
    original = source
    replacements = [
        ('help="Exactly four GPU IDs, mapped in order to folds 1..4 (default: 0 1 2 3).",',
         'help="One to four GPU workers; all four folds run serially per assigned GPU.",'),
        ('''    if len(gpus) != FOLD_COUNT:
        parser.error(
            f"This runner maps one fold to each GPU and therefore requires "
            f"exactly {FOLD_COUNT} GPU IDs."
        )
''', '''    if len(gpus) > FOLD_COUNT:
        parser.error(f"At most {FOLD_COUNT} GPU workers are supported.")
    cuda_tokens = resolve_reproduction_cuda_tokens(gpus)
'''),
        ('"assigned_gpu": gpus[fold_id],', '"assigned_gpu": gpus[fold_id % len(gpus)],'),
        ('print(f"GPUs (fold 1..4): {gpus}")', 'print(f"GPU workers: {gpus}; four folds queued across these workers")'),
        ('''    plans_by_fold = {
        fold_id: [plan for plan in pending_plans if int(plan["fold_id"]) == fold_id]
        for fold_id in range(FOLD_COUNT)
    }''', '''    plans_by_gpu = {
        gpu: [plan for plan in pending_plans if plan["assigned_gpu"] == gpu]
        for gpu in gpus
    }'''),
        ('environment["CUDA_VISIBLE_DEVICES"] = gpu', 'environment["CUDA_VISIBLE_DEVICES"] = cuda_tokens[gpu]'),
        ('''    def gpu_worker(gpu: str, fold_id: int) -> None:
        nonlocal completed_counter
        for plan in plans_by_fold[fold_id]:''', '''    def gpu_worker(gpu: str) -> None:
        nonlocal completed_counter
        for plan in plans_by_gpu[gpu]:
            fold_id = int(plan["fold_id"])'''),
        ('''    with ThreadPoolExecutor(max_workers=FOLD_COUNT) as executor:
        futures = [
            executor.submit(gpu_worker, gpus[fold_id], fold_id)
            for fold_id in range(FOLD_COUNT)
        ]''', '''    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(gpu_worker, gpu) for gpu in gpus]'''),
        ('"title": "Conv2D class-balanced focal-loss four-fold tuning",',
         '"title": "Conv2D class-balanced focal-loss four-fold tuning",\n        "runtime_gpu_workers": gpus,\n        "runtime_cuda_mapping": cuda_tokens,\n        "parallel_worker_count": len(gpus),'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError(f"Unexpected Conv2D controller version: anchor count {source.count(before)} for {before[:80]!r}")
        source = source.replace(before, after, 1)
    before = "def stable_hash(value: Any, length: int = 12) -> str:"
    if source.count(before) != 1:
        raise ValueError("Could not locate CUDA resolver insertion point")
    source = source.replace(before, RESOLVER + before, 1)
    original_functions = function_sources(original)
    adapted_functions = function_sources(source)
    preserved = set(original_functions) - {"main", "add_arguments"}
    assert all(original_functions[name] == adapted_functions[name] for name in preserved)
    compile(source, "adapted_conv2d_focal_controller", "exec")
    return source


def adapt_cpu(source):
    original = function_sources(source)
    if source.count(GPU_CHECK) != 1:
        raise ValueError("Unexpected version of the worker GPU check")
    source = source.replace(GPU_CHECK, CPU_CHECK, 1)
    resolver = MARKER if MARKER in original else "resolve_cuda_tokens"
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == resolver)
    first = node.body[0]
    after_docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    insertion = first.end_lineno if after_docstring else node.lineno
    lines = source.splitlines(keepends=True)
    lines.insert(insertion, CPU_PREFIX)
    source = "".join(lines)
    updated = function_sources(source)
    # Reversing these two hardware-only edits must restore each original function.
    for name, text in original.items():
        restored = updated[name].replace(CPU_CHECK.strip(), GPU_CHECK.strip(), 1) if name == "run_training_worker" else updated[name]
        if name == resolver:
            restored = restored.replace(CPU_PREFIX, "", 1)
        if restored != text:
            raise ValueError(f"Unexpected non-device change to {name}")
    compile(source, "cpu_reproduction_controller", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    marker = repo / "fixed_reproduction_origin.json"
    if not marker.is_file():
        parser.error("Only a workspace created by the fixed-parameter launcher may be adapted")
    origin = Path(json.loads(marker.read_text())["source_repository"]).resolve()
    if origin == repo:
        parser.error("Refusing to modify the original repository")
    report_path = repo / "analysis/gpu_scheduling_adaptation.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["cpu_allowed"] != args.allow_cpu:
            raise ValueError("Use a separate reproduction directory when switching between CPU and GPU modes")
        for change in report["changes"]:
            if hashlib.sha256(Path(change["copied_source"]).read_bytes()).hexdigest() != change["adapted_sha256"]:
                raise ValueError("Copied controller has changed since its recorded hardware adaptation")
        print("Isolated reproduction hardware adaptation already verified.")
        return
    names = CPU_FILES if args.allow_cpu else CPU_FILES[:1]
    changes = []
    patches = []
    prepared = []
    for name in names:
        path = repo / "src" / name
        before = path.read_text()
        if before != (origin / "src" / name).read_text():
            raise ValueError(f"Copied source differs from the original before adaptation: {name}")
        after = adapt(before) if name == CPU_FILES[0] else before
        if args.allow_cpu:
            after = adapt_cpu(after)
        functions = function_sources(before)
        allowed = {"main", "add_arguments"} if name == CPU_FILES[0] else set()
        if args.allow_cpu:
            allowed.update({"run_training_worker", "resolve_cuda_tokens"})
        protected = sorted(set(functions) - allowed)
        updated = function_sources(after)
        if any(functions[n] != updated[n] for n in protected):
            raise ValueError(f"Unexpected change to a protected function in {name}")
        changes.append({"copied_source": str(path),
                        "original_sha256": hashlib.sha256(before.encode()).hexdigest(),
                        "adapted_sha256": hashlib.sha256(after.encode()).hexdigest(),
                        "unchanged_functions": {n: hashlib.sha256(functions[n].encode()).hexdigest() for n in protected},
                        "allowed_function_changes": sorted(allowed)})
        patches.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=f"original/{name}", tofile=f"copied/{name}"))
        prepared.append((path, after))
    # Validate every source before writing any copies.
    for path, after in prepared:
        path.write_text(after)
    report = {"original_repository_modified": False, "changes": changes,
              "maximum_gpu_workers": 4, "fold_count": 4, "cpu_allowed": args.allow_cpu,
              "scheduling": "One thread per GPU; each thread executes its assigned folds serially",
              "cuda_mapping": "Logical GPU IDs resolve within inherited CUDA_VISIBLE_DEVICES; CPU mode hides all GPUs",
              "cpu_changes": "Only the CUDA resolver and GPU-presence guard; all fitting code is unchanged" if args.allow_cpu else None}
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    (repo / "analysis/hardware_adaptation.patch").write_text("".join(patches))
    print("Adapted isolated hardware scheduling/device checks; all fitting and data-processing code is unchanged.")


if __name__ == "__main__":
    main()
