"""CPU/GPU workers and paths for the paper experiments.

Each worker fits a complete model. Worker count changes concurrency without
changing batches, folds, seeds, loss, or optimizer settings. Device discovery
runs separately so the controller does not reserve a GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


@dataclass(frozen=True)
class Runtime:
    device: str
    workers: tuple[str, ...]
    cuda_tokens: dict[str, str]
    visibility_checked: bool

    def metadata(self):
        return {"device": self.device, "workers": list(self.workers),
                "cuda_mapping": self.cuda_tokens, "parallel_workers": len(self.workers),
                "visibility_checked": self.visibility_checked}


def add_arguments(parser: argparse.ArgumentParser, *, paths=True):
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto",
                        help="Automatically use visible GPUs, or explicitly choose CPU/GPU.")
    parser.add_argument("--gpus", "--gpu-ids", nargs="+", default=None,
                        help="Logical GPU IDs within CUDA_VISIBLE_DEVICES; defaults to all visible GPUs.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Maximum independent fits at once; CPU default is one.")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Compatibility alias for --device cpu.")
    if paths:
        parser.add_argument("--data-dir", type=Path, default=None)
        parser.add_argument("--results-dir", type=Path, default=None)


def visible_gpu_count() -> int:
    code = "import json,tensorflow as tf; print(json.dumps(len(tf.config.list_physical_devices('GPU'))))"
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=30, check=True)
        return int(json.loads(result.stdout.strip().splitlines()[-1]))
    except (subprocess.SubprocessError, ValueError, IndexError) as error:
        raise RuntimeError("TensorFlow device discovery failed. Check the installed experiment dependencies.") from error


def parse_gpu_ids(values: Sequence[str] | None) -> list[str]:
    tokens = [p.strip() for value in (values or ()) for p in value.split(",") if p.strip()]
    if len(tokens) != len(set(tokens)):
        raise ValueError("GPU IDs must be unique.")
    return tokens


def resolve_cuda_tokens(gpus: Sequence[str]) -> dict[str, str]:
    allocated = [p.strip() for p in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if p.strip()]
    mapping = {}
    for gpu in gpus:
        if gpu.isdecimal() and allocated and int(gpu) < len(allocated):
            mapping[gpu] = allocated[int(gpu)]
        else:
            mapping[gpu] = gpu
    return mapping


def select_runtime(args, *, gpu_count=None) -> Runtime:
    """Explicit GPU dry runs can be planned on a machine with no GPUs."""
    if args.workers is not None and args.workers < 1:
        raise ValueError("--workers must be positive.")
    requested = parse_gpu_ids(args.gpus)
    device = args.device
    if args.allow_cpu:
        if device == "gpu":
            raise ValueError("--allow-cpu conflicts with --device gpu.")
        device = "cpu"
    if device == "cpu":
        if requested:
            raise ValueError("--gpus cannot be combined with CPU execution.")
        workers = tuple(f"cpu{i}" for i in range(args.workers or 1))
        return Runtime("cpu", workers, {w: "" for w in workers}, False)
    dry_explicit = bool(getattr(args, "dry_run", False) and requested)
    count = gpu_count if gpu_count is not None else (None if dry_explicit else visible_gpu_count())
    if not requested and count == 0:
        if device == "gpu":
            raise ValueError("No TensorFlow GPU is visible. Choose --device cpu or check CUDA_VISIBLE_DEVICES.")
        workers = tuple(f"cpu{i}" for i in range(args.workers or 1))
        return Runtime("cpu", workers, {w: "" for w in workers}, True)
    gpus = requested or [str(i) for i in range(count or 0)]
    if count is not None:
        allocated = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        for gpu in gpus:
            if gpu.isdecimal():
                valid = 0 <= int(gpu) < count
            else:
                valid = gpu in allocated
            if not valid:
                raise ValueError(f"GPU {gpu!r} is outside the {count} visible TensorFlow devices. Use logical IDs starting at zero.")
    if not gpus:
        raise ValueError("GPU execution requires at least one visible GPU.")
    if args.workers is not None:
        if args.workers > len(gpus):
            raise ValueError("GPU workers cannot exceed the selected GPU count.")
        gpus = gpus[:args.workers]
    return Runtime("gpu", tuple(gpus), resolve_cuda_tokens(gpus), count is not None)


def configure_controller(args, parser):
    try:
        runtime = select_runtime(args)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    args.allow_cpu = runtime.device == "cpu"
    return runtime


def validate_worker_devices(visible_gpus, *, allow_cpu: bool):
    expected = 0 if allow_cpu else 1
    if len(visible_gpus) != expected:
        raise RuntimeError(f"This worker expects {expected} visible GPUs; TensorFlow sees {len(visible_gpus)}: {visible_gpus}.")


def data_directory(args, repo_root: Path) -> Path:
    return (args.data_dir or repo_root / "data").expanduser().resolve()


def results_directory(args, repo_root: Path) -> Path:
    return (args.results_dir or repo_root / "results").expanduser().resolve()
