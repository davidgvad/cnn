"""Load the paper settings without server-specific paths."""
from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "reproduction/paper_config.json"
ARCHITECTURES = ("conv2d", "conv1d", "transformer", "mlp")
REGIMES = ("baseline", "focal_only", "batch_only", "focal_batch")


def load_config(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    if config["schema_version"] != 1 or config["feature_count"] != 121:
        raise ValueError("Unsupported paper configuration.")
    if config["class_order"] != ["DoS", "Probe", "R2L", "U2R", "Normal"]:
        raise ValueError("Class ordering must match the model probability columns.")
    training = config["training"]
    seeds = training["seeds"]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("At least two unique nonnegative seeds are needed for sample SD.")
    if training["fold_count"] != 4 or training["fold_seed"] < 0:
        raise ValueError("The paper controllers use four stratified folds.")
    for key in ("epochs", "batch_size", "minority_per_batch_per_class"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError(f"Invalid training setting: {key}")
    if set(config["architectures"]) != set(ARCHITECTURES):
        raise ValueError("All four paper backbones are required.")
    for architecture, settings in config["architectures"].items():
        beta, gamma = settings["focal_beta"], settings["focal_gamma"]
        if not 0 < beta < 1 or not math.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"Invalid focal settings: {architecture}")
        if set(settings["score_scaling"]) != set(REGIMES):
            raise ValueError(f"Missing training-regime coefficients: {architecture}")
        for pair in settings["score_scaling"].values():
            if set(pair) != {"r2l", "u2r"} or any(not math.isfinite(v) or v <= 0 for v in pair.values()):
                raise ValueError(f"Invalid score coefficients: {architecture}")
    return config
