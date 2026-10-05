"""Calculate paper metrics and factorial contrasts from class counts."""
from __future__ import annotations

import csv
import hashlib
import math
import statistics

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
