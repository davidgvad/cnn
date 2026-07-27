"""
CNN + CTGAN Augmentation + Class-Balanced Focal Loss for NSL-KDD (5-Class).

Design goal
-----------
This script is meant to be a clean, comparable extension of:
- `src/cnn_baseline.py`  (plain cross-entropy)
- `src/cnn_focal.py`     (class-balanced focal loss)

Key additions vs focal-only
---------------------------
- **CTGAN augmentation (data-side)**: generate/load synthetic training rows for minority classes.
- **Adjusted focal weighting**: alpha weights are recomputed from the *augmented* training labels
  (so we don't over-penalize minority classes after oversampling).

Important notes
---------------
- This script never touches the official test split. CTGAN is fit only on KDDTrain+.
- CTGAN requires extra dependencies (PyTorch + CTGAN). If you already have a synthetic CSV,
  you can run without installing CTGAN by using `--synth-path ...` and `--no-ctgan-train`.

Run (from repo root)
-------------------
python3 src/cnn_gan_foc.py --epochs 25

Example (generate synth if missing)
----------------------------------
python3 src/cnn_gan_foc.py --ctgan-train --ctgan-epochs 200 --target-r2l 5000 --target-u2r 5000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_curve,
    auc,
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, label_binarize


NSL_KDD_COLUMNS: List[str] = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "class",
    "difficulty",
]

CLASS_TO_ID: Dict[str, int] = {"DoS": 0, "Probe": 1, "R2L": 2, "U2R": 3, "normal": 4}
ID_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_ID.items()}
CLASS_NAMES: List[str] = [ID_TO_CLASS[i] for i in range(5)]


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    data_dir: Path
    results_dir: Path
    model_dir: Path


def _repo_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[1]
    return Paths(
        repo_root=repo_root,
        data_dir=repo_root / "data",
        results_dir=repo_root / "results",
        model_dir=repo_root / "model",
    )


def load_nsl_kdd_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = NSL_KDD_COLUMNS
    df = df.drop(columns=["difficulty"])
    return df


def collapse_attack_labels(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()

    # Fix binary feature: requested to map 2 -> 1
    df["su_attempted"] = df["su_attempted"].replace(2, 1)

    if is_train:
        df["class"] = df["class"].replace(
            ["neptune", "smurf", "back", "teardrop", "pod", "land"], "DoS"
        )
        df["class"] = df["class"].replace(
            ["satan", "ipsweep", "portsweep", "nmap"], "Probe"
        )
        df["class"] = df["class"].replace(
            [
                "warezclient",
                "guess_passwd",
                "warezmaster",
                "imap",
                "ftp_write",
                "multihop",
                "phf",
                "spy",
            ],
            "R2L",
        )
        df["class"] = df["class"].replace(
            ["buffer_overflow", "rootkit", "loadmodule", "perl"], "U2R"
        )
    else:
        df["class"] = df["class"].replace(
            [
                "neptune",
                "apache2",
                "processtable",
                "smurf",
                "back",
                "mailbomb",
                "pod",
                "teardrop",
                "land",
                "udpstorm",
            ],
            "DoS",
        )
        df["class"] = df["class"].replace(
            ["mscan", "satan", "saint", "portsweep", "ipsweep", "nmap"], "Probe"
        )
        df["class"] = df["class"].replace(
            [
                "guess_passwd",
                "warezmaster",
                "snmpguess",
                "snmpgetattack",
                "httptunnel",
                "multihop",
                "named",
                "sendmail",
                "xlock",
                "xsnoop",
                "ftp_write",
                "worm",
                "phf",
                "imap",
            ],
            "R2L",
        )
        df["class"] = df["class"].replace(
            ["buffer_overflow", "ps", "rootkit", "xterm", "loadmodule", "perl", "sqlattack"],
            "U2R",
        )

    unknown = sorted(set(df["class"].unique()) - set(CLASS_TO_ID.keys()))
    if unknown:
        raise ValueError(f"Found unmapped class labels: {unknown}")

    df["class"] = df["class"].map(CLASS_TO_ID).astype(int)
    return df


def fit_one_hot_encoders(
    train_df: pd.DataFrame, categorical_columns: List[str]
) -> Tuple[Dict[str, OneHotEncoder], Dict[str, List[str]]]:
    encoders: Dict[str, OneHotEncoder] = {}
    feature_names: Dict[str, List[str]] = {}

    for col in categorical_columns:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        enc.fit(train_df[[col]])
        encoders[col] = enc
        feature_names[col] = enc.get_feature_names_out([col]).tolist()

    return encoders, feature_names


def apply_one_hot(
    df: pd.DataFrame,
    encoders: Dict[str, OneHotEncoder],
    feature_names: Dict[str, List[str]],
    categorical_columns: List[str],
) -> pd.DataFrame:
    out = df.copy()
    for col in categorical_columns:
        enc = encoders[col]
        encoded = enc.transform(out[[col]])
        cols = feature_names[col]
        encoded_df = pd.DataFrame(encoded, columns=cols, index=out.index)
        out = out.drop(columns=[col]).join(encoded_df)
    return out


def fit_scaler(train_df: pd.DataFrame, columns_to_scale: List[str]) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_df[columns_to_scale])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, columns_to_scale: List[str]) -> pd.DataFrame:
    out = df.copy()
    out[columns_to_scale] = scaler.transform(out[columns_to_scale])
    return out


def prepare_xy_from_processed(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_train = train_df["class"].to_numpy(dtype=np.int64)
    y_test = test_df["class"].to_numpy(dtype=np.int64)

    X_train = train_df.drop(columns=["class"]).to_numpy(dtype=np.float32)
    X_test = test_df.drop(columns=["class"]).to_numpy(dtype=np.float32)

    if X_train.shape[1] != 121 or X_test.shape[1] != 121:
        raise ValueError(
            f"Expected 121 features (11x11). Got train={X_train.shape[1]} test={X_test.shape[1]}."
        )

    X_train = X_train.reshape(-1, 11, 11, 1)
    X_test = X_test.reshape(-1, 11, 11, 1)
    return X_train, y_train, X_test, y_test


def compute_cb_alpha_effective_number(
    y_train: np.ndarray,
    beta: float,
    num_classes: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float64)
    beta = float(beta)
    if not (0.0 < beta < 1.0):
        raise ValueError("--cb-beta must be in (0, 1). Typical values: 0.99, 0.999, 0.9999")

    effective = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective, 1e-12)
    weights = np.where(counts > 0, weights, 0.0)

    if weights.sum() <= 0:
        weights = np.ones(num_classes, dtype=np.float64)

    weights = weights / weights.sum() * num_classes
    return weights.astype(np.float32), counts.astype(np.int64)


@tf.keras.utils.register_keras_serializable(package="cnn")
class ClassBalancedFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha: np.ndarray | List[float],
        gamma: float = 2.0,
        reduction: str | tf.keras.losses.Reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        name: str = "class_balanced_focal_loss",
    ) -> None:
        super().__init__(name=name, reduction=reduction)
        alpha_arr = np.asarray(alpha, dtype=np.float32)
        if alpha_arr.ndim != 1:
            raise ValueError("alpha must be a 1D array of per-class weights")
        self.alpha = tf.constant(alpha_arr, dtype=tf.float32)
        self.gamma = float(gamma)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)

        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

        batch_indices = tf.range(tf.shape(y_true)[0], dtype=tf.int32)
        gather_idx = tf.stack([batch_indices, y_true], axis=1)
        p_t = tf.gather_nd(y_pred, gather_idx)

        alpha_t = tf.gather(self.alpha, y_true)
        focal = tf.pow(1.0 - p_t, self.gamma)
        return -alpha_t * focal * tf.math.log(p_t)

    def get_config(self) -> Dict[str, object]:
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha.numpy().tolist(),
                "gamma": self.gamma,
                "name": self.name,
            }
        )
        return config


def build_cnn(loss: tf.keras.losses.Loss) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(11, 11, 1))
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_gan_foc")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def save_confusion_matrices(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred, normalize=None)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_confusion_matrix.png", dpi=400)
    plt.close()

    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_normalized_confusion_matrix.png", dpi=400)
    plt.close()


def save_multiclass_roc(y_true: np.ndarray, y_proba: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true_onehot = label_binarize(y_true, classes=list(range(5)))
    fpr: Dict[int, np.ndarray] = {}
    tpr: Dict[int, np.ndarray] = {}
    roc_auc: Dict[int, float] = {}

    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ["blue", "red", "green", "orange", "purple"]
    for i, color in zip(range(5), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_multi_class_auc.png", dpi=400)
    plt.close()


def save_training_plot(history: tf.keras.callbacks.History, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(history.history.get("accuracy", []), label="Training Accuracy")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_training_plot.png", dpi=400)
    plt.close()


def _maybe_load_synth(synth_path: Path) -> pd.DataFrame | None:
    if not synth_path.exists():
        return None
    df = pd.read_csv(synth_path)
    return df


def _generate_synth_ctgan(
    train_df_raw: pd.DataFrame,
    seed: int,
    epochs: int,
    batch_size: int,
    pac: int,
    targets: Dict[int, int],
) -> pd.DataFrame:
    """
    Train CTGAN on the *raw* (pre-OHE) training dataframe and conditionally sample rows per class.

    Requires:
      pip install ctgan torch
    """
    try:
        from ctgan import CTGAN  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "CTGAN dependency missing. Install with: `pip install ctgan torch` "
            "or run with an existing --synth-path and omit --ctgan-train."
        ) from e

    # CTGAN treats these as discrete/categorical.
    discrete_columns = ["protocol_type", "service", "flag", "class"]

    # Best-effort seeding + auto device selection (works whether torch is CPU-only or CUDA-enabled).
    np.random.seed(seed)
    cuda = False
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False

    pac = int(pac)
    if pac <= 0:
        raise ValueError("--ctgan-pac must be >= 1")

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("--ctgan-batch-size must be >= 1")

    # CTGAN uses "PacGAN" style packing in the discriminator; it asserts batch_size % pac == 0.
    batch_size_adj = batch_size - (batch_size % pac)
    if batch_size_adj <= 0:
        batch_size_adj = pac
    if batch_size_adj != batch_size:
        print(f"CTGAN: adjusting batch_size {batch_size} -> {batch_size_adj} to satisfy batch_size % pac == 0 (pac={pac})")
    batch_size = batch_size_adj

    try:
        import inspect

        kwargs = {"epochs": epochs, "batch_size": batch_size, "verbose": True}
        sig = inspect.signature(CTGAN)
        if "cuda" in sig.parameters:
            kwargs["cuda"] = cuda
        if "pac" in sig.parameters:
            kwargs["pac"] = pac
        if "random_state" in sig.parameters:
            kwargs["random_state"] = seed
        ctgan = CTGAN(**kwargs)
    except Exception:
        # Fallback for older/newer ctgan variants.
        try:
            ctgan = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True, cuda=cuda, pac=pac)
        except TypeError:
            ctgan = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True, cuda=cuda)

    ctgan.fit(train_df_raw, discrete_columns=discrete_columns)

    real_counts = train_df_raw["class"].value_counts().to_dict()
    synth_parts: List[pd.DataFrame] = []
    for cls_id, target in targets.items():
        need = max(0, int(target) - int(real_counts.get(cls_id, 0)))
        if need <= 0:
            continue
        cls_id = int(cls_id)
        # Different ctgan versions expose conditional sampling with different APIs.
        try:
            part = ctgan.sample(need, condition_column="class", condition_value=cls_id)
        except TypeError:
            try:
                part = ctgan.sample(need, conditions={"class": cls_id})
            except TypeError:
                # Last-resort fallback: unconditional sampling then filter.
                # (May yield fewer rows than requested if the generator doesn't respect class well.)
                part = ctgan.sample(max(need * 5, need))
                part = part[part.get("class") == cls_id].head(need)

        # IMPORTANT: We requested conditional samples for `class == cls_id`.
        # Some CTGAN versions/models may output a noisy `class` column; for downstream supervised
        # learning, we treat the condition as the label and overwrite it to ensure the intended
        # class distribution in the synthetic dataset.
        if "class" in part.columns:
            match_rate = float((part["class"] == cls_id).mean()) if len(part) else 0.0
            print(f"CTGAN: condition class={cls_id}, generated_rows={len(part)}, class_match_rate={match_rate:.3f}")
        part["class"] = cls_id
        synth_parts.append(part)

    if not synth_parts:
        return pd.DataFrame(columns=train_df_raw.columns)

    synth_df = pd.concat(synth_parts, ignore_index=True)

    # Basic cleanup: enforce int class ids and filter unknowns.
    synth_df["class"] = pd.to_numeric(synth_df["class"], errors="coerce").round().astype("Int64")
    synth_df = synth_df[synth_df["class"].isin(list(range(5)))]
    synth_df["class"] = synth_df["class"].astype(int)
    return synth_df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CNN + CTGAN augmentation + focal loss (NSL-KDD 5-class).")

    # CNN training knobs
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)

    # Focal knobs
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--cb-beta", type=float, default=0.9999)

    # CTGAN knobs
    parser.add_argument("--synth-path", type=str, default="data/synth_ctgan_5class.csv")
    parser.add_argument("--ctgan-train", action="store_true", help="Train CTGAN and generate synth if synth-path missing.")
    parser.add_argument("--ctgan-regenerate", action="store_true", help="Force regeneration even if synth-path exists.")
    parser.add_argument(
        "--ctgan-only",
        action="store_true",
        help="Generate/load CTGAN synthetic CSV then exit (skip CNN training). Useful to precompute synth once.",
    )
    parser.add_argument("--ctgan-epochs", type=int, default=200)
    parser.add_argument("--ctgan-batch-size", type=int, default=4096)
    parser.add_argument("--ctgan-pac", type=int, default=10, help="CTGAN discriminator packing factor; requires ctgan-batch-size divisible by this.")
    parser.add_argument("--target-dos", type=int, default=0)
    parser.add_argument("--target-probe", type=int, default=0)
    parser.add_argument("--target-r2l", type=int, default=5000)
    parser.add_argument("--target-u2r", type=int, default=5000)
    parser.add_argument("--target-normal", type=int, default=0)

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    paths = _repo_paths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_nsl_kdd_txt(paths.data_dir / "KDDTrain+.txt")
    test_df = load_nsl_kdd_txt(paths.data_dir / "KDDTest+.txt")

    # Drop constant all-zero feature (matches thesis code)
    train_df = train_df.drop(columns=["num_outbound_cmds"])
    test_df = test_df.drop(columns=["num_outbound_cmds"])

    train_df = collapse_attack_labels(train_df, is_train=True)
    test_df = collapse_attack_labels(test_df, is_train=False)

    if args.max_train_samples and args.max_train_samples > 0:
        train_df = train_df.iloc[: args.max_train_samples].copy()
    if args.max_test_samples and args.max_test_samples > 0:
        test_df = test_df.iloc[: args.max_test_samples].copy()

    synth_path = (paths.repo_root / args.synth_path).resolve()
    synth_df = None if args.ctgan_regenerate else _maybe_load_synth(synth_path)

    targets = {
        0: args.target_dos,
        1: args.target_probe,
        2: args.target_r2l,
        3: args.target_u2r,
        4: args.target_normal,
    }

    if synth_df is None:
        if args.ctgan_train:
            synth_df = _generate_synth_ctgan(
                train_df_raw=train_df,
                seed=args.seed,
                epochs=args.ctgan_epochs,
                batch_size=args.ctgan_batch_size,
                pac=args.ctgan_pac,
                targets=targets,
            )
            synth_path.parent.mkdir(parents=True, exist_ok=True)
            synth_df.to_csv(synth_path, index=False)
        else:
            synth_df = pd.DataFrame(columns=train_df.columns)

    if args.ctgan_only:
        if (not args.ctgan_train) and (not synth_path.exists()):
            raise ValueError(
                f"--ctgan-only was set but no synth file exists at {synth_path}. "
                "Run with --ctgan-train (or point --synth-path to an existing CSV)."
            )
        synth_counts_only = (
            np.bincount(synth_df["class"].to_numpy(dtype=np.int64), minlength=5)
            if len(synth_df)
            else np.zeros(5, dtype=np.int64)
        )
        print("\n=== CTGAN precompute complete (ctgan-only) ===")
        print("Synth path:", synth_path)
        print("Targets:", targets)
        print("Synth class counts:", synth_counts_only)
        return

    # Combine real + synth (training only)
    train_aug_df = pd.concat([train_df, synth_df], ignore_index=True)
    train_aug_df = train_aug_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Preprocessing: fit OHE + scaler on REAL train only to preserve the 121-feature (11x11) layout.
    categorical_columns = ["protocol_type", "service", "flag"]
    columns_to_scale = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "count",
        "srv_count",
        "dst_host_count",
        "dst_host_srv_count",
    ]

    encoders, feature_names = fit_one_hot_encoders(train_df, categorical_columns)
    scaler = fit_scaler(apply_one_hot(train_df, encoders, feature_names, categorical_columns), columns_to_scale)

    train_proc = apply_one_hot(train_aug_df, encoders, feature_names, categorical_columns)
    test_proc = apply_one_hot(test_df, encoders, feature_names, categorical_columns)

    train_proc = apply_scaler(train_proc, scaler, columns_to_scale)
    test_proc = apply_scaler(test_proc, scaler, columns_to_scale)

    # Align column order (safety)
    test_proc = test_proc[train_proc.columns]

    X_train, y_train, X_test, y_test = prepare_xy_from_processed(train_proc, test_proc)

    # Focal alpha computed on *augmented* labels (this is the "adjustment" after CTGAN).
    alpha, counts_aug = compute_cb_alpha_effective_number(y_train, beta=args.cb_beta, num_classes=5)
    loss_obj = ClassBalancedFocalLoss(alpha=alpha, gamma=args.focal_gamma)

    prefix = "gan_foc"

    model = build_cnn(loss=loss_obj)
    checkpoint_path = paths.model_dir / f"{prefix}_best.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        verbose=1,
        callbacks=callbacks,
    )

    save_training_plot(history, paths.results_dir, prefix)

    if checkpoint_path.exists():
        model = tf.keras.models.load_model(
            checkpoint_path,
            custom_objects={
                "ClassBalancedFocalLoss": ClassBalancedFocalLoss,
                "cnn>ClassBalancedFocalLoss": ClassBalancedFocalLoss,
            },
        )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    report = classification_report(y_test, y_pred, digits=4, target_names=CLASS_NAMES)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc_sklearn = accuracy_score(y_test, y_pred)

    save_confusion_matrices(y_test, y_pred, paths.results_dir, prefix)
    save_multiclass_roc(y_test, y_proba, paths.results_dir, prefix)

    out_txt = paths.results_dir / f"{prefix}_results.txt"
    real_counts = np.bincount(train_df["class"].to_numpy(dtype=np.int64), minlength=5)
    synth_counts = np.bincount(synth_df["class"].to_numpy(dtype=np.int64), minlength=5) if len(synth_df) else np.zeros(5, dtype=np.int64)
    synth_used = bool(int(synth_counts.sum()) > 0)

    if not synth_used:
        print(
            "\nWARNING: Synth class counts are all zero. No CTGAN data was loaded/generated, "
            "so this run is effectively *focal-only* (not GAN+focal)."
        )

    with out_txt.open("w", encoding="utf-8") as f:
        f.write("CNN + CTGAN Augmentation + Class-Balanced Focal Loss (5-Class)\n")
        f.write(f"synth_path: {synth_path}\n")
        f.write(f"ctgan_train: {bool(args.ctgan_train)}\n")
        f.write(f"synth_used: {synth_used}\n")
        f.write(f"targets: {targets}\n\n")
        f.write(f"focal_gamma: {args.focal_gamma}\n")
        f.write(f"cb_beta: {args.cb_beta}\n")
        f.write(f"class_counts_real: {real_counts.tolist()}\n")
        f.write(f"class_counts_synth: {synth_counts.tolist()}\n")
        f.write(f"class_counts_aug: {counts_aug.tolist()}\n")
        f.write(f"alpha (aug-derived, normalized): {[float(x) for x in alpha.tolist()]}\n\n")
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy (keras): {acc}\n")
        f.write(f"Test Accuracy (sklearn): {acc_sklearn}\n")
        f.write(f"MCC: {mcc}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    print("\n=== GAN + Focal Test Results ===")
    print("Real train class counts:", real_counts)
    print("Synth class counts:", synth_counts)
    print("Aug train class counts:", counts_aug)
    print("Alpha weights:", alpha)
    print(f"Test loss: {loss:.6f}")
    print(f"Test accuracy: {acc:.6f}")
    print(f"MCC: {mcc:.6f}")
    print(report)
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()


