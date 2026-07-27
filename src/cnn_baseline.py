"""
Baseline CNN (NO CTGAN) for NSL-KDD 5-Class Intrusion Detection.

What this script does
---------------------
- Loads NSL-KDD `KDDTrain+.txt` and `KDDTest+.txt`
- Collapses fine-grained attacks into 5 super-classes: DoS / Probe / R2L / U2R / normal
- One-hot encodes: protocol_type, service, flag  (fit on train, transform test)
- MinMax scales select numeric columns (fit on train, transform test)
- Reshapes 121 features into 11x11 grayscale "images" (no PNGs are written/loaded)
- Trains a small CNN with standard cross-entropy (plain baseline objective)
- Evaluates on the official test split and saves plots + a text report into `results/`

Run (from repo root)
-------------------
python3 src/cnn_baseline.py --epochs 25
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")  # safe for headless runs; plots are saved to disk

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
    # Drop difficulty: not used in this project
    df = df.drop(columns=["difficulty"])
    return df


def collapse_attack_labels(df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    """
    Map fine-grained NSL-KDD attack names into 5 super-classes.
    Mirrors the mapping in `src/thesis_ID_5-class.py`.
    """
    df = df.copy()

    # Fix binary feature: requested to map 2 -> 1
    df["su_attempted"] = df["su_attempted"].replace(2, 1)

    # Train/test differ slightly in which attack names appear.
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

    # Anything not mapped is either "normal" or already one of the super-classes.
    # Enforce the expected label set.
    unknown = sorted(set(df["class"].unique()) - set(CLASS_TO_ID.keys()))
    if unknown:
        # This usually means new/typo attack names that weren't covered above.
        raise ValueError(f"Found unmapped class labels: {unknown}")

    df["class"] = df["class"].map(CLASS_TO_ID).astype(int)
    return df


def one_hot_encode(
    train_df: pd.DataFrame, test_df: pd.DataFrame, categorical_columns: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in categorical_columns:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        train_encoded = enc.fit_transform(train_df[[col]])
        test_encoded = enc.transform(test_df[[col]])

        feature_names = enc.get_feature_names_out([col])
        train_encoded_df = pd.DataFrame(train_encoded, columns=feature_names, index=train_df.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=feature_names, index=test_df.index)

        train_df = train_df.drop(columns=[col]).join(train_encoded_df)
        test_df = test_df.drop(columns=[col]).join(test_encoded_df)

    return train_df, test_df


def minmax_scale(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns_to_scale: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = train_df.copy()
    test_df = test_df.copy()

    scaler = MinMaxScaler()
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
    return train_df, test_df


def prepare_xy(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_train = train_df["class"].to_numpy(dtype=np.int64)
    y_test = test_df["class"].to_numpy(dtype=np.int64)

    X_train = train_df.drop(columns=["class"]).to_numpy(dtype=np.float32)
    X_test = test_df.drop(columns=["class"]).to_numpy(dtype=np.float32)

    if X_train.shape[1] != 121 or X_test.shape[1] != 121:
        raise ValueError(
            f"Expected 121 features (11x11). Got train={X_train.shape[1]} test={X_test.shape[1]}.\n"
            "This script assumes the exact preprocessing used in the thesis code."
        )

    X_train = X_train.reshape(-1, 11, 11, 1)
    X_test = X_test.reshape(-1, 11, 11, 1)
    return X_train, y_train, X_test, y_test


def build_cnn(input_shape: Tuple[int, int, int] = (11, 11, 1), num_classes: int = 5) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_baseline")
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def save_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> None:
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


def save_multiclass_roc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> None:
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
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"Class {CLASS_NAMES[i]} (AUC = {roc_auc[i]:.2f})",
        )

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline CNN (no CTGAN) for NSL-KDD 5-class classification.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0, help="If >0, truncate train set for quick smoke tests.")
    parser.add_argument("--max-test-samples", type=int, default=0, help="If >0, truncate test set for quick smoke tests.")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    paths = _repo_paths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    train_path = paths.data_dir / "KDDTrain+.txt"
    test_path = paths.data_dir / "KDDTest+.txt"

    train_df = load_nsl_kdd_txt(train_path)
    test_df = load_nsl_kdd_txt(test_path)

    # Drop constant all-zero feature (matches thesis script)
    train_df = train_df.drop(columns=["num_outbound_cmds"])
    test_df = test_df.drop(columns=["num_outbound_cmds"])

    # Collapse labels into 5 classes and convert to integer ids
    train_df = collapse_attack_labels(train_df, is_train=True)
    test_df = collapse_attack_labels(test_df, is_train=False)

    # Optional truncation for quick testing
    if args.max_train_samples and args.max_train_samples > 0:
        train_df = train_df.iloc[: args.max_train_samples].copy()
    if args.max_test_samples and args.max_test_samples > 0:
        test_df = test_df.iloc[: args.max_test_samples].copy()

    # One-hot encode categorical features
    categorical_columns = ["protocol_type", "service", "flag"]
    train_df, test_df = one_hot_encode(train_df, test_df, categorical_columns)

    # Scale numeric columns
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
    train_df, test_df = minmax_scale(train_df, test_df, columns_to_scale)

    # Ensure deterministic column order across train/test after encoding
    # (OneHotEncoder returns identical feature columns by construction; this is just a safeguard.)
    test_df = test_df[train_df.columns]

    X_train, y_train, X_test, y_test = prepare_xy(train_df, test_df)

    prefix = "baseline"

    # Build and train baseline CNN
    model = build_cnn()
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

    # Evaluate on test set (load best checkpoint if it exists)
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    report = classification_report(y_test, y_pred, digits=4, target_names=CLASS_NAMES)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc_sklearn = accuracy_score(y_test, y_pred)

    save_confusion_matrices(y_test, y_pred, paths.results_dir, prefix)
    save_multiclass_roc(y_test, y_proba, paths.results_dir, prefix)

    # Save a compact text report
    out_txt = paths.results_dir / f"{prefix}_results.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Baseline CNN (NO CTGAN) النتائج / Results\n")
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy (keras): {acc}\n")
        f.write(f"Test Accuracy (sklearn): {acc_sklearn}\n")
        f.write(f"MCC: {mcc}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    print("\n=== Baseline (NO CTGAN) Test Results ===")
    print(f"Test loss: {loss:.6f}")
    print(f"Test accuracy: {acc:.6f}")
    print(f"MCC: {mcc:.6f}")
    print(report)
    print(f"\nSaved: {out_txt}")
    print(f"Saved plots under: {paths.results_dir}")


if __name__ == "__main__":
    main()


