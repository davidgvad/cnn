"""
cnn_fin.py — "Ultimate" NSL-KDD 5-Class experiment (CTGAN + Focal + Optimized CNN)
---------------------------------------------------------------------------------

This script builds on your working pipeline:
- CTGAN augmentation (cached CSV or generated once)
- Class-Balanced Focal Loss (alpha recomputed after augmentation/undersampling)
- "Optimized" CNN via grouped convolution (multi-group) + optional BN + residual mixing
- Optional majority-class undersampling caps (data-side technique from the original thesis script)

Recommended workflow (cluster)
------------------------------
1) Precompute CTGAN once:
   python -u src/cnn_fin.py --ctgan-train --ctgan-regenerate --ctgan-only --target-r2l 5000 --target-u2r 0

2) Train/eval repeatedly (no CTGAN retrain):
   python -u src/cnn_fin.py --epochs 25

Default settings
----------------
The defaults are tuned to the current best config from your 120-trial sweep:
- groups=1
- focal_gamma=1.5
- cb_beta=0.9999
- undersampling disabled by default
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

# Reuse the proven preprocessing + CTGAN utilities from `cnn_gan_foc.py`
from cnn_gan_foc import (  # type: ignore
    CLASS_NAMES,
    ClassBalancedFocalLoss,
    _generate_synth_ctgan,
    _maybe_load_synth,
    _repo_paths,
    apply_one_hot,
    apply_scaler,
    collapse_attack_labels,
    compute_cb_alpha_effective_number,
    fit_one_hot_encoders,
    fit_scaler,
    load_nsl_kdd_txt,
    prepare_xy_from_processed,
    save_confusion_matrices,
    save_multiclass_roc,
)


def undersample_caps(y: np.ndarray, caps: Dict[int, int], seed: int) -> np.ndarray:
    """Return indices for random undersampling without replacement."""
    rng = np.random.default_rng(int(seed))
    parts: List[np.ndarray] = []
    for cls in range(5):
        idx = np.where(y == cls)[0]
        cap = int(caps.get(cls, -1))
        if cap <= 0 or cap >= len(idx):
            parts.append(idx)
        else:
            parts.append(rng.choice(idx, size=cap, replace=False))
    keep = np.concatenate(parts) if parts else np.arange(len(y))
    rng.shuffle(keep)
    return keep


class ValF1Callback(tf.keras.callbacks.Callback):
    """Logs val_macro_f1 + val_rare_f1 so we can checkpoint on imbalance-aware metrics."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, batch_size: int) -> None:
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = int(batch_size)

    def on_epoch_end(self, epoch: int, logs: Dict[str, float] | None = None) -> None:
        logs = logs or {}
        y_proba = self.model.predict(self.X_val, batch_size=self.batch_size, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)

        macro_f1 = float(f1_score(self.y_val, y_pred, average="macro"))
        r2l_f1 = float(f1_score(self.y_val == 2, y_pred == 2))
        u2r_f1 = float(f1_score(self.y_val == 3, y_pred == 3))
        rare_f1 = float((r2l_f1 + u2r_f1) / 2.0)

        logs["val_macro_f1"] = macro_f1
        logs["val_rare_f1"] = rare_f1

        print(f" — val_macro_f1: {macro_f1:.4f} — val_rare_f1: {rare_f1:.4f}", flush=True)


def build_grouped_cnn(
    loss: tf.keras.losses.Loss,
    groups: int = 4,
    base_filters: int = 64,
    dense_units: int = 256,
    dropout1: float = 0.25,
    dropout2: float = 0.30,
    use_batch_norm: bool = True,
    use_residual: bool = True,
) -> tf.keras.Model:
    """
    Optimized-ish CNN for 11x11 grayscale inputs:
    - stem conv
    - grouped conv block (multi-group) + 1x1 channel mixing + optional residual
    - dense head
    """
    groups = int(groups)
    base_filters = int(base_filters)
    if base_filters % groups != 0:
        raise ValueError(f"--groups must divide --base-filters. Got groups={groups}, base_filters={base_filters}")

    inputs = tf.keras.Input(shape=(11, 11, 1))
    x = tf.keras.layers.Conv2D(base_filters, 3, padding="same", use_bias=not use_batch_norm)(inputs)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    shortcut = x
    x = tf.keras.layers.Conv2D(
        base_filters,
        3,
        padding="same",
        groups=groups,
        use_bias=not use_batch_norm,
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # channel mixing
    x = tf.keras.layers.Conv2D(base_filters, 1, padding="same", use_bias=not use_batch_norm)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    if use_residual:
        x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_fin_grouped")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def save_training_plot(history: tf.keras.callbacks.History, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(history.history.get("accuracy", []), label="Train Acc")
    if "val_accuracy" in history.history:
        plt.plot(history.history["val_accuracy"], label="Val Acc")
    if "val_macro_f1" in history.history:
        plt.plot(history.history["val_macro_f1"], label="Val Macro-F1")
    if "val_rare_f1" in history.history:
        plt.plot(history.history["val_rare_f1"], label="Val Rare-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Training Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_training_plot.png", dpi=400)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="NSL-KDD 5-class: CTGAN + focal + grouped CNN (ultimate).")

    # Output naming (important for running many trials / slurm arrays)
    parser.add_argument(
        "--run-name",
        type=str,
        default="fin",
        help="Prefix for outputs under results/ and model/. Use this to avoid overwriting when running sweeps.",
    )

    # Train
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.20)

    # Focal
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--cb-beta", type=float, default=0.9999)

    # CTGAN
    parser.add_argument("--synth-path", type=str, default="data/synth_ctgan_5class.csv")
    parser.add_argument("--ctgan-train", action="store_true")
    parser.add_argument("--ctgan-regenerate", action="store_true")
    parser.add_argument("--ctgan-only", action="store_true")
    parser.add_argument("--ctgan-epochs", type=int, default=200)
    parser.add_argument("--ctgan-batch-size", type=int, default=4096)
    parser.add_argument("--ctgan-pac", type=int, default=10)

    # CTGAN targets (total desired per class in training after augmentation)
    parser.add_argument("--target-dos", type=int, default=0)
    parser.add_argument("--target-probe", type=int, default=0)
    parser.add_argument("--target-r2l", type=int, default=5000)
    parser.add_argument("--target-u2r", type=int, default=0)
    parser.add_argument("--target-normal", type=int, default=0)

    # Data optimization (thesis-style): cap majority classes via undersampling
    # Defaults tuned from sweep: undersampling OFF.
    parser.add_argument(
        "--undersample",
        dest="undersample",
        action="store_true",
        help="Enable majority-class undersampling caps (DoS/Probe/normal).",
    )
    parser.add_argument(
        "--no-undersample",
        dest="undersample",
        action="store_false",
        help="Disable majority-class undersampling caps (default).",
    )
    parser.set_defaults(undersample=False)
    parser.add_argument("--cap-dos", type=int, default=30500)
    parser.add_argument("--cap-probe", type=int, default=29500)
    parser.add_argument("--cap-normal", type=int, default=19000)

    # Architecture
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout1", type=float, default=0.25)
    parser.add_argument("--dropout2", type=float, default=0.30)
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--no-residual", action="store_true")

    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    paths = _repo_paths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    # Load + preprocess labels
    train_df = load_nsl_kdd_txt(paths.data_dir / "KDDTrain+.txt").drop(columns=["num_outbound_cmds"])
    test_df = load_nsl_kdd_txt(paths.data_dir / "KDDTest+.txt").drop(columns=["num_outbound_cmds"])
    train_df = collapse_attack_labels(train_df, is_train=True)
    test_df = collapse_attack_labels(test_df, is_train=False)

    # CTGAN: load or generate
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
            synth_df = train_df.iloc[:0].copy()

    if args.ctgan_only:
        synth_counts = np.bincount(synth_df["class"].to_numpy(dtype=np.int64), minlength=5) if len(synth_df) else np.zeros(5, dtype=np.int64)
        print("\n=== CTGAN precompute complete (ctgan-only) ===")
        print("Synth path:", synth_path)
        print("Targets:", targets)
        print("Synth class counts:", synth_counts)
        return

    # Combine train + synth (training only)
    import pandas as pd  # local import to avoid an unconditional top-level dependency in this file

    train_aug_df = (
        pd.concat([train_df, synth_df], ignore_index=True)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

    # Fit OHE + scaler on REAL train only (stable 121-feature layout)
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
    train_real_ohe = apply_one_hot(train_df, encoders, feature_names, categorical_columns)
    scaler = fit_scaler(train_real_ohe, columns_to_scale)

    train_proc = apply_scaler(apply_one_hot(train_aug_df, encoders, feature_names, categorical_columns), scaler, columns_to_scale)
    test_proc = apply_scaler(apply_one_hot(test_df, encoders, feature_names, categorical_columns), scaler, columns_to_scale)
    test_proc = test_proc[train_proc.columns]

    X_all, y_all, X_test, y_test = prepare_xy_from_processed(train_proc, test_proc)

    real_counts = np.bincount(train_df["class"].to_numpy(dtype=np.int64), minlength=5)
    synth_counts = np.bincount(synth_df["class"].to_numpy(dtype=np.int64), minlength=5) if len(synth_df) else np.zeros(5, dtype=np.int64)
    aug_counts = np.bincount(y_all, minlength=5)

    # Thesis-style majority undersampling caps (data-side optimization)
    if args.undersample:
        caps = {0: args.cap_dos, 1: args.cap_probe, 4: args.cap_normal}
        keep = undersample_caps(y_all, caps=caps, seed=args.seed)
        X_all = X_all[keep]
        y_all = y_all[keep]

    final_counts = np.bincount(y_all, minlength=5)

    # Train/val split (stratified) so we can monitor macro-F1
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all,
        y_all,
        test_size=float(args.val_split),
        random_state=args.seed,
        stratify=y_all,
    )

    # Focal alpha after augmentation/undersampling
    alpha, alpha_counts = compute_cb_alpha_effective_number(y_tr, beta=args.cb_beta, num_classes=5)
    loss_obj = ClassBalancedFocalLoss(alpha=alpha, gamma=args.focal_gamma)

    model = build_grouped_cnn(
        loss=loss_obj,
        groups=args.groups,
        base_filters=args.base_filters,
        dense_units=args.dense_units,
        dropout1=args.dropout1,
        dropout2=args.dropout2,
        use_batch_norm=not args.no_bn,
        use_residual=not args.no_residual,
    )

    prefix = args.run_name.strip() or "fin"
    weights_path = paths.model_dir / f"{prefix}_best.weights.h5"
    model_path = paths.model_dir / f"{prefix}_best.keras"

    callbacks: List[tf.keras.callbacks.Callback] = [
        ValF1Callback(X_val=X_val, y_val=y_val, batch_size=args.batch_size),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_macro_f1",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1",
            mode="max",
            patience=8,
            restore_best_weights=False,
        ),
    ]

    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    save_training_plot(history, paths.results_dir, prefix)

    # Load best weights (no full-model deserialization headaches)
    if weights_path.exists():
        model.load_weights(weights_path)
    model.save(model_path)

    # Test eval
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    report = classification_report(y_test, y_pred, digits=4, target_names=CLASS_NAMES)
    mcc = matthews_corrcoef(y_test, y_pred)
    acc_sklearn = accuracy_score(y_test, y_pred)

    save_confusion_matrices(y_test, y_pred, paths.results_dir, prefix)
    save_multiclass_roc(y_test, y_proba, paths.results_dir, prefix)

    out_txt = paths.results_dir / f"{prefix}_results.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("CNN_FIN (CTGAN + Focal + Grouped CNN + optional undersampling)\n\n")
        f.write(f"synth_path: {synth_path}\n")
        f.write(f"targets: {targets}\n")
        f.write(f"real_counts: {real_counts.tolist()}\n")
        f.write(f"synth_counts: {synth_counts.tolist()}\n")
        f.write(f"aug_counts: {aug_counts.tolist()}\n")
        f.write(f"final_counts(after_undersample): {final_counts.tolist()}\n\n")
        f.write(f"focal_gamma: {args.focal_gamma}\n")
        f.write(f"cb_beta: {args.cb_beta}\n")
        f.write(f"alpha_counts(train_split): {alpha_counts.tolist()}\n")
        f.write(f"alpha: {[float(x) for x in alpha.tolist()]}\n\n")
        f.write(f"groups: {args.groups}\n")
        f.write(f"base_filters: {args.base_filters}\n")
        f.write(f"dense_units: {args.dense_units}\n")
        f.write(f"use_batch_norm: {not args.no_bn}\n")
        f.write(f"use_residual: {not args.no_residual}\n\n")
        f.write(f"Test Loss: {loss}\n")
        f.write(f"Test Accuracy (keras): {acc}\n")
        f.write(f"Test Accuracy (sklearn): {acc_sklearn}\n")
        f.write(f"MCC: {mcc}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\n")

    print("\n=== CNN_FIN Test Results ===")
    print("Real train counts:", real_counts)
    print("Synth counts:", synth_counts)
    print("Aug counts:", aug_counts)
    print("Final counts (after undersampling):", final_counts)
    print("Alpha:", alpha)
    print(f"Test loss: {loss:.6f}")
    print(f"Test accuracy: {acc:.6f}")
    print(f"MCC: {mcc:.6f}")
    print(report)
    print(f"\nSaved: {out_txt}")
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()


