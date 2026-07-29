"""
Conv1D/MLP architecture baselines for cnn_opt with mirrored GPU training.

The data preparation, optimized feature order, CTGAN augmentation, focal loss,
balanced batches, validation metrics, thresholds, and training settings match
cnn_opt.py. Conv1D reads the ordered features as a (121, 1) sequence. The MLP
reads the same ordered features as a flat vector and removes convolution.

Run:
    python -u src/cnn_opt_1d_4gpu.py --num-gpus 4 --epochs 25
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.model_selection import train_test_split

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
from cnn_opt import (  # type: ignore
    BalancedBatchSequence,
    ValF1Callback,
    apply_prediction_thresholds,
    build_optimized_feature_order,
    save_feature_grid,
    save_training_plot,
)


def build_opt_cnn_1d(
    loss: tf.keras.losses.Loss,
    groups: int = 1,
    base_filters: int = 64,
    dense_units: int = 256,
    dropout1: float = 0.25,
    dropout2: float = 0.30,
    use_batch_norm: bool = True,
    use_residual: bool = True,
) -> tf.keras.Model:
    """The cnn_opt model with Conv1D and MaxPooling1D instead of 2D layers."""
    groups = int(groups)
    base_filters = int(base_filters)
    if base_filters % groups != 0:
        raise ValueError(
            f"--groups must divide --base-filters. "
            f"Got groups={groups}, base_filters={base_filters}"
        )

    inputs = tf.keras.Input(shape=(121, 1))

    x = tf.keras.layers.Conv1D(
        base_filters,
        3,
        padding="same",
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    shortcut = x
    x = tf.keras.layers.Conv1D(
        base_filters,
        3,
        padding="same",
        groups=groups,
        use_bias=not use_batch_norm,
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv1D(
        base_filters,
        1,
        padding="same",
        use_bias=not use_batch_norm,
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    if use_residual:
        x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling1D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_opt_1d")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def build_opt_mlp(
    loss: tf.keras.losses.Loss,
    dense_units: int = 256,
    dropout1: float = 0.25,
    dropout2: float = 0.30,
    use_batch_norm: bool = True,
) -> tf.keras.Model:
    """Two-hidden-layer MLP using the same 121 ordered input features."""
    dense_units = int(dense_units)
    inputs = tf.keras.Input(shape=(121,))

    x = tf.keras.layers.Dense(
        dense_units,
        use_bias=not use_batch_norm,
    )(inputs)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout1)(x)

    x = tf.keras.layers.Dense(
        dense_units,
        use_bias=not use_batch_norm,
    )(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="mlp_opt")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def create_gpu_strategy(num_gpus: int, global_batch_size: int) -> tf.distribute.Strategy:
    """Create one complete model replica on each requested GPU."""
    available_gpus = tf.config.list_physical_devices("GPU")
    if len(available_gpus) < num_gpus:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs, but TensorFlow sees only "
            f"{len(available_gpus)}: {available_gpus}"
        )
    if global_batch_size % num_gpus != 0:
        raise ValueError(
            f"--batch-size ({global_batch_size}) must be divisible by "
            f"--num-gpus ({num_gpus})."
        )

    if num_gpus == 1:
        print("Using TensorFlow's default strategy with one visible GPU.")
        print(f"Global batch size: {global_batch_size}")
        return tf.distribute.get_strategy()

    devices = [f"/GPU:{index}" for index in range(num_gpus)]
    strategy = tf.distribute.MirroredStrategy(devices=devices)
    print(f"Using devices: {devices}")
    print(f"Model replicas: {strategy.num_replicas_in_sync}")
    print(f"Global batch size: {global_batch_size}")
    print(f"Batch size per GPU: {global_batch_size // num_gpus}")
    return strategy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NSL-KDD Conv1D/MLP baselines using the cnn_opt training pipeline."
    )

    parser.add_argument("--run-name", type=str, default="opt1d_4gpu")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument(
        "--architecture",
        choices=["conv1d", "mlp"],
        default="conv1d",
        help="Change only the neural backbone; the training pipeline stays the same.",
    )

    # Same training defaults as cnn_opt.
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-split", type=float, default=0.20)
    parser.add_argument("--minority-per-batch", type=int, default=1)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--cb-beta", type=float, default=0.9999)

    # Same CTGAN settings as cnn_opt.
    parser.add_argument(
        "--synth-path", type=str, default="data/synth_ctgan_5class.csv"
    )
    parser.add_argument("--ctgan-train", action="store_true")
    parser.add_argument("--ctgan-regenerate", action="store_true")
    parser.add_argument("--ctgan-only", action="store_true")
    parser.add_argument("--ctgan-epochs", type=int, default=200)
    parser.add_argument("--ctgan-batch-size", type=int, default=4096)
    parser.add_argument("--ctgan-pac", type=int, default=10)
    parser.add_argument("--target-dos", type=int, default=0)
    parser.add_argument("--target-probe", type=int, default=0)
    parser.add_argument("--target-r2l", type=int, default=5000)
    parser.add_argument("--target-u2r", type=int, default=0)
    parser.add_argument("--target-normal", type=int, default=0)

    parser.add_argument(
        "--feature-layout",
        choices=["optimized", "legacy"],
        default="optimized",
    )
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--base-filters", type=int, default=64)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--dropout1", type=float, default=0.25)
    parser.add_argument("--dropout2", type=float, default=0.30)
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--no-residual", action="store_true")

    # Same post-training decision policy as cnn_opt.
    parser.add_argument("--r2l-threshold", type=float, default=0.55)
    parser.add_argument("--u2r-threshold", type=float, default=0.40)
    parser.add_argument(
        "--no-thresholds",
        action="store_true",
        help="Use ordinary argmax predictions instead of rare-class rejection thresholds.",
    )
    args = parser.parse_args()

    if args.num_gpus <= 0:
        parser.error("--num-gpus must be greater than 0.")
    if args.minority_per_batch < 0:
        parser.error("--minority-per-batch must be 0 or greater.")
    if not 0.0 <= args.r2l_threshold <= 1.0:
        parser.error("--r2l-threshold must be between 0 and 1.")
    if not 0.0 <= args.u2r_threshold <= 1.0:
        parser.error("--u2r-threshold must be between 0 and 1.")

    tf.keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)

    paths = _repo_paths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_nsl_kdd_txt(
        paths.data_dir / "KDDTrain+.txt"
    ).drop(columns=["num_outbound_cmds"])
    test_df = load_nsl_kdd_txt(
        paths.data_dir / "KDDTest+.txt"
    ).drop(columns=["num_outbound_cmds"])
    train_df = collapse_attack_labels(train_df, is_train=True)
    test_df = collapse_attack_labels(test_df, is_train=False)

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
            raise ValueError(
                f"CTGAN synth file not found at {synth_path}. "
                "Run with --ctgan-train --ctgan-only first."
            )

    if args.ctgan_only:
        synth_counts = (
            np.bincount(
                synth_df["class"].to_numpy(dtype=np.int64), minlength=5
            )
            if len(synth_df)
            else np.zeros(5, dtype=np.int64)
        )
        print("CTGAN data saved at:", synth_path)
        print("Synthetic class counts:", synth_counts)
        return

    train_aug_df = (
        pd.concat([train_df, synth_df], ignore_index=True)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )

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

    encoders, feature_names = fit_one_hot_encoders(
        train_df, categorical_columns
    )
    train_real_ohe = apply_one_hot(
        train_df, encoders, feature_names, categorical_columns
    )
    scaler = fit_scaler(train_real_ohe, columns_to_scale)

    train_proc = apply_scaler(
        apply_one_hot(
            train_aug_df, encoders, feature_names, categorical_columns
        ),
        scaler,
        columns_to_scale,
    )
    test_proc = apply_scaler(
        apply_one_hot(test_df, encoders, feature_names, categorical_columns),
        scaler,
        columns_to_scale,
    )
    test_proc = test_proc[train_proc.columns]

    feature_columns = [c for c in train_proc.columns if c != "class"]
    if args.feature_layout == "optimized":
        ordered_features = build_optimized_feature_order(feature_columns)
    else:
        ordered_features = feature_columns

    ordered_columns = ordered_features + ["class"]
    train_proc = train_proc[ordered_columns]
    test_proc = test_proc[ordered_columns]

    prefix = args.run_name.strip() or "opt1d_4gpu"
    grid_path = paths.results_dir / f"{prefix}_feature_order.tsv"
    save_feature_grid(ordered_features, grid_path)

    X_all_2d, y_all, X_test_2d, y_test = prepare_xy_from_processed(
        train_proc, test_proc
    )
    if args.architecture == "conv1d":
        X_all = X_all_2d.reshape(-1, 121, 1)
        X_test = X_test_2d.reshape(-1, 121, 1)
    else:
        X_all = X_all_2d.reshape(-1, 121)
        X_test = X_test_2d.reshape(-1, 121)

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=y_all,
    )

    alpha, alpha_counts = compute_cb_alpha_effective_number(
        y_train, beta=args.cb_beta, num_classes=5
    )

    strategy = create_gpu_strategy(args.num_gpus, args.batch_size)
    with strategy.scope():
        loss = ClassBalancedFocalLoss(
            alpha=alpha, gamma=args.focal_gamma
        )
        if args.architecture == "conv1d":
            model = build_opt_cnn_1d(
                loss=loss,
                groups=args.groups,
                base_filters=args.base_filters,
                dense_units=args.dense_units,
                dropout1=args.dropout1,
                dropout2=args.dropout2,
                use_batch_norm=not args.no_bn,
                use_residual=not args.no_residual,
            )
        else:
            model = build_opt_mlp(
                loss=loss,
                dense_units=args.dense_units,
                dropout1=args.dropout1,
                dropout2=args.dropout2,
                use_batch_norm=not args.no_bn,
            )
    model_parameters = int(model.count_params())

    weights_path = paths.model_dir / f"{prefix}_best.weights.h5"
    model_path = paths.model_dir / f"{prefix}_best.keras"
    callbacks: List[tf.keras.callbacks.Callback] = [
        ValF1Callback(X_val, y_val, args.batch_size),
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

    if args.minority_per_batch > 0:
        training_data = BalancedBatchSequence(
            X_train,
            y_train,
            batch_size=args.batch_size,
            minority_per_batch=args.minority_per_batch,
            seed=args.seed,
        )
        history = model.fit(
            training_data,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
        )
    else:
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1,
            callbacks=callbacks,
        )

    save_training_plot(history, paths.results_dir, prefix)

    if weights_path.exists():
        model.load_weights(weights_path)
    model.save(model_path)

    best_val_macro_f1 = max(history.history.get("val_macro_f1", [0.0]))
    test_loss, keras_argmax_accuracy = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    raw_predictions = np.argmax(probabilities, axis=1)

    class_thresholds: Dict[int, float] = {
        2: float(args.r2l_threshold),
        3: float(args.u2r_threshold),
    }
    thresholds_applied = not args.no_thresholds
    predictions = (
        apply_prediction_thresholds(probabilities, class_thresholds)
        if thresholds_applied
        else raw_predictions.copy()
    )
    changed_predictions = int(
        np.count_nonzero(predictions != raw_predictions)
    )

    raw_report = classification_report(
        y_test,
        raw_predictions,
        digits=8,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    report = classification_report(
        y_test,
        predictions,
        digits=8,
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    raw_per_class_recall = recall_score(
        y_test,
        raw_predictions,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    raw_accuracy = float(accuracy_score(y_test, raw_predictions))
    raw_macro_f1 = float(
        f1_score(y_test, raw_predictions, average="macro", zero_division=0)
    )
    raw_macro_recall = float(np.mean(raw_per_class_recall))
    raw_mcc = float(matthews_corrcoef(y_test, raw_predictions))

    per_class_recall = recall_score(
        y_test,
        predictions,
        labels=np.arange(5),
        average=None,
        zero_division=0,
    )
    sklearn_accuracy = float(accuracy_score(y_test, predictions))
    test_macro_f1 = float(
        f1_score(y_test, predictions, average="macro", zero_division=0)
    )
    macro_recall = float(np.mean(per_class_recall))
    mcc = float(matthews_corrcoef(y_test, predictions))

    if thresholds_applied:
        save_confusion_matrices(
            y_test,
            raw_predictions,
            paths.results_dir,
            f"{prefix}_raw_argmax",
        )
    save_confusion_matrices(
        y_test, predictions, paths.results_dir, prefix
    )
    save_multiclass_roc(
        y_test, probabilities, paths.results_dir, prefix
    )

    real_counts = np.bincount(
        train_df["class"].to_numpy(dtype=np.int64), minlength=5
    )
    synth_counts = (
        np.bincount(
            synth_df["class"].to_numpy(dtype=np.int64), minlength=5
        )
        if len(synth_df)
        else np.zeros(5, dtype=np.int64)
    )
    augmented_counts = np.bincount(y_all, minlength=5)

    result_path = paths.results_dir / f"{prefix}_results.txt"
    with result_path.open("w", encoding="utf-8") as output_file:
        output_file.write(
            f"CNN_OPT {args.architecture.upper()} architecture baseline\n\n"
        )
        output_file.write(f"run_name: {prefix}\n")
        output_file.write(f"architecture: {args.architecture}\n")
        output_file.write(f"seed: {args.seed}\n")
        output_file.write(f"epochs: {args.epochs}\n")
        output_file.write(f"val_split: {args.val_split}\n")
        output_file.write(f"num_gpus: {args.num_gpus}\n")
        output_file.write(f"global_batch_size: {args.batch_size}\n")
        output_file.write(
            f"batch_size_per_gpu: {args.batch_size // args.num_gpus}\n"
        )
        output_file.write(f"feature_layout: {args.feature_layout}\n")
        output_file.write(f"feature_order_tsv: {grid_path}\n")
        output_file.write(f"real_counts: {real_counts.tolist()}\n")
        output_file.write(f"synth_counts: {synth_counts.tolist()}\n")
        output_file.write(
            f"augmented_counts: {augmented_counts.tolist()}\n\n"
        )
        output_file.write(f"focal_gamma: {args.focal_gamma}\n")
        output_file.write(f"cb_beta: {args.cb_beta}\n")
        output_file.write(
            f"minority_per_batch: {args.minority_per_batch}\n"
        )
        output_file.write(
            f"alpha_counts: {alpha_counts.tolist()}\n"
        )
        output_file.write(
            f"groups: {args.groups if args.architecture == 'conv1d' else 'not applicable'}\n"
        )
        output_file.write(
            "base_filters: "
            f"{args.base_filters if args.architecture == 'conv1d' else 'not applicable'}\n"
        )
        output_file.write(f"dense_units: {args.dense_units}\n")
        output_file.write(f"dropout1: {args.dropout1}\n")
        output_file.write(f"dropout2: {args.dropout2}\n")
        output_file.write(f"use_batch_norm: {not args.no_bn}\n")
        output_file.write(
            "use_residual: "
            f"{not args.no_residual if args.architecture == 'conv1d' else 'not applicable'}\n"
        )
        output_file.write(f"Model Parameters: {model_parameters}\n")
        output_file.write(
            f"Best Validation Macro F1: {best_val_macro_f1}\n\n"
        )
        output_file.write(f"thresholds_applied: {thresholds_applied}\n")
        output_file.write(f"r2l_threshold: {args.r2l_threshold}\n")
        output_file.write(f"u2r_threshold: {args.u2r_threshold}\n")
        output_file.write(
            f"threshold_changed_predictions: {changed_predictions}\n\n"
        )
        output_file.write(f"Test Loss: {test_loss}\n")
        output_file.write(
            f"Test Accuracy (keras): {keras_argmax_accuracy}\n"
        )
        output_file.write(f"Raw Argmax Test Accuracy: {raw_accuracy}\n")
        output_file.write(f"Raw Argmax Test Macro F1: {raw_macro_f1}\n")
        output_file.write(
            f"Raw Argmax Test Macro Recall: {raw_macro_recall}\n"
        )
        output_file.write(
            f"Raw Argmax R2L Recall: {float(raw_per_class_recall[2])}\n"
        )
        output_file.write(
            f"Raw Argmax U2R Recall: {float(raw_per_class_recall[3])}\n"
        )
        output_file.write(f"Raw Argmax MCC: {raw_mcc}\n")
        output_file.write(
            f"Test Accuracy (sklearn): {sklearn_accuracy}\n"
        )
        output_file.write(f"Test Macro F1: {test_macro_f1}\n")
        output_file.write(f"Test Macro Recall: {macro_recall}\n")
        output_file.write(
            f"R2L Recall: {float(per_class_recall[2])}\n"
        )
        output_file.write(
            f"U2R Recall: {float(per_class_recall[3])}\n"
        )
        output_file.write(f"MCC: {mcc}\n\n")
        output_file.write("Raw argmax report:\n")
        output_file.write(raw_report)
        output_file.write("\n\n")
        output_file.write("Classification report:\n")
        output_file.write(report)
        output_file.write("\n")

    print(
        f"\n=== CNN_OPT {args.architecture.upper()} Test Results ==="
    )
    print("Thresholds applied:", thresholds_applied)
    print("Changed predictions:", changed_predictions)
    print(f"Test loss: {test_loss:.6f}")
    print(f"Raw argmax accuracy: {raw_accuracy:.6f}")
    print(f"Raw argmax macro-F1: {raw_macro_f1:.6f}")
    print(f"Raw argmax macro recall: {raw_macro_recall:.6f}")
    print(f"Raw argmax R2L recall: {raw_per_class_recall[2]:.6f}")
    print(f"Raw argmax U2R recall: {raw_per_class_recall[3]:.6f}")
    print(f"Raw argmax MCC: {raw_mcc:.6f}")
    print(f"Final accuracy: {sklearn_accuracy:.6f}")
    print(f"Final macro-F1: {test_macro_f1:.6f}")
    print(f"Final macro recall: {macro_recall:.6f}")
    print(f"Final R2L recall: {per_class_recall[2]:.6f}")
    print(f"Final U2R recall: {per_class_recall[3]:.6f}")
    print(f"Final MCC: {mcc:.6f}")
    print(f"Model parameters: {model_parameters}")
    print(f"Best validation macro-F1: {best_val_macro_f1:.6f}")
    print(report)
    print("Saved results:", result_path)
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
