"""Conv2D model and minority-guaranteed batches"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import tensorflow as tf


class BalancedBatchSequence(tf.keras.utils.Sequence):
    """Create batches that always include R2L and U2R samples when available."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        minority_per_batch: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.X = X
        self.y = np.asarray(y)
        self.batch_size = int(batch_size)
        self.minority_per_batch = int(minority_per_batch)
        self.seed = int(seed)
        self.epoch = 0

        if len(self.X) != len(self.y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(self.y) == 0:
            raise ValueError("Training data cannot be empty.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if self.minority_per_batch <= 0:
            raise ValueError("minority_per_batch must be greater than 0.")

        # NSL-KDD labels: 2 = R2L and 3 = U2R.
        self.minority_indices = {
            class_id: np.flatnonzero(self.y == class_id) for class_id in (2, 3)
        }
        self.available_minority_classes = [
            class_id for class_id, indices in self.minority_indices.items() if len(indices) > 0
        ]

        guaranteed_count = self.minority_per_batch * len(self.available_minority_classes)
        if guaranteed_count > self.batch_size:
            raise ValueError(
                "batch_size is too small for the requested minority samples per batch."
            )

        self.all_indices = np.arange(len(self.y))
        self.steps_per_epoch = int(np.ceil(len(self.y) / self.batch_size))

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, batch_index: int) -> Tuple[np.ndarray, np.ndarray]:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("Batch index is out of range.")

        # Draw a repeatable sample for each batch and epoch.
        rng = np.random.default_rng(
            self.seed + self.epoch * self.steps_per_epoch + int(batch_index)
        )

        selected: List[int] = []
        for class_id in self.available_minority_classes:
            class_indices = self.minority_indices[class_id]
            # Repeat rows only when the class has fewer examples than its quota.
            chosen = rng.choice(
                class_indices,
                size=self.minority_per_batch,
                replace=len(class_indices) < self.minority_per_batch,
            )
            selected.extend(chosen.tolist())

        # Sample the remaining examples with replacement from all training rows.
        remaining = self.batch_size - len(selected)
        if remaining > 0:
            selected.extend(
                rng.choice(self.all_indices, size=remaining, replace=True).tolist()
            )

        batch_indices = np.asarray(selected, dtype=np.int64)
        rng.shuffle(batch_indices)
        return self.X[batch_indices], self.y[batch_indices]

    def on_epoch_end(self) -> None:
        self.epoch += 1


def build_opt_cnn(
    loss: tf.keras.losses.Loss,
    groups: int = 1,
    base_filters: int = 64,
    dense_units: int = 256,
    dropout1: float = 0.25,
    dropout2: float = 0.30,
    use_batch_norm: bool = True,
    use_residual: bool = True,
) -> tf.keras.Model:
    """
    CNN for 11x11 grayscale:
    - stem conv
    - optional grouped conv block (groups param)
    - 1x1 channel mixing + optional residual
    - dense head
    """
    groups = int(groups)
    base_filters = int(base_filters)
    if base_filters % groups != 0:
        raise ValueError(f"--groups must divide --base-filters. Got groups={groups}, base_filters={base_filters}")

    inputs = tf.keras.Input(shape=(11, 11, 1))

    # A single input channel requires groups=1.
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

    x = tf.keras.layers.Conv2D(base_filters, 1, padding="same", use_bias=not use_batch_norm)(x)
    if use_batch_norm:
        x = tf.keras.layers.BatchNormalization()(x)

    if use_residual:
        x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(int(dense_units), activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout2)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_opt")
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model
