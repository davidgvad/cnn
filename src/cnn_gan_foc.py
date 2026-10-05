"""Class-balanced focal loss used by the paper models."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import tensorflow as tf


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
