"""Conv1D, MLP, and feature-token Transformer models."""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="cnn")
class AddLearnedPositionEmbedding(tf.keras.layers.Layer):
    """Add one learned identity/position vector to each feature token."""

    def build(self, input_shape: tf.TensorShape) -> None:
        if len(input_shape) != 3:
            raise ValueError(
                "Position embeddings expect shape "
                f"(batch, tokens, dimensions), got {input_shape}."
            )
        token_count = input_shape[1]
        embedding_size = input_shape[2]
        if token_count is None or embedding_size is None:
            raise ValueError(
                "Token count and embedding size must be known when building."
            )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(1, int(token_count), int(embedding_size)),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + tf.cast(self.position_embeddings, inputs.dtype)


def build_vanilla_transformer(
    loss: tf.keras.losses.Loss,
    d_model: int = 64,
    num_heads: int = 4,
    num_blocks: int = 2,
    ff_dim: int = 128,
    dense_units: int = 512,
    transformer_dropout: float = 0.10,
    head_dropout: float = 0.30,
) -> tf.keras.Model:
    """Build a small vanilla encoder over the 121 ordered feature tokens."""
    d_model = int(d_model)
    num_heads = int(num_heads)
    num_blocks = int(num_blocks)
    ff_dim = int(ff_dim)
    dense_units = int(dense_units)
    if d_model <= 0:
        raise ValueError("--d-model must be greater than zero.")
    if num_heads <= 0:
        raise ValueError("--num-heads must be greater than zero.")
    if d_model % num_heads != 0:
        raise ValueError(
            f"--num-heads must divide --d-model. "
            f"Got num_heads={num_heads}, d_model={d_model}."
        )
    if num_blocks <= 0:
        raise ValueError("--transformer-blocks must be greater than zero.")
    if ff_dim <= 0:
        raise ValueError("--ff-dim must be greater than zero.")
    if dense_units <= 0:
        raise ValueError("--dense-units must be greater than zero.")
    if not 0.0 <= transformer_dropout < 1.0:
        raise ValueError("--transformer-dropout must be in [0, 1).")
    if not 0.0 <= head_dropout < 1.0:
        raise ValueError("--dropout2 must be in [0, 1).")

    inputs = tf.keras.Input(shape=(121, 1))
    x = tf.keras.layers.Dense(d_model, name="scalar_projection")(inputs)
    x = AddLearnedPositionEmbedding(name="feature_position_embedding")(x)
    x = tf.keras.layers.Dropout(
        transformer_dropout,
        name="embedding_dropout",
    )(x)

    for block_index in range(num_blocks):
        block_name = f"encoder_{block_index + 1}"
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=transformer_dropout,
            name=f"{block_name}_attention",
        )(x, x)
        attention = tf.keras.layers.Dropout(
            transformer_dropout,
            name=f"{block_name}_attention_dropout",
        )(attention)
        x = tf.keras.layers.Add(name=f"{block_name}_attention_residual")(
            [x, attention]
        )
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{block_name}_attention_norm",
        )(x)

        feed_forward = tf.keras.layers.Dense(
            ff_dim,
            activation="relu",
            name=f"{block_name}_ffn_expand",
        )(x)
        feed_forward = tf.keras.layers.Dropout(
            transformer_dropout,
            name=f"{block_name}_ffn_dropout_1",
        )(feed_forward)
        feed_forward = tf.keras.layers.Dense(
            d_model,
            name=f"{block_name}_ffn_project",
        )(feed_forward)
        feed_forward = tf.keras.layers.Dropout(
            transformer_dropout,
            name=f"{block_name}_ffn_dropout_2",
        )(feed_forward)
        x = tf.keras.layers.Add(name=f"{block_name}_ffn_residual")(
            [x, feed_forward]
        )
        x = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name=f"{block_name}_ffn_norm",
        )(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="token_average")(x)
    x = tf.keras.layers.Dense(
        dense_units,
        activation="relu",
        name="classifier_hidden",
    )(x)
    x = tf.keras.layers.Dropout(
        head_dropout,
        name="classifier_dropout",
    )(x)
    outputs = tf.keras.layers.Dense(
        5,
        activation="softmax",
        name="class_probabilities",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="vanilla_feature_transformer",
    )
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


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
