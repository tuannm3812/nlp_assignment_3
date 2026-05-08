"""Neural network model builders."""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
import keras


def build_mlp_model(input_dim: int, hidden_layers: list[int] | None = None, output_dim: int = 10):
    """Construct a feed-forward classification model."""
    hidden_layers = hidden_layers or [128, 64]
    layers = [keras.Input(shape=(input_dim,))]

    for units in hidden_layers:
        layers.append(keras.layers.Dense(units, activation="relu"))
        layers.append(keras.layers.Dropout(0.2))

    layers.append(keras.layers.Dense(output_dim, activation="softmax"))

    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
