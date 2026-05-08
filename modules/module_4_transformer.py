"""Transformer components used by AfriWeave."""

from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")
import keras
from keras import layers


class MultiHeadAttention(layers.Layer):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value, mask=None):
        score = keras.ops.matmul(query, keras.ops.swapaxes(key, -1, -2))
        dim_key = keras.ops.cast(keras.ops.shape(key)[-1], "float32")
        scaled_score = score / keras.ops.sqrt(dim_key)

        if mask is not None:
            scaled_score += (mask * -1e9)

        weights = keras.activations.softmax(scaled_score, axis=-1)
        output = keras.ops.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = keras.ops.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return keras.ops.transpose(x, (0, 2, 1, 3))

    def call(self, inputs):
        batch_size = keras.ops.shape(inputs)[0]

        query = self.separate_heads(self.query_dense(inputs), batch_size)
        key = self.separate_heads(self.key_dense(inputs), batch_size)
        value = self.separate_heads(self.value_dense(inputs), batch_size)

        attention, _ = self.attention(query, key, value)
        attention = keras.ops.transpose(attention, (0, 2, 1, 3))
        concat_attention = keras.ops.reshape(attention, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)


def build_transformer_slm(vocab_size, max_len=200):
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = layers.Embedding(vocab_size, 256)(inputs)
    attention_output = MultiHeadAttention(256, 4)(embedding_layer)
    x = layers.Add()([embedding_layer, attention_output])
    x = layers.LayerNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)
