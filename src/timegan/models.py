"""TimeGAN network definitions.

All five networks from the original paper are implemented as
``tf.keras.Model`` subclasses built from ``tf.keras.layers.GRU`` stacks,
executed eagerly (or under ``tf.function`` tracing) with ``GradientTape`` —
no ``tf.compat.v1``, sessions, placeholders, or ``MultiRNNCell``.

A ``tf.keras.layers.Dense`` applied to a rank-3 tensor of shape
``(batch, time, features)`` operates independently on the last axis for
every timestep, which is exactly the "time-distributed dense" behaviour
the original TF1 implementation built by hand with ``tf.contrib`` helpers.
So plain ``Dense`` layers are used directly instead of wrapping them in
``TimeDistributed`` — one fewer moving part, same computation.
"""

from __future__ import annotations

from typing import List

import tensorflow as tf
from tensorflow.keras import layers


def _gru_stack(hidden_dim: int, num_layers: int, name_prefix: str) -> List[layers.GRU]:
    """Build a stack of ``return_sequences=True`` GRU layers.

    This is the modern, eager-mode replacement for the original paper's
    ``MultiRNNCell`` + ``dynamic_rnn`` construction: stacking Keras GRU
    layers has identical semantics (the full hidden-state sequence of
    layer ``i`` feeds layer ``i + 1``) without any graph-construction
    boilerplate.
    """
    return [
        layers.GRU(
            hidden_dim,
            return_sequences=True,
            recurrent_dropout=0.1,
            name=f"{name_prefix}_gru_{i}",
        )
        for i in range(num_layers)
    ]


class Embedder(tf.keras.Model):
    """Maps real data ``X`` (batch, T, feature_dim) to latent codes ``H``.

    Output activation is sigmoid, matching the original paper, so the
    latent space is bounded to ``[0, 1]`` just like the (min-max scaled)
    input space.
    """

    def __init__(self, hidden_dim: int, num_layers: int, name: str = "embedder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.grus = _gru_stack(hidden_dim, num_layers, "embedder")
        self.out = layers.Dense(hidden_dim, activation="sigmoid", name="embedder_out")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = x
        for gru in self.grus:
            h = gru(h, training=training)
        return self.out(h)


class Recovery(tf.keras.Model):
    """Maps latent codes ``H`` back to data space ``X_tilde``."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_layers: int, name: str = "recovery", **kwargs):
        super().__init__(name=name, **kwargs)
        self.grus = _gru_stack(hidden_dim, num_layers, "recovery")
        self.out = layers.Dense(feature_dim, activation="sigmoid", name="recovery_out")

    def call(self, h: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = h
        for gru in self.grus:
            x = gru(x, training=training)
        return self.out(x)


class Generator(tf.keras.Model):
    """Maps random noise ``Z`` (batch, T, z_dim) to synthetic latent codes.

    The generator never touches data space directly — as in the paper, it
    only ever produces vectors in the same bounded latent space that the
    Embedder produces, which is what makes the supervised loss (next-step
    prediction consistency) meaningful.
    """

    def __init__(self, hidden_dim: int, num_layers: int, name: str = "generator", **kwargs):
        super().__init__(name=name, **kwargs)
        self.grus = _gru_stack(hidden_dim, num_layers, "generator")
        self.out = layers.Dense(hidden_dim, activation="sigmoid", name="generator_out")

    def call(self, z: tf.Tensor, training: bool = False) -> tf.Tensor:
        h = z
        for gru in self.grus:
            h = gru(h, training=training)
        return self.out(h)


class Supervisor(tf.keras.Model):
    """Predicts the "next-step" latent dynamics given a latent sequence.

    Uses one fewer layer than the other networks, matching the original
    paper's implementation (the supervisor is a lighter-weight network
    that only needs to learn local temporal transitions, not the full
    embedding/generation mapping).
    """

    def __init__(self, hidden_dim: int, num_layers: int, name: str = "supervisor", **kwargs):
        super().__init__(name=name, **kwargs)
        effective_layers = max(1, num_layers - 1)
        self.grus = _gru_stack(hidden_dim, effective_layers, "supervisor")
        self.out = layers.Dense(hidden_dim, activation="sigmoid", name="supervisor_out")

    def call(self, h: tf.Tensor, training: bool = False) -> tf.Tensor:
        s = h
        for gru in self.grus:
            s = gru(s, training=training)
        return self.out(s)


class Discriminator(tf.keras.Model):
    """Classifies a latent sequence as real or synthetic.

    Outputs per-timestep logits (no final activation) of shape
    ``(batch, T, 1)``; ``tf.nn.sigmoid_cross_entropy_with_logits`` is used
    downstream in ``losses.py`` for numerical stability instead of a
    sigmoid activation followed by binary cross-entropy.
    """

    def __init__(self, hidden_dim: int, num_layers: int, name: str = "discriminator", **kwargs):
        super().__init__(name=name, **kwargs)
        self.grus = _gru_stack(hidden_dim, num_layers, "discriminator")
        self.out = layers.Dense(1, activation=None, name="discriminator_logits")

    def call(self, h: tf.Tensor, training: bool = False) -> tf.Tensor:
        d = h
        for gru in self.grus:
            d = gru(d, training=training)
        return self.out(d)
