"""GRU classifier architecture for electricity theft detection.

Design rationale
-----------------
- **GRU over LSTM.** GRUs have fewer gates/parameters than LSTMs while
  reaching comparable performance on most sequence-classification tasks.
  With ``T = 1033`` timesteps per sequence, that parameter saving keeps
  training time and memory usage reasonable, which matters for a
  lightweight academic pipeline.
- **Two stacked GRU layers.** The first GRU (``return_sequences=True``)
  extracts local/short-term consumption dynamics at every timestep; the
  second GRU compresses that sequence of representations into a single
  summary vector describing the whole 1033-day consumption history. This
  is a standard, well-documented pattern for sequence classification.
- **Dropout between the two GRU layers.** The theft class is augmented
  with TimeGAN-synthesized sequences, which increases the risk of the
  network memorizing generator-specific artefacts. Dropout regularizes
  against this without adding parameters.
- **A small Dense(16, relu) head before the sigmoid.** Slightly improves
  the model's ability to draw a non-linear decision boundary on top of
  the GRU summary vector, at a negligible parameter cost - this is the
  "slight improvement" allowed on top of the suggested architecture.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_gru_classifier(
    seq_len: int,
    n_features: int = 1,
    gru_units: tuple[int, int] = (64, 32),
    dropout_rate: float = 0.3,
    dense_units: int = 16,
    learning_rate: float = 1e-3,
    seed: int = 42,
) -> keras.Model:
    """Build and compile the GRU classifier.

    Architecture
    ------------
    Input(T, n_features)
      -> GRU(gru_units[0], return_sequences=True)
      -> Dropout(dropout_rate)
      -> GRU(gru_units[1])
      -> Dense(dense_units, relu)
      -> Dropout(dropout_rate / 2)
      -> Dense(1, sigmoid)
    """
    keras.utils.set_random_seed(seed)

    inputs = keras.Input(shape=(seq_len, n_features), name="consumption_sequence")

    x = layers.GRU(gru_units[0], return_sequences=True, name="gru_1")(inputs)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.GRU(gru_units[1], name="gru_2")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = layers.Dropout(dropout_rate / 2, name="dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="theft_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="gru_theft_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    return model
