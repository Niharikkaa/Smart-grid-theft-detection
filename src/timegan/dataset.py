"""Data loading utilities for TimeGAN training.

Loads the sequence tensor produced by Notebook 3
(``models/theft_sequences.npy``, shape ``(N, T, F)``) and turns it into a
``tf.data.Dataset`` of ``float32`` batches.

Notebook 3 fits its ``MinMaxScaler`` on the *honest* population and then
only calls ``.transform`` (not ``.fit_transform``) on the theft population.
Because a small number of theft users have consumption values outside the
range seen in the honest population, this means ``theft_sequences.npy`` is
not strictly bounded to ``[0, 1]`` (a handful of sequences contain values
up to ~459). TimeGAN's Embedder/Recovery/Generator networks use sigmoid
output activations, as in the original paper, which assumes inputs live in
``[0, 1]``. Silently feeding out-of-range values would make those
sequences fundamentally unreconstructable and would poison the shared
GRU state during training. Rather than editing Notebook 3, this loader
clips the input to ``[0, 1]`` at read time and reports how many points/
sequences were affected so the clipping is visible, not silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class SequenceStats:
    """Summary statistics reported when a sequence array is loaded."""

    num_sequences: int
    seq_len: int
    feature_dim: int
    num_points_clipped: int
    num_sequences_clipped: int
    raw_min: float
    raw_max: float


def load_sequences(path: str | Path, clip_to_unit_interval: bool = False) -> Tuple[np.ndarray, SequenceStats]:
    """Load a ``(N, T, F)`` sequence array saved by Notebook 3.

    Parameters
    ----------
    path:
        Path to a ``.npy`` file such as ``models/theft_sequences.npy``.
    clip_to_unit_interval:
        If ``True`` (default), values are clipped to ``[0, 1]`` after
        loading. See module docstring for why this is necessary for the
        theft sequences specifically.

    Returns
    -------
    data:
        ``float32`` array of shape ``(N, T, F)``.
    stats:
        Summary statistics describing the raw array and how much clipping
        (if any) was applied.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Run Notebook 3 first so that "
            "models/theft_sequences.npy exists."
        )

    raw = np.load(path)
    if raw.ndim != 3:
        raise ValueError(
            f"Expected a rank-3 array of shape (N, T, F), got shape {raw.shape}."
        )

    raw_min, raw_max = float(raw.min()), float(raw.max())
    num_points_clipped = int(np.sum((raw < 0.0) | (raw > 1.0)))
    seq_has_out_of_range = np.any((raw < 0.0) | (raw > 1.0), axis=(1, 2))
    num_sequences_clipped = int(np.sum(seq_has_out_of_range))

    data = raw.astype(np.float32)
    if clip_to_unit_interval:
        data = np.clip(data, 0.0, 1.0)

    stats = SequenceStats(
        num_sequences=data.shape[0],
        seq_len=data.shape[1],
        feature_dim=data.shape[2],
        num_points_clipped=num_points_clipped,
        num_sequences_clipped=num_sequences_clipped,
        raw_min=raw_min,
        raw_max=raw_max,
    )
    return data, stats


def make_dataset(
    data: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    shuffle_buffer: int | None = None,
    seed: int | None = 42,
    drop_remainder: bool = True,
) -> tf.data.Dataset:
    """Wrap a ``(N, T, F)`` array in a batched, shuffled ``tf.data.Dataset``.

    ``drop_remainder=True`` by default so every batch fed to the GRU-based
    networks has a static, known batch size, which keeps graph tracing
    (via ``tf.function`` inside the trainer) stable across steps.
    """
    if shuffle_buffer is None:
        shuffle_buffer = data.shape[0]

    ds = tf.data.Dataset.from_tensor_slices(data.astype(np.float32))
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
