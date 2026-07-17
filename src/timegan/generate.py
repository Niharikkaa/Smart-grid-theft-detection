"""Generate and persist synthetic sequences from a trained TimeGAN.

Used by Notebook 4 to produce ``models/synthetic_theft_sequences.npy`` for
Notebook 5.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .trainer import TimeGAN


def generate_synthetic_sequences(model: TimeGAN, num_samples: int, batch_size: int = 64) -> np.ndarray:
    """Generate ``num_samples`` synthetic sequences from a trained :class:`TimeGAN`.

    Returns a ``float32`` array of shape ``(num_samples, seq_len, feature_dim)``,
    matching the layout of ``models/theft_sequences.npy`` produced by
    Notebook 3, with values bounded to ``[0, 1]`` by construction (every
    network's output layer is a sigmoid).
    """
    return model.generate(num_samples=num_samples, batch_size=batch_size)


def save_synthetic_sequences(data: np.ndarray, path: str | Path) -> Path:
    """Save a synthetic sequence array to ``path`` (creating parent dirs as needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data.astype(np.float32))
    return path
