"""Loading, merging, labeling and saving utilities for Notebook 5.

All functions here are pure (no plotting, no filtering) so that they stay
easy to unit-test independently of the DSP code in ``filtering.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np


class RawSequences(NamedTuple):
    """Container for the three arrays produced by Notebooks 3 and 4."""

    honest: np.ndarray
    theft_real: np.ndarray
    theft_synthetic: np.ndarray


def _load_npy(path: Path, name: str) -> np.ndarray:
    """Load a single ``.npy`` file with a clear error if it is missing.

    Parameters
    ----------
    path : Path
        Location of the ``.npy`` file.
    name : str
        Human-readable name used in error messages (e.g. ``"honest"``).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the '{name}' sequences at {path}. "
            "Make sure Notebook 4 has been run and saved its outputs "
            "into the models/ folder before running Notebook 5."
        )
    array = np.load(path)
    if array.ndim != 3 or array.shape[2] != 1:
        raise ValueError(
            f"Expected '{name}' sequences with shape (samples, T, 1), "
            f"got {array.shape} from {path}."
        )
    return array.astype(np.float32)


def load_raw_sequences(
    honest_path: Path,
    theft_path: Path,
    synthetic_theft_path: Path,
) -> RawSequences:
    """Load the three TimeGAN-stage arrays produced by Notebooks 3 and 4.

    Raises
    ------
    FileNotFoundError
        If any of the three files is missing.
    ValueError
        If any array does not have shape ``(samples, T, 1)``, or the three
        arrays do not share the same sequence length ``T``.
    """
    honest = _load_npy(honest_path, "honest")
    theft_real = _load_npy(theft_path, "theft (real)")
    theft_synth = _load_npy(synthetic_theft_path, "theft (synthetic)")

    seq_lens = {honest.shape[1], theft_real.shape[1], theft_synth.shape[1]}
    if len(seq_lens) != 1:
        raise ValueError(
            "Sequence length mismatch across the three arrays: "
            f"honest={honest.shape[1]}, theft_real={theft_real.shape[1]}, "
            f"theft_synthetic={theft_synth.shape[1]}. "
            "All three must come from the same Notebook 3 pipeline."
        )

    return RawSequences(honest=honest, theft_real=theft_real, theft_synthetic=theft_synth)


def merge_theft_sequences(theft_real: np.ndarray, theft_synthetic: np.ndarray) -> np.ndarray:
    """Concatenate real and synthetic theft sequences into one theft pool.

    Honest data is intentionally left untouched by the caller — only the
    minority (theft) class is augmented with TimeGAN samples, which is the
    whole point of using TimeGAN in this pipeline.
    """
    return np.concatenate([theft_real, theft_synthetic], axis=0).astype(np.float32)


def build_labels(honest: np.ndarray, theft: np.ndarray) -> np.ndarray:
    """Return a binary label vector aligned with ``concat([honest, theft])``.

    Honest -> 0, Theft -> 1.
    """
    labels = np.concatenate(
        [np.zeros(len(honest), dtype=np.int64), np.ones(len(theft), dtype=np.int64)]
    )
    return labels


def shuffle_dataset(
    sequences: np.ndarray, labels: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle sequences and labels together using a fixed-seed permutation."""
    if len(sequences) != len(labels):
        raise ValueError(
            f"sequences ({len(sequences)}) and labels ({len(labels)}) "
            "must have the same length."
        )
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(sequences))
    return sequences[permutation], labels[permutation]


def save_dataset(
    sequences: np.ndarray, labels: np.ndarray, models_dir: Path
) -> tuple[Path, Path]:
    """Save the final FFT-filtered dataset that Notebook 6 will load.

    Writes ``train_sequences_fft.npy`` and ``train_labels.npy`` into
    ``models_dir`` and returns their paths.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    sequences_path = models_dir / "train_sequences_fft.npy"
    labels_path = models_dir / "train_labels.npy"

    np.save(sequences_path, sequences.astype(np.float32))
    np.save(labels_path, labels.astype(np.int64))

    return sequences_path, labels_path


def save_fft_config(cutoff_ratio: float, results_dir: Path) -> Path:
    """Save the FFT experiment configuration used for this run.

    Writes ``results/fft_config.json`` so the exact cutoff ratio used to
    produce ``train_sequences_fft.npy`` is always recoverable later,
    independent of the notebook's in-memory state — this is what makes
    the FFT preprocessing step reproducible.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    config_path = results_dir / "fft_config.json"
    config = {"cutoff_ratio": cutoff_ratio}

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    return config_path
