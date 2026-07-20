"""Loading and stratified splitting of the Notebook 5 output for Notebook 6."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
from sklearn.model_selection import train_test_split


class Splits(NamedTuple):
    """Train/validation/test split, stratified on the label."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_fft_dataset(sequences_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the FFT-filtered sequences and labels saved by Notebook 5.

    Raises
    ------
    FileNotFoundError
        If either file is missing (i.e. Notebook 5 has not been run yet).
    ValueError
        If the arrays have mismatched lengths or an unexpected shape.
    """
    sequences_path = Path(sequences_path)
    labels_path = Path(labels_path)

    for path, name in ((sequences_path, "train_sequences_fft.npy"), (labels_path, "train_labels.npy")):
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {name} at {path}. Run Notebook 5 first — "
                "Notebook 6 only performs classification and does not "
                "regenerate this file."
            )

    X = np.load(sequences_path).astype(np.float32)
    y = np.load(labels_path).astype(np.int64)

    if X.ndim != 3 or X.shape[2] != 1:
        raise ValueError(f"Expected sequences of shape (samples, T, 1), got {X.shape}.")
    if len(X) != len(y):
        raise ValueError(f"sequences ({len(X)}) and labels ({len(y)}) length mismatch.")

    return X, y


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Splits:
    """Split into train/validation/test sets, stratified by label at each step.

    ``val_size`` and ``test_size`` are both fractions of the *full* dataset.
    Internally this performs two stratified splits: first test is carved
    off, then validation is carved off the remainder, so the final
    proportions match the requested fractions.
    """
    if not (0.0 < val_size < 1.0) or not (0.0 < test_size < 1.0):
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.0.")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    # Re-express val_size as a fraction of the remaining (temp) data.
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=seed
    )

    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)
