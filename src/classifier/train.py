"""Training utilities: callbacks and the training loop for Notebook 6."""

from __future__ import annotations

from pathlib import Path

from tensorflow import keras


def get_callbacks(
    checkpoint_path: Path,
    es_patience: int = 10,
    rlr_patience: int = 5,
    rlr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list[keras.callbacks.Callback]:
    """Build the standard EarlyStopping / ReduceLROnPlateau / ModelCheckpoint trio.

    All three monitor validation loss, which is the most reliable single
    signal for this imbalanced-then-augmented binary classification task.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=rlr_factor,
            patience=rlr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train_model(
    model: keras.Model,
    X_train,
    y_train,
    X_val,
    y_val,
    callbacks: list[keras.callbacks.Callback],
    epochs: int = 100,
    batch_size: int = 32,
    verbose: int = 1,
) -> keras.callbacks.History:
    """Fit the model with validation monitoring. Thin wrapper kept for readability."""
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
