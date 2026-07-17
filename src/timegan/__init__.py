"""TimeGAN: a modern TensorFlow 2 / Keras 3 implementation.

Reference
---------
Yoon, J., Jarrett, D., & van der Schaar, M. (2019).
"Time-series Generative Adversarial Networks." NeurIPS 2019.

This package re-implements the original algorithm (Embedder, Recovery,
Generator, Supervisor, Discriminator; three-phase training) using only
eager-mode Keras 3 APIs (``tf.keras.Model`` subclasses, ``tf.GradientTape``).
No ``tf.compat.v1`` symbols, sessions, or placeholders are used anywhere in
this package.
"""

from .models import Discriminator, Embedder, Generator, Recovery, Supervisor
from .trainer import TimeGAN, TimeGANConfig

__all__ = [
    "Embedder",
    "Recovery",
    "Generator",
    "Supervisor",
    "Discriminator",
    "TimeGAN",
    "TimeGANConfig",
]
