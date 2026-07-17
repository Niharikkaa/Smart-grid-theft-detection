"""Three-phase TimeGAN training loop (Yoon et al., 2019), Keras 3 / eager-mode.

Phase 1 — Embedder + Recovery pretraining (autoencoding reconstruction).
Phase 2 — Supervisor pretraining (next-step latent dynamics), on real
          embeddings only.
Phase 3 — Joint adversarial training of Generator/Supervisor, Embedder/
          Recovery, and Discriminator, alternating per the original
          algorithm (generator/supervisor updated twice per outer step,
          discriminator only updated when its loss exceeds a threshold
          so it does not overpower the generator).

Everything here runs under ``tf.GradientTape`` in eager mode (optionally
JIT-compiled per step via ``tf.function``); there is no
``tf.compat.v1.Session``, placeholder, or ``tf.compat.v1`` symbol anywhere
in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import tensorflow as tf

from . import losses
from .models import Discriminator, Embedder, Generator, Recovery, Supervisor


@dataclass
class TimeGANConfig:
    """Hyperparameters for a :class:`TimeGAN` instance.

    Attributes
    ----------
    feature_dim:
        Number of features per timestep in the real data (``1`` for the
        univariate electricity-consumption series here).
    seq_len:
        Number of timesteps per sequence (``1033`` for this dataset).
    hidden_dim:
        Width of the shared latent space used by every network.
    num_layers:
        Number of stacked GRU layers in Embedder/Recovery/Generator/
        Discriminator (the Supervisor uses ``num_layers - 1``, per the
        original paper).
    z_dim:
        Dimensionality of the per-timestep noise vector fed to the
        Generator. Defaults to ``feature_dim`` as in the paper.
    gamma:
        Weight applied to the discriminator/generator loss terms that
        involve the *raw* generator output ``E_hat`` (pre-supervisor).
    supervised_weight, moment_weight:
        Weights on the generator's supervised and moment-matching loss
        terms (Phase 3).
    embedder_supervised_weight:
        Weight on the small supervised regulariser added to the
        embedder's loss during Phase 3.
    discriminator_loss_threshold:
        The discriminator is only updated on a Phase-3 step if its loss
        exceeds this value, preventing it from overpowering the
        generator (as in the original implementation).
    learning_rate:
        Shared Adam learning rate for all five optimizers.
    """

    feature_dim: int
    seq_len: int
    hidden_dim: int = 64
    num_layers: int = 3
    z_dim: int | None = None
    gamma: float = 1.0
    supervised_weight: float = 100.0
    moment_weight: float = 100.0
    embedder_supervised_weight: float = 0.1
    discriminator_loss_threshold: float = 0.15
    learning_rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.z_dim is None:
            self.z_dim = self.feature_dim


class TimeGAN:
    """Owns the five TimeGAN networks, their optimizers, and the training loop."""

    def __init__(self, config: TimeGANConfig, seed: int | None = 42):
        self.config = config
        self._rng = np.random.default_rng(seed)

        c = config
        self.embedder = Embedder(c.hidden_dim, c.num_layers)
        self.recovery = Recovery(c.feature_dim, c.hidden_dim, c.num_layers)
        self.generator = Generator(c.hidden_dim, c.num_layers)
        self.supervisor = Supervisor(c.hidden_dim, c.num_layers)
        self.discriminator = Discriminator(c.hidden_dim, c.num_layers)

        self.embedder_optimizer = tf.keras.optimizers.Adam(
    learning_rate=c.learning_rate,
    clipnorm=1.0,
    )

        self.recovery_optimizer = tf.keras.optimizers.Adam(
    learning_rate=c.learning_rate,
    clipnorm=1.0,
    )

        self.generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=c.learning_rate,
    clipnorm=1.0,
    )

        self.supervisor_optimizer = tf.keras.optimizers.Adam(
    learning_rate=c.learning_rate,
    clipnorm=1.0,
    )

        self.discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=c.learning_rate,
    clipnorm=1.0,
    )

        # Build all networks once so `.trainable_variables` is populated
        # before the first `tf.function`-traced train step.
        self._build(batch_size=2)

        self.history: Dict[str, List[float]] = {
            "phase1_reconstruction_loss": [],
            "phase2_supervised_loss": [],
            "phase3_generator_loss": [],
            "phase3_discriminator_loss": [],
            "phase3_embedder_loss": [],
        }

    def _build(self, batch_size: int) -> None:
        c = self.config
        dummy_x = tf.zeros((batch_size, c.seq_len, c.feature_dim))
        dummy_z = tf.zeros((batch_size, c.seq_len, c.z_dim))
        h = self.embedder(dummy_x)
        self.recovery(h)
        e_hat = self.generator(dummy_z)
        self.supervisor(h)
        self.discriminator(e_hat)

    def _sample_z(self, batch_size: int) -> tf.Tensor:
        """Sample per-timestep noise, uniform on [0, 1] as in the original paper."""
        c = self.config
        return tf.random.uniform(
            (batch_size, c.seq_len, c.z_dim), minval=0.0, maxval=1.0, dtype=tf.float32
        )

    # ------------------------------------------------------------------
    # Phase 1: Embedder + Recovery pretraining
    # ------------------------------------------------------------------
    @tf.function
    def _phase1_step(self, x_real: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            h = self.embedder(x_real, training=True)
            x_tilde = self.recovery(h, training=True)
            e_loss_t0 = losses.reconstruction_loss(x_real, x_tilde)
            e_loss0 = 10.0 * tf.sqrt(e_loss_t0 + 1e-8)
        variables = self.embedder.trainable_variables + self.recovery.trainable_variables
        grads = tape.gradient(e_loss0, variables)
        self.embedder_optimizer.apply_gradients(zip(grads, variables))
        return e_loss_t0

    def pretrain_embedder(self, dataset: tf.data.Dataset, epochs: int, verbose: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for x_real in dataset:
                loss = self._phase1_step(x_real)
                epoch_losses.append(float(loss))
            mean_loss = float(np.mean(epoch_losses))
            self.history["phase1_reconstruction_loss"].append(mean_loss)
            if verbose:
                print(f"[Phase 1 | Embedder] epoch {epoch}/{epochs} - reconstruction_loss={mean_loss:.6f}")

    # ------------------------------------------------------------------
    # Phase 2: Supervisor pretraining
    # ------------------------------------------------------------------
    @tf.function
    def _phase2_step(self, x_real: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            h = self.embedder(x_real, training=False)
            h_hat_supervise = self.supervisor(h, training=True)
            g_loss_s = losses.supervised_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
        # NOTE: the original TF1 implementation minimizes G_loss_S with an
        # optimizer whose var_list includes both the supervisor's and the
        # generator's variables. The generator, however, never appears in
        # this forward pass, so its gradient is identically zero — that
        # inclusion is a no-op in the original code. We update only the
        # supervisor's variables here, which is functionally identical.
        variables = self.supervisor.trainable_variables
        grads = tape.gradient(g_loss_s, variables)
        self.supervisor_optimizer.apply_gradients(zip(grads, variables))
        return g_loss_s

    def pretrain_supervisor(self, dataset: tf.data.Dataset, epochs: int, verbose: int = 1) -> None:
        for epoch in range(1, epochs + 1):
            epoch_losses = []
            for x_real in dataset:
                loss = self._phase2_step(x_real)
                epoch_losses.append(float(loss))
            mean_loss = float(np.mean(epoch_losses))
            self.history["phase2_supervised_loss"].append(mean_loss)
            if verbose:
                print(f"[Phase 2 | Supervisor] epoch {epoch}/{epochs} - supervised_loss={mean_loss:.6f}")

    # ------------------------------------------------------------------
    # Phase 3: Joint adversarial training
    # ------------------------------------------------------------------
    @tf.function
    def _phase3_generator_step(self, x_real: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        c = self.config
        with tf.GradientTape() as tape:
            h = self.embedder(x_real, training=False)
            e_hat = self.generator(z, training=True)
            h_hat = self.supervisor(e_hat, training=True)
            h_hat_supervise = self.supervisor(h, training=True)
            x_hat = self.recovery(h_hat, training=False)

            y_fake = self.discriminator(h_hat, training=False)
            y_fake_e = self.discriminator(e_hat, training=False)

            g_loss_u = losses.generator_adversarial_loss(y_fake, y_fake_e, gamma=c.gamma)
            g_loss_s = losses.supervised_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
            g_loss_v = losses.moment_matching_loss(x_real, x_hat)

            g_loss = losses.generator_total_loss(
                g_loss_u, g_loss_s, g_loss_v,
                supervised_weight=c.supervised_weight,
                moment_weight=c.moment_weight,
            )
        variables = self.generator.trainable_variables + self.supervisor.trainable_variables
        grads = tape.gradient(g_loss, variables)
        self.generator_optimizer.apply_gradients(zip(grads, variables))
        return g_loss

    @tf.function
    def _phase3_embedder_step(self, x_real: tf.Tensor) -> tf.Tensor:
        c = self.config
        with tf.GradientTape() as tape:
            h = self.embedder(x_real, training=True)
            x_tilde = self.recovery(h, training=True)
            h_hat_supervise = self.supervisor(h, training=False)

            e_loss_t0 = losses.reconstruction_loss(x_real, x_tilde)
            g_loss_s = losses.supervised_loss(h[:, 1:, :], h_hat_supervise[:, :-1, :])
            e_loss = losses.embedder_total_loss(
                e_loss_t0, g_loss_s, supervised_weight=c.embedder_supervised_weight
            )
        variables = self.embedder.trainable_variables + self.recovery.trainable_variables
        grads = tape.gradient(e_loss, variables)
        self.embedder_optimizer.apply_gradients(zip(grads, variables))
        return e_loss

    @tf.function
    def _phase3_discriminator_step(self, x_real: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        c = self.config
        with tf.GradientTape() as tape:
            h = self.embedder(x_real, training=False)
            e_hat = self.generator(z, training=False)
            h_hat = self.supervisor(e_hat, training=False)

            y_real = self.discriminator(h, training=True)
            y_fake = self.discriminator(h_hat, training=True)
            y_fake_e = self.discriminator(e_hat, training=True)

            d_loss = losses.discriminator_loss(y_real, y_fake, y_fake_e, gamma=c.gamma)
        variables = self.discriminator.trainable_variables
        grads = tape.gradient(d_loss, variables)
        self.discriminator_optimizer.apply_gradients(zip(grads, variables))
        return d_loss

    @tf.function
    def _phase3_discriminator_loss_only(self, x_real: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        c = self.config
        h = self.embedder(x_real, training=False)
        e_hat = self.generator(z, training=False)
        h_hat = self.supervisor(e_hat, training=False)
        y_real = self.discriminator(h, training=False)
        y_fake = self.discriminator(h_hat, training=False)
        y_fake_e = self.discriminator(e_hat, training=False)
        return losses.discriminator_loss(y_real, y_fake, y_fake_e, gamma=c.gamma)

    def train_joint(self, dataset: tf.data.Dataset, epochs: int, batch_size: int, verbose: int = 1) -> None:
        """Run Phase 3 (joint adversarial training) for ``epochs`` passes over the data."""
        for epoch in range(1, epochs + 1):
            g_losses, d_losses, e_losses = [], [], []
            for x_real in dataset:
                # Generator and Supervisor are updated twice per outer
                # step, then the Embedder once, matching the original
                # algorithm's ratio between the two update blocks.
                for _ in range(2):
                    z = self._sample_z(batch_size)
                    g_loss = self._phase3_generator_step(x_real, z)
                    e_loss = self._phase3_embedder_step(x_real)

                z = self._sample_z(batch_size)
                d_loss_check = float(self._phase3_discriminator_loss_only(x_real, z))
                if d_loss_check > self.config.discriminator_loss_threshold:
                    z = self._sample_z(batch_size)
                    d_loss = float(self._phase3_discriminator_step(x_real, z))
                else:
                    d_loss = d_loss_check

                g_losses.append(float(g_loss))
                e_losses.append(float(e_loss))
                d_losses.append(d_loss)

            mean_g, mean_d, mean_e = float(np.mean(g_losses)), float(np.mean(d_losses)), float(np.mean(e_losses))
            self.history["phase3_generator_loss"].append(mean_g)
            self.history["phase3_discriminator_loss"].append(mean_d)
            self.history["phase3_embedder_loss"].append(mean_e)
            if verbose:
                print(
                    f"[Phase 3 | Joint] epoch {epoch}/{epochs} - "
                    f"g_loss={mean_g:.4f} d_loss={mean_d:.4f} e_loss={mean_e:.4f}"
                )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def fit(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        phase1_epochs: int = 50,
        phase2_epochs: int = 50,
        phase3_epochs: int = 100,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Run all three TimeGAN training phases in sequence."""
        self.pretrain_embedder(dataset, phase1_epochs, verbose=verbose)
        self.pretrain_supervisor(dataset, phase2_epochs, verbose=verbose)
        self.train_joint(dataset, phase3_epochs, batch_size=batch_size, verbose=verbose)
        return self.history

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @tf.function
    def _generate_batch(self, z: tf.Tensor) -> tf.Tensor:
        e_hat = self.generator(z, training=False)
        h_hat = self.supervisor(e_hat, training=False)
        x_hat = self.recovery(h_hat, training=False)
        return x_hat

    def generate(self, num_samples: int, batch_size: int = 64) -> np.ndarray:
        """Generate ``num_samples`` synthetic sequences via Generator→Supervisor→Recovery."""
        c = self.config
        outputs = []
        remaining = num_samples
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            z = self._sample_z(current_batch)
            x_hat = self._generate_batch(z)
            outputs.append(x_hat.numpy())
            remaining -= current_batch
        return np.concatenate(outputs, axis=0).astype(np.float32)
