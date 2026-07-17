"""Loss functions for the three TimeGAN training phases.

Implements, faithfully to Yoon et al. (2019):

* Reconstruction loss (Phase 1: Embedder + Recovery)
* Supervised loss (Phase 2: Supervisor; also used inside Phase 1/3)
* Unsupervised adversarial losses (Phase 3: Generator + Discriminator)
* Moment-matching loss (Phase 3: Generator), which penalises mismatches
  between the mean and standard deviation of real vs. synthetic data so
  the generator does not just win the adversarial game locally while
  drifting in aggregate statistics.
"""

from __future__ import annotations

import tensorflow as tf


def reconstruction_loss(x_real: tf.Tensor, x_tilde: tf.Tensor) -> tf.Tensor:
    """Mean squared error between real data and its embed→recover round trip."""
    return tf.reduce_mean(tf.square(x_real - x_tilde))


def supervised_loss(h_real_next: tf.Tensor, h_hat_supervise: tf.Tensor) -> tf.Tensor:
    """Next-step latent prediction error.

    ``h_real_next`` is the real latent sequence shifted one step forward
    (i.e. ``H[:, 1:, :]``); ``h_hat_supervise`` is the supervisor's
    prediction from the *unshifted* sequence, truncated to align
    (i.e. ``Supervisor(H)[:, :-1, :]``). Alignment is handled by the
    caller so this function stays a plain MSE.
    """
    return tf.reduce_mean(tf.square(h_real_next - h_hat_supervise))


def moment_matching_loss(x_real: tf.Tensor, x_synth: tf.Tensor) -> tf.Tensor:
    """Penalise mismatch between first and second moments of real vs synthetic data.

    Computed over the batch and time axes (axis 0, 1), per feature channel,
    matching the original paper's ``G_loss_V1`` / ``G_loss_V2`` terms.
    """
    real_mean, real_var = tf.nn.moments(x_real, axes=[0, 1])
    synth_mean, synth_var = tf.nn.moments(x_synth, axes=[0, 1])

    real_std = tf.sqrt(real_var + 1e-6)
    synth_std = tf.sqrt(synth_var + 1e-6)

    g_loss_std = tf.reduce_mean(tf.abs(real_std - synth_std))
    g_loss_mean = tf.reduce_mean(tf.abs(real_mean - synth_mean))
    return g_loss_std + g_loss_mean


def _bce_with_logits(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    )


def discriminator_loss(
    d_real_logits: tf.Tensor,
    d_fake_logits: tf.Tensor,
    d_fake_embedding_logits: tf.Tensor,
    gamma: float = 1.0,
) -> tf.Tensor:
    """Discriminator adversarial loss over three sources.

    * ``d_real_logits``: D(H) on real embedded sequences (label = 1)
    * ``d_fake_logits``: D(H_hat) on generator→supervisor sequences (label = 0)
    * ``d_fake_embedding_logits``: D(E_hat) on raw generator output,
      pre-supervisor (label = 0). The original paper weights this term by
      ``gamma`` (default 1.0) since it is a slightly easier fake to spot.
    """
    loss_real = _bce_with_logits(tf.ones_like(d_real_logits), d_real_logits)
    loss_fake = _bce_with_logits(tf.zeros_like(d_fake_logits), d_fake_logits)
    loss_fake_e = _bce_with_logits(
        tf.zeros_like(d_fake_embedding_logits), d_fake_embedding_logits
    )
    return loss_real + loss_fake + gamma * loss_fake_e


def generator_adversarial_loss(
    d_fake_logits: tf.Tensor, d_fake_embedding_logits: tf.Tensor, gamma: float = 1.0
) -> tf.Tensor:
    """Generator's adversarial loss: fool the discriminator on both fake sources."""
    loss_fake = _bce_with_logits(tf.ones_like(d_fake_logits), d_fake_logits)
    loss_fake_e = _bce_with_logits(
        tf.ones_like(d_fake_embedding_logits), d_fake_embedding_logits
    )
    return loss_fake + gamma * loss_fake_e


def generator_total_loss(
    g_loss_adversarial: tf.Tensor,
    g_loss_supervised: tf.Tensor,
    g_loss_moments: tf.Tensor,
    supervised_weight: float = 100.0,
    moment_weight: float = 100.0,
) -> tf.Tensor:
    """Combine the generator's three loss terms with the paper's weighting.

    The supervised term uses ``sqrt`` (as in the original implementation)
    so that the generator is not overwhelmed by supervised loss early in
    training when it can be numerically much larger than the adversarial
    term.
    """
    return (
        g_loss_adversarial
        + supervised_weight * tf.sqrt(g_loss_supervised + 1e-8)
        + moment_weight * g_loss_moments
    )


def embedder_total_loss(
    e_loss_reconstruction: tf.Tensor,
    e_loss_supervised: tf.Tensor,
    supervised_weight: float = 0.1,
) -> tf.Tensor:
    """Combine reconstruction loss with a small supervised regulariser.

    Matches the paper's joint-phase embedder objective:
    ``E_loss = 10 * sqrt(E_loss_T0) + 0.1 * G_loss_S``.
    """
    e_loss0 = 10.0 * tf.sqrt(e_loss_reconstruction + 1e-8)
    return e_loss0 + supervised_weight * e_loss_supervised
