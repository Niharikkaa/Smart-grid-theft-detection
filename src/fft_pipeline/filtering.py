"""FFT-based low-pass filtering for electricity consumption sequences.

Design rationale
-----------------
Electricity theft manifests as gradual, sustained deviations in consumption
behaviour (e.g. an artificially suppressed daily baseline, or missing
usage during specific periods) rather than as brief single-day spikes.
That behaviour lives in the **low-frequency** part of the spectrum -
slow trends, weekly cycles (7 days), and seasonal cycles.

Meter noise, one-off misreads and day-to-day jitter, on the other hand,
show up as **high-frequency** energy spread across many spectral bins.

We therefore use a real-valued FFT low-pass filter:

1. ``rfft`` the sequence (assumes a real-valued time series, which
   consumption readings always are - this halves the spectrum we need to
   handle and guarantees a real-valued signal after inverting, with no
   manual conjugate-symmetry bookkeeping).
2. Zero out every frequency bin above ``cutoff_ratio * Nyquist``.
3. ``irfft`` back to the time domain, forcing the original length so the
   output shape always matches the input shape exactly.

A hard rectangular cutoff is used deliberately instead of a Butterworth /
Chebyshev / windowed filter: it is fully explainable in one sentence
("we keep the lowest X% of frequency components"), has a single,
interpretable hyperparameter, and is standard practice for this kind of
academic denoising step. It does introduce mild ringing (Gibbs
phenomenon) at sharp edges, which is an acceptable, well-documented
trade-off for this use case since the classifier consumes the whole
sequence rather than relying on exact edge values.
"""

from __future__ import annotations

import numpy as np


def lowpass_filter_batch(
    sequences: np.ndarray,
    cutoff_ratio: float,
) -> np.ndarray:
    """Apply an FFT low-pass filter to every sequence in a batch.

    Parameters
    ----------
    sequences : np.ndarray
        Array of shape ``(N, T, 1)`` (or ``(N, T)``), real-valued.
    cutoff_ratio : float
        Fraction of the Nyquist frequency to keep, in ``(0, 1]``. This is
        an experiment parameter and is deliberately **not** given a
        default value here — it must be supplied explicitly by the
        caller (Notebook 5), which is where the experiment configuration
        (``CUT_OFF_RATIO``) lives. Keeping this module free of a
        hardcoded default ensures every run's cutoff is traceable to a
        single, visible line in the notebook rather than buried in
        library code.

    Returns
    -------
    np.ndarray
        Filtered sequences with exactly the same shape and dtype
        (``float32``) as the input.
    """
    if not (0.0 < cutoff_ratio <= 1.0):
        raise ValueError(f"cutoff_ratio must be in (0, 1], got {cutoff_ratio}.")

    squeeze_back = False
    if sequences.ndim == 3:
        if sequences.shape[2] != 1:
            raise ValueError(
                f"Expected a single feature channel, got shape {sequences.shape}."
            )
        seq2d = sequences[:, :, 0]
    elif sequences.ndim == 2:
        seq2d = sequences
        squeeze_back = True
    else:
        raise ValueError(f"Expected a 2D or 3D array, got shape {sequences.shape}.")

    n_timesteps = seq2d.shape[1]

    # rfft of a length-T real signal returns T//2 + 1 complex bins,
    # covering frequencies 0 (DC) ... Nyquist.
    spectrum = np.fft.rfft(seq2d, axis=1)
    n_bins = spectrum.shape[1]

    cutoff_bin = np.clip(
    int(np.ceil(cutoff_ratio * n_bins)),1, n_bins
)
    filtered_spectrum = spectrum.copy()
    filtered_spectrum[:, cutoff_bin:] = 0.0

    filtered = np.fft.irfft(filtered_spectrum, n=n_timesteps, axis=1)
    filtered = filtered.astype(np.float32)

    if squeeze_back:
        return filtered
    return filtered[:, :, np.newaxis]


def magnitude_spectrum(sequence_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequency_bin_index, magnitude) for a single 1D sequence.

    Used purely for visualization (before/after filtering spectra).
    Magnitude is reported on a log scale (``20 * log10``) which is the
    standard way to display FFT magnitude spectra, with a small epsilon
    added to avoid ``log(0)``.
    """
    sequence_1d = np.asarray(sequence_1d).reshape(-1)
    spectrum = np.fft.rfft(sequence_1d)
    magnitude = np.abs(spectrum)
    magnitude_db = 20 * np.log10(magnitude + 1e-8)
    freq_bins = np.arange(len(magnitude))
    return freq_bins, magnitude_db
