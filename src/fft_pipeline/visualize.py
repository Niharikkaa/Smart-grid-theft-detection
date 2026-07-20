"""Publication-quality plots for the Notebook 5 FFT preprocessing step."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .filtering import magnitude_spectrum

PUBLICATION_RCPARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_publication_style() -> None:
    """Apply a consistent, publication-friendly matplotlib style."""
    plt.rcParams.update(PUBLICATION_RCPARAMS)


def plot_original_vs_filtered(
    original: np.ndarray,
    filtered: np.ndarray,
    title: str = "Original vs. FFT-Filtered Signal",
    save_path: Path | None = None,
) -> None:
    """Overlay the raw and low-pass-filtered version of one sequence."""
    original = np.asarray(original).reshape(-1)
    filtered = np.asarray(filtered).reshape(-1)
    t = np.arange(len(original))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, original, color="#9aa5b1", linewidth=1.0, label="Original", alpha=0.8)
    ax.plot(t, filtered, color="#1f77b4", linewidth=1.6, label="Filtered (low-pass)")
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalized consumption")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_magnitude_spectra(
    original: np.ndarray,
    filtered: np.ndarray,
    cutoff_ratio: float,
    title: str = "FFT Magnitude Spectrum: Before vs. After Filtering",
    save_path: Path | None = None,
) -> None:
    """Plot the magnitude spectrum of a sequence before and after filtering.

    A vertical dashed line marks the cutoff bin so the reader can see
    exactly which frequency components were removed.
    """
    freq_before, mag_before = magnitude_spectrum(original)
    freq_after, mag_after = magnitude_spectrum(filtered)
    cutoff_bin = int(np.ceil(cutoff_ratio * len(freq_before)))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(freq_before, mag_before, color="#9aa5b1", linewidth=1.0, label="Before filtering", alpha=0.8)
    ax.plot(freq_after, mag_after, color="#d62728", linewidth=1.4, label="After filtering")
    ax.axvline(cutoff_bin, color="black", linestyle="--", linewidth=1.0, label=f"Cutoff (bin {cutoff_bin})")
    ax.set_title(title)
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Magnitude (dB)")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_class_examples(
    honest_original: np.ndarray,
    honest_filtered: np.ndarray,
    theft_original: np.ndarray,
    theft_filtered: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Side-by-side original-vs-filtered examples for one honest and one theft sequence."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    for ax, original, filtered, label in (
        (axes[0], honest_original, honest_filtered, "Honest (label = 0)"),
        (axes[1], theft_original, theft_filtered, "Theft (label = 1)"),
    ):
        original = np.asarray(original).reshape(-1)
        filtered = np.asarray(filtered).reshape(-1)
        t = np.arange(len(original))
        ax.plot(t, original, color="#9aa5b1", linewidth=1.0, label="Original", alpha=0.8)
        ax.plot(t, filtered, color="#2ca02c", linewidth=1.6, label="Filtered")
        ax.set_title(label)
        ax.set_ylabel("Normalized consumption")
        ax.legend(loc="upper right", frameon=False)

    axes[-1].set_xlabel("Day")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
