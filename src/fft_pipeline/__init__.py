"""FFT-based preprocessing pipeline for Notebook 5.

Loads the TimeGAN outputs from Notebook 4, merges real + synthetic theft
sequences, applies a low-pass FFT filter to denoise every sequence while
staying in the time domain, and writes the final classifier-ready dataset
consumed by Notebook 6.
"""
