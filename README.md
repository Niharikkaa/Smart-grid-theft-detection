# Cyber-Resilient Electricity Theft Detection Using TimeGAN, FFT, and GRU

## Overview

Electricity theft is a major challenge for power utilities because fraudulent consumption patterns can be difficult to distinguish from legitimate usage. Machine learning-based detection is further affected by class imbalance, where normal electricity-consumption samples are substantially more numerous than theft samples.

This project proposes a deep learning framework that combines synthetic time-series generation, frequency-domain feature extraction, and sequential deep learning for electricity theft detection. The framework uses TimeGAN to generate synthetic electricity-theft sequences, Fast Fourier Transform (FFT) to extract frequency-domain features, and a Gated Recurrent Unit (GRU) classifier to distinguish between honest and theft-related consumption patterns.

---

## Problem Statement

Electricity theft detection is challenging because theft patterns can be complex, irregular, and difficult to identify from raw smart-meter time-series data. In addition, the limited availability of theft samples creates class imbalance, which can reduce the ability of machine learning models to learn diverse fraudulent consumption behaviour.

Therefore, there is a need for a framework that can:

- address the imbalance between honest and theft samples,
- learn temporal patterns in electricity consumption,
- generate additional realistic theft sequences,
- extract informative features from time-series data, and
- accurately classify electricity consumption as honest or theft-related.

---

## Objective

The objective of this project is to develop a cyber-resilient electricity theft detection framework that:

1. Preprocesses and prepares smart-meter electricity consumption data as time-series sequences.
2. Uses TimeGAN to generate synthetic electricity-theft sequences.
3. Applies Fast Fourier Transform (FFT) to transform time-series data into frequency-domain representations.
4. Uses a GRU-based deep learning classifier to detect electricity theft.
5. Evaluates the classification performance using standard machine learning metrics.

---

## Proposed Framework

The proposed framework consists of the following stages:

```text
Raw Smart-Meter Data
        ↓
Data Exploration and Visualization
        ↓
Data Preprocessing
        ↓
Time-Series Sequence Preparation
        ↓
TimeGAN Training
        ↓
Synthetic Theft Sequence Generation
        ↓
FFT Feature Engineering
        ↓
GRU-Based Classification
        ↓
Model Evaluation
