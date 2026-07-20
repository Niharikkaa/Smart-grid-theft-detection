# Cyber-Resilient Electricity Theft Detection Using TimeGAN, FFT, and GRU

## Overview

Electricity theft is a major challenge for power utilities, leading to significant financial losses and reduced grid reliability. Detecting fraudulent electricity consumption is difficult because theft patterns often resemble legitimate consumption behavior. Additionally, smart-meter datasets are highly imbalanced, with honest consumers significantly outnumbering theft cases, making it difficult for machine learning models to learn representative fraud patterns.

This project presents a deep learning framework for electricity theft detection that combines **TimeGAN**, **Fast Fourier Transform (FFT)**, and **Gated Recurrent Units (GRU)**. The proposed pipeline generates realistic synthetic theft samples to address class imbalance, extracts frequency-domain features using FFT, and classifies electricity consumption sequences using a GRU-based neural network.

---

## Problem Statement

Electricity theft detection faces two major challenges:

- **Class imbalance**, where theft samples are much fewer than honest samples.
- **Complex temporal consumption patterns**, making fraudulent behavior difficult to distinguish from normal usage.

Traditional machine learning models often struggle with these challenges, resulting in reduced detection accuracy and poor generalization.

---

## Objectives

The objectives of this project are to:

- Preprocess smart-meter electricity consumption data.
- Convert consumption records into sequential time-series data.
- Generate realistic synthetic theft samples using TimeGAN.
- Apply FFT to extract frequency-domain representations.
- Train a GRU classifier for electricity theft detection.
- Evaluate model performance using standard classification metrics.

---

## Proposed Framework

```
Raw Smart Meter Data
        │
        ▼
Data Exploration & Visualization
        │
        ▼
Data Preprocessing
        │
        ▼
Time-Series Sequence Preparation
        │
        ▼
TimeGAN Training
        │
        ▼
Synthetic Theft Sequence Generation
        │
        ▼
FFT Feature Engineering
        │
        ▼
GRU-Based Classification
        │
        ▼
Model Evaluation
```

---

## Project Structure

```
Smart-grid-theft-detection/

├── notebooks/
│   ├── 01_Data_Exploration_Visualization.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_TimeSeries_Preparation.ipynb
│   ├── Notebook4_TimeGAN.ipynb
│   ├── Notebook5_FFT.ipynb
│   ├── Notebook6_GRU.ipynb
│   └── Notebook7_Evaluation.ipynb
│
├── src/
│   ├── classifier/
│   ├── fft_pipeline/
│   └── timegan/
│
├── models/
│
├── plots/
│
├── results/
│
├── requirements.txt
└── README.md
```

---

## Dataset

The project uses the **State Grid Corporation of China (SGCC)** electricity consumption dataset.

Dataset characteristics:

- **42,372** smart-meter consumers
- **1,035** daily electricity consumption readings
- Binary classification:
  - Honest consumers
  - Electricity theft consumers

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- TimeGAN
- Fast Fourier Transform (FFT)

---

## Methodology

### 1. Data Exploration

- Dataset visualization
- Missing value analysis
- Class distribution analysis

### 2. Data Preprocessing

- Data cleaning
- Missing value imputation
- Feature scaling

### 3. Time-Series Preparation

- Conversion of consumption records into sequential samples
- Dataset preparation for TimeGAN and GRU

### 4. TimeGAN

- Train TimeGAN using theft samples
- Generate synthetic theft sequences
- Reduce class imbalance

### 5. FFT Feature Engineering

- Apply Fast Fourier Transform
- Remove high-frequency noise
- Extract frequency-domain information

### 6. GRU Classifier

- Train a GRU neural network on FFT-transformed sequences
- Perform binary classification:
  - Honest
  - Theft

### 7. Evaluation

Evaluate the model using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve

---

## How to Run

Clone the repository

```bash
git clone https://github.com/Niharikkaa/Smart-grid-theft-detection.git
cd Smart-grid-theft-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebooks in the following order:

1. Data Exploration
2. Data Preprocessing
3. Time-Series Preparation
4. TimeGAN
5. FFT Feature Engineering
6. GRU Training
7. Evaluation

---

## Results

The proposed framework successfully:

- Addresses class imbalance using TimeGAN.
- Learns temporal electricity consumption patterns.
- Extracts informative frequency-domain features using FFT.
- Classifies electricity theft using a GRU-based deep learning model.
- Evaluates performance using multiple classification metrics and visualization plots.

---

## Authors

**Niharika Bansal - 095**
**Nancy -090**

B.Tech: Electronics & Communication Engineering - Artificial Intelligence

Indira Gandhi Delhi Technical University for Women (IGDTUW)

---

## License

This project is intended for academic and research purposes.
