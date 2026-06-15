# 💵 Banknote Authentication: Real vs. Fake Detection

A robust machine learning solution designed to **distinguish authentic banknotes from counterfeit ones** using statistical features extracted from banknote images. The project utilizes advanced wavelet transform metrics combined with multi-scaler preprocessing pipelines to build a highly accurate classification system.

---

## 🎯 Key Features
- **Exploratory Data Analysis (EDA):** Detailed statistical distribution analysis and pair-plot visualizations of structural image features.
- **Wavelet Feature Utilization:** Leverages high-impact features extracted from image wavelet transforms to capture subtle textures and patterns.
- **Advanced Feature Scaling:** Benchmarks model performance across `StandardScaler`, `RobustScaler`, and `QuantileTransformer` to identify the optimal data distribution for the classification engine.
- **Data-Driven Diagnostics:** Implements a strict validation pipeline to guarantee flawless classification boundaries with minimal margin for error.

---

## 📊 Dataset Structure
The project processes a structured dataset (`data_banknote_authentication.csv`) containing structural features derived from continuous Wavelet Transformed images of banknotes. Key features analyzed include:
- `Variance_Wavelet`: Variance of the Wavelet Transformed image.
- `Skewness_Wavelet`: Skewness of the Wavelet Transformed image.
- `Curtosis_Wavelet`: Kurtosis (curtosis) of the Wavelet Transformed image.
- `Entropy_Image`: Entropy of the raw banknote image.
- `Class`: The target label (`0` for authentic/real banknotes, `1` for counterfeit/fake banknotes).

---

## 🛠️ Data Pipeline & Preprocessing

### 1. Multi-Scaler Assessment
To maximize model stability and handle potential variance across the continuous image metrics, the dataset is tested against three scaling approaches:
- **StandardScaler:** Standardizes features by centering the mean and scaling to unit variance.
- **RobustScaler:** Minimizes the impact of structural image anomalies or extreme values using interquartile ranges (IQR).
- **QuantileTransformer:** Maps features to a normal distribution, smoothing out non-linear variances across the wavelet domains.

### 2. Validation Framework
The records are divided into distinct training and testing matrices via `train_test_split` to allow strict, unbiased performance evaluation before final deployment.

---

## 🤖 Modeling & Performance Evaluation

The classification engine maps mathematical wavelet properties against the biological/structural authenticity profile of the printed money.

### 📈 Core Metrics Covered
Given the zero-tolerance nature of financial fraud and counterfeit detection, models are evaluated against rigorous performance standards:
- **Accuracy:** The overall percentage of correct authentication decisions.
- **Precision:** The reliability rate when the model flags a banknote as counterfeit.
- **Recall (Sensitivity):** The critical financial metric—ensuring that no fake banknote accidentally passes through the system as genuine.
- **F1-Score:** The harmonic balance between precision and recall to verify model stability across both classes.

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python installed on your system. Install the baseline data-science dependencies via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
