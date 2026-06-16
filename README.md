# Machine Learning-Based Implied Volatility Surface Reconstruction for Options Markets

## Overview

This project focuses on reconstructing missing Implied Volatility (IV) values in NIFTY options market data. The task is formulated as a regression and matrix completion problem, where missing IV observations are estimated using quantitative finance techniques and machine learning models.

The objective is to recover a complete volatility surface while preserving the structural characteristics of option markets such as volatility smiles and skews.

---

## Problem Statement

Options datasets frequently contain missing implied volatility values due to:

- Illiquid strikes
- Sparse trading activity
- Data collection issues
- Market microstructure noise

Missing IV values distort volatility surface analysis and negatively affect pricing, risk management, and quantitative trading models.

This project aims to reconstruct the missing values accurately using a combination of:

- Surface interpolation techniques
- Machine Learning models
- Matrix completion methods

---

## Dataset Description

The dataset contains:

- Timestamp (`datetime`)
- Underlying asset price (`underlying_price`)
- Option contracts across multiple strikes
- Implied Volatility (IV) values

Several IV observations are intentionally removed and must be predicted.

### Example Structure

| datetime | underlying_price | 25200CE | 25300CE | 25400CE |
|-----------|-----------------|----------|----------|----------|
| 09:15 | 26111.65 | 0.11 | 0.12 | NaN |
| 09:20 | 26141.40 | 0.10 | NaN | 0.11 |

---

## Project Workflow

### 1. Data Preprocessing

- Parse option contract names
- Extract strike prices
- Identify option type (Call / Put)
- Compute moneyness

#### Engineered Features

- Strike
- Option Type
- Underlying Price
- Moneyness
- Hour
- Minute
- Distance From ATM
- Absolute Distance From ATM

---

### 2. Surface Interpolation

For each timestamp:

- Construct the IV smile
- Apply spline interpolation across strikes
- Estimate missing values

This provides a strong baseline approximation of the volatility surface.

---

### 3. Machine Learning Model

A LightGBM Regressor is trained using known IV observations.

#### Features

```python
[
    "underlying_price",
    "strike",
    "option_type",
    "moneyness",
    "hour",
    "minute",
    "distance_from_atm",
    "abs_distance_from_atm",
    "spline_pred"
]
```

### Target

```python
iv
```

The spline prediction is included as an additional feature, allowing the model to learn deviations from the interpolated surface.

---

## Matrix Completion (SVD)

To capture global relationships between option contracts:

1. Convert IV data into a matrix format
2. Replace missing values temporarily
3. Apply Singular Value Decomposition (SVD)
4. Reconstruct the volatility surface
5. Recover missing entries

This method exploits the low-rank structure often present in volatility surfaces.

---

## Model Architecture

### Baseline

- Spline Interpolation

### Machine Learning

- LightGBM Regressor

### Matrix Completion

- Truncated SVD

### Ensemble Experiments

- LightGBM + Spline
- SVD Reconstruction
- Hybrid Surface Models

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- LightGBM
- SciPy
- Matplotlib
- Jupyter Notebook

---

## Evaluation Metric

The competition uses:

```text
Mean Squared Error (MSE)
```

Lower scores indicate better reconstruction quality.

---

## Results

| Method | Public Score |
|----------|-------------|
| Initial LightGBM Model | 0.0718 |
| LightGBM + Surface Features | 0.0674 |
| SVD Matrix Completion | Experimental |
| Hybrid Ensemble | In Progress |

The addition of volatility surface features significantly improved prediction accuracy compared to the baseline model.

---

## Key Learnings

- Volatility surfaces exhibit strong structural regularities.
- Surface interpolation provides a powerful prior for missing IV estimation.
- Machine learning models perform better when informed by financial domain features.
- Matrix completion methods can capture latent relationships between option contracts.
- Combining quantitative finance techniques with machine learning leads to more robust predictions.

---

## Future Improvements

- SVI (Stochastic Volatility Inspired) Surface Fitting
- Radial Basis Function (RBF) Interpolation
- Temporal Feature Engineering
- XGBoost and CatBoost Models
- Ensemble Learning
- Deep Learning-based Surface Reconstruction

---

## Repository Structure

```text
.
├── dataset.csv
├── filled_dataset.csv
├── submission.csv
├── lgbm_v1.pkl
├── hello.ipynb
├── svd_v1.ipynb
├── requirements.txt
└── README.md
```

---

## Author

**Akul Katta**

Machine Learning & Quantitative Finance Project

Focus Areas:
- Machine Learning
- Quantitative Finance
- Derivatives Analytics
- Volatility Modeling
- Financial Data Science

---
