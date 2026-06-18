# Student Performance Prediction — Machine Learning

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> **Bottom line:** Built an end-to-end ML pipeline to predict student academic performance. Ridge Regression achieved **R² = 0.72**, reducing prediction error by **~48% vs. a baseline model** — outperforming Linear Regression, KNN, and a mean-prediction baseline.

---

## Project Overview

This project applies supervised machine learning to predict students' final mathematics grades (G3) using demographic, academic, social, and lifestyle features from the UCI Student Performance dataset. Four models were trained, evaluated, and compared to identify the best-performing approach.

---

## Results

| Model                      | MSE      | RMSE     | R² Score |
| -------------------------- | -------- | -------- | -------- |
| Baseline (mean prediction) | 21.00    | 4.58     | 0.00     |
| Linear Regression          | 5.66     | 2.38     | 0.72     |
| **Ridge Regression**       | **5.64** | **2.38** | **0.72** |
| KNN Regression             | 11.41    | 3.38     | 0.44     |

**Winner: Ridge Regression** — lowest MSE, highest R², improved stability through L2 regularization, ~48% reduction in prediction error vs. baseline.

---

## ML Pipeline

1. **Data Loading** — UCI Student Performance dataset (395 students, 33 features, 0 missing values)
2. **Preprocessing** — One-hot encoding for categorical variables, StandardScaler for numerical features, train/test split
3. **Model Training** — Baseline, Linear Regression, Ridge Regression, KNN Regression
4. **Evaluation** — MSE, RMSE, R² across all models
5. **PCA Analysis** — Dimensionality reduction and variance visualization

---

## Dataset

**Source:** [UCI Machine Learning Repository — Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)

| Property       | Value            |
| -------------- | ---------------- |
| Students       | 395              |
| Features       | 33               |
| Target         | G3 (Final Grade) |
| Missing Values | 0                |

Feature categories: demographic info, family background, academic history, social activities, lifestyle habits, prior grades (G1, G2).

---

## Repository Structure

```
Student-performance-prediction/
├── data/
│   ├── raw/student-mat.csv
│   └── processed/
├── src/
│   └── student_performance.py
├── notebooks/
├── results/
│   ├── figures/
│   └── tables/
├── reports/
├── README.md
└── requirements.txt
```

---

## Installation & Usage

```bash
# Clone the repo
git clone https://github.com/elsid01/Student-performance-prediction.git
cd Student-performance-prediction

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/student_performance.py
```

The script loads data, preprocesses, trains all models, evaluates performance, generates visualizations, and saves results.

---

## Technologies

`Python` `Pandas` `NumPy` `Scikit-Learn` `Matplotlib` `Seaborn`

---

## Key Takeaways

- Ridge Regression slightly outperformed standard Linear Regression due to L2 regularization reducing overfitting
- KNN performed significantly worse in this high-dimensional feature space
- PCA showed the top 2 components explained only ~15.5% of variance, confirming the value of the full feature set for prediction

---

_Project completed May 2026 — UNLV CS422 Machine Learning course._
