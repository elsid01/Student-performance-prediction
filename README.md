# Student Performance Prediction Using Machine Learning

## Overview

This project applies machine learning techniques to predict students' final mathematics grades using demographic, academic, social, and lifestyle factors. The project uses the Student Performance dataset from the UCI Machine Learning Repository and compares multiple regression models to determine which approach provides the best predictive performance.

The project was developed as part of a CS422 Machine Learning course project.

---

## Project Objectives

- Predict students' final mathematics grade (G3)
- Explore the impact of student characteristics on academic performance
- Compare multiple machine learning algorithms
- Evaluate model performance using quantitative metrics
- Apply dimensionality reduction techniques for data analysis and visualization

---

## Dataset

**Dataset:** Student Performance Dataset

**Source:** UCI Machine Learning Repository

### Dataset Statistics

| Property        | Value            |
| --------------- | ---------------- |
| Students        | 395              |
| Features        | 33               |
| Target Variable | G3 (Final Grade) |
| Missing Values  | 0                |

### Feature Categories

- Demographic information
- Family background
- Academic history
- Social activities
- Lifestyle habits
- Previous grades (G1 and G2)

---

## Machine Learning Pipeline

### 1. Data Loading

- Load the mathematics dataset
- Explore dataset structure and feature types

### 2. Data Preprocessing

- Handle categorical variables using One-Hot Encoding
- Scale numerical features using StandardScaler
- Split data into training and testing sets

### 3. Model Development

Implemented the following models:

#### Baseline Model

Predicts the average training grade for all students.

#### Linear Regression

Models linear relationships between features and final grades.

#### Ridge Regression

Applies L2 regularization to improve model stability and reduce overfitting.

#### K-Nearest Neighbors (KNN) Regression

Predicts grades using the nearest neighboring students in feature space.

### 4. PCA Analysis

- Reduce dimensionality
- Analyze explained variance
- Visualize data structure

### 5. Evaluation

Models are evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-Squared (R²)

---

## Project Results

| Model             | MSE   | RMSE | R² Score |
| ----------------- | ----- | ---- | -------- |
| Baseline          | 21.00 | 4.58 | 0.00     |
| Linear Regression | 5.66  | 2.38 | 0.72     |
| Ridge Regression  | 5.64  | 2.38 | 0.72     |
| KNN Regression    | 11.41 | 3.38 | 0.44     |

### Best Performing Model

**Ridge Regression**

Reasons:

- Lowest Mean Squared Error
- Highest R² Score
- Improved stability through regularization

---

## PCA Results

| Metric                   | Value          |
| ------------------------ | -------------- |
| Principal Components     | 2              |
| Explained Variance Ratio | [0.091, 0.065] |
| Total Explained Variance | 15.55%         |

---

## Repository Structure

```text
Student-performance-prediction/
│
├── data/
│   ├── raw/
│   │   └── student-mat.csv
│   └── processed/
│
├── src/
│   └── student_performance.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── reports/
│   ├── final_report.docx
│   └── presentation.pptx
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd Student-performance-prediction
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Execute the main script:

```bash
python src/student_performance.py
```

The script will:

- Load the dataset
- Perform preprocessing
- Train all models
- Evaluate performance
- Generate visualizations
- Save comparison results

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

---

## Key Insights

- Machine learning can effectively predict student performance.
- Linear models performed best on this dataset.
- Ridge Regression slightly outperformed standard Linear Regression.
- KNN Regression showed lower performance in the high-dimensional feature space.
- PCA provided useful dimensionality reduction and visualization capabilities.

---

## Future Improvements

- Hyperparameter tuning
- Cross-validation
- Additional machine learning models
- Feature selection techniques
- Larger educational datasets

---

## Authors

### Ely Sidibe

Computer Science Graduate

### Brain Sands

CS422 Machine Learning Project

---

## License

This project is intended for educational and academic purposes.
