import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
#=================
# Load Dataset
#=================

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dataset path
DATA_PATH = PROJECT_ROOT / "data"/ "raw"/"student-mat.csv"

# Load dataset
df = pd.read_csv(DATA_PATH, sep=";")

#=================
# Dataset Inspection
#=================

print("Dataset loaded successfully.\n")

print("Dataset shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head(5))

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Types:")
print(df.dtypes)

#==========================
# Define Input and Output
#==========================

#Target variable
print("-------------------------------------")

y = df["G3"]

# Input feature
X = df.drop(columns=["G3"])

print("\nInput/Output Definition")
print("Feature matrix (X) shape:")
print(X.shape)

print("\nTarget vector (y) shape:")
print(y.shape)

print("\nNumber of input features:")
print(X.shape[1])
print("\nTarget variable:")
print(y.name)

#===================================
# Encode Categorical Variables
#===================================

# Apply one-hot encoding

X_encoded = pd.get_dummies(X, drop_first=True)

print("\nEncoded Feature Matrix")
print("-------------------------------------")

print("Encoded X shape:")
print(X_encoded.shape)

print("\nFirst 5 rows of encoded data:")
print(X_encoded.head())

print("\nEncoded data types:")
print(X_encoded.dtypes)


#====================================
# Feature Scaling
#====================================

#Initialize scaler
scaler = StandardScaler()

#Scale feature
X_scaled = scaler.fit_transform(X_encoded)

print("\nFeature scaling")
print("------------------------------------")

print("Scaling feature matrix shape:")
print(X_scaled.shape)

print("\nFirst 5 rows of scaled feature matrix:")
print(X_scaled[:5])


#========================================
# Train & Test Split
#========================================

X_trian, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
)

print("\nTrain and test split:")
print("-----------------------------------")

print("X_train shape:")
print(X_trian.shape)

print("\nX_test shape:")
print(X_test.shape)

print("\ny_train shape:")
print(y_train.shape)
print("\ny_test shape:")
print(y_test.shape)


#===============================
# Baseline Model
#===============================

# Average target value form training set
baseline_prediction = y_train.mean()

# Predict the mean for every test sample
y_pred_baseline = np.full(len(y_train), baseline_prediction)

# Evaluation metrics
baseline_mse = mean_squared_error(y_train, y_pred_baseline)
baseline_r2 = r2_score(y_train, y_pred_baseline)
baseline_rmse = np.sqrt(baseline_mse)

print("\nBaseline Model:")
print("------------------------------")

print(f"Mean prediction: {baseline_prediction:.2f}")

print(f"\nBaseline MSE: {baseline_mse:.2f}")
print(f"Baseline R-squared: {baseline_r2:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")


#=================================================
# Linear Regression Model
#==================================================

# Initializing Model
linear_model = LinearRegression()

# Train model
linear_model.fit(X_trian, y_train)

#Generate prediction
y_pred_linear = linear_model.predict(X_test)

#Evaluation metrics
linear_mse = mean_squared_error(y_test, y_pred_linear)
linear_rmse = np.sqrt(linear_mse)
linear_r2 = r2_score(y_test, y_pred_linear)

print("\n Linear Regression Model:")
print("-----------------------------------")

print(f"Linear Regression MSE: {linear_mse:.2f}")
print(f"Linear Regression R-squared: {linear_r2:.2f}")
print(f"Linear Regression RMSE: {linear_rmse:.2f}")

#=============================================
# Ridge Regression Model
#=============================================

# Initializing Ridge model
ridge_model = Ridge(alpha=1.0)
#Train model
ridge_model.fit(X_trian, y_train)

#Generate predictions
y_pred_ridge = ridge_model.predict(X_test)

# Evaluation metric
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(ridge_mse)

print("\n Ridge Regression Model:")
print("---------------------------------")

print(f"Ridge Regression MSE: {ridge_mse:.2f}")
print(f"Ridge Regression R-squared: {ridge_r2:.2f}")
print(f"Ridge Regression RMSE: {ridge_rmse:.2f}")


#==================================
# KNN Regression Model
#===================================

# Initializing KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Train model
knn_model.fit(X_trian, y_train)

#Generate predictions
y_pred_knn = knn_model.predict(X_test)

#Evaluation metrics
knn_mse = mean_squared_error(y_test, y_pred_knn)
knn_r2 = r2_score(y_test, y_pred_knn)
knn_rmse = np.sqrt(knn_mse)

print("\n KNN Regression Model:")
print("-------------------------------------")

print(f"KNN Regression MSE: {knn_mse:.2f}")
print(f"KNN Regression R-squared: {knn_r2:.2f}")
print(f"KNN Regression RMSE: {knn_rmse:.2f}")


#=======================================
# PCA Analysis
#========================================

pca = PCA(n_components=2)

#Apply PCA
x_pca = pca.fit_transform(X_scaled)

print("\nPCA Analysis")
print("-----------------------------")

print("\nPCA transformed data shape:")
print(pca.explained_variance_ratio_)

print("\nTotal explained variance:")
print(pca.explained_variance_ratio_.sum())


#==================================
# PCA visualization
#==================================

plt.figure(figsize=(10, 10))
plt.scatter(
    x_pca[:, 0],
    x_pca[:, 1],
    alpha=0.7,
)

plt.title("PCA Projection of Student Dataset")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.grid(True)

plt.savefig(PROJECT_ROOT / "results" / "figures" / "pca_scatter.png")

#plt.show()

#====================================
# Actual vs Predicted Plot
#====================================

plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred_linear, alpha=0.5)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")

plt.title("Actual vs Predicted Grades (Linear regression)")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--",
)

plt.grid(True)

plt.savefig(PROJECT_ROOT / "results" / "figures" / "linear_regression_predictions.png")
#plt.show()

#========================================
# Model Performance Comparison
#========================================

comparison_df = pd.DataFrame({
    "Model": [
        "Baseline",
        "Linear Regression",
        "Ridge Regression",
        "KNN Regression"
    ],
    "MSE":[
        baseline_mse,
        linear_mse,
        ridge_mse,
        knn_mse
    ],
    "RMSE":[
        baseline_rmse,
        linear_rmse,
        ridge_rmse,
        knn_rmse
    ],
    "R2 Score":[
        baseline_r2,
        linear_r2,\
        ridge_r2,
        knn_r2
    ]
})

print("\nModel Performance Comparison:")
print(comparison_df)


# Saving the table
comparison_df.to_csv(
    PROJECT_ROOT / "results"/ "tables" / "model_comparison.csv",
    index=False,
)



#===============================
# RMSE Comparison Plot
#===============================
plt.figure(figsize=(10, 10))

sns.barplot(
    x="Model",
    y="MSE",
    data=comparison_df
)

plt.title("Model RMSE Comparison")
plt.xticks(rotation=10)

plt.savefig(PROJECT_ROOT / "results" / "figures" / "rmse_comparison.png")


















