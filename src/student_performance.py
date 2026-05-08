import pandas as pd
from pathlib import Path


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