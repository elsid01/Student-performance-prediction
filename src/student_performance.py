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
