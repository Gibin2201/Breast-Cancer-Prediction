import pandas as pd

# Load the Breast Cancer dataset
df = pd.read_csv("Dataset/breast_cancer.csv")

# Display first five records
print("\nFIRST FIVE RECORDS")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns)

# Display dataset information
print("\nDATASET INFORMATION")
df.info()