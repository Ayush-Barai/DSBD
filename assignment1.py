# Install required libraries (only in Jupyter)
%pip install pandas numpy matplotlib seaborn scikit-learn

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 2. Load the Iris dataset using seaborn
iris = sns.load_dataset('iris')
print("First 5 rows of the dataset:")
print(iris.head())

# 3. Display initial statistics
print("\nDataset shape:")
print(iris.shape)

print("\nDataset information:")
print(iris.info())

print("\nDescriptive statistics:")
print(iris.describe())

print("\nClass distribution:")
print(iris['species'].value_counts())

# 4. Check for missing values and duplicates
print("\nMissing values in each column:")
print(iris.isnull().sum())

print("\nNumber of duplicate rows:")
print(iris.duplicated().sum())

# 5. Identify outliers using IQR method
numeric_columns = iris.select_dtypes(include=[np.number]).columns

Q1 = iris[numeric_columns].quantile(0.25)
Q3 = iris[numeric_columns].quantile(0.75)
IQR = Q3 - Q1

outliers = ((iris[numeric_columns] < (Q1 - 1.5 * IQR)) | 
            (iris[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)

print("\nNumber of rows with outliers:", outliers.sum())
print("\nOutlier rows:")
print(iris[outliers])

# 6. Apply Min-Max Scaling
scaler = MinMaxScaler()
iris_scaled = iris.copy()
iris_scaled[numeric_columns] = scaler.fit_transform(iris_scaled[numeric_columns])

print("\nAfter Min-Max scaling - descriptive statistics:")
print(iris_scaled[numeric_columns].describe())

# 7. Apply Label Encoding
label_encoder = LabelEncoder()
iris_encoded = iris.copy()
iris_encoded['species_encoded'] = label_encoder.fit_transform(iris_encoded['species'])

print("\nAfter Label Encoding:")
print(iris_encoded[['species', 'species_encoded']].head(10))
