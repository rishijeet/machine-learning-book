# Chapter 3: Data Preprocessing

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Machine learning models are powerful, but they have a critical weakness: they are extremely sensitive to the quality of the data they are fed. Real-world data is almost always messy, incomplete, and inconsistent. Feeding this "dirty" data directly into a model will lead to inaccurate and unreliable results.

**Data preprocessing** is the crucial step of cleaning, transforming, and preparing your raw data to make it suitable for a machine learning model.

> **Analogy: Cooking a Gourmet Meal**
> You can't just throw raw, unwashed vegetables, unseasoned meat, and a cup of flour into a pot and expect a delicious meal. You must first wash the vegetables, chop them, marinate the meat, and measure the ingredients precisely. Data preprocessing is the "kitchen prep" for machine learning. The better your preparation, the better your final dish (the model).

Common preprocessing tasks include:
- **Handling Missing Data:** What do you do with empty cells or unknown values?
- **Encoding Categorical Data:** How do you convert text labels (like "Red", "Green", "Blue") into numbers that a model can understand?
- **Feature Scaling:** How do you ensure that one feature (like salary in dollars) doesn't overpower another (like age in years)?

## Python Code Example

Let's take a small, messy dataset and clean it up step-by-step using `pandas` and `scikit-learn`.

Our raw data represents customer information:

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# 1. Create a messy dummy dataset
data = {
    'Country': ['France', 'Spain', 'Germany', 'Spain', 'Germany', 'France', np.nan, 'France'],
    'Age': [44, 27, 30, 38, 40, 35, 38, np.nan],
    'Salary': [72000, 48000, 54000, 61000, np.nan, 58000, 52000, 79000],
    'Purchased': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

print("----------- Raw Data -----------")
print(df)
print("\nMissing values before processing:")
print(df.isnull().sum())

# 2. Handling Missing Data
# We'll use an imputer to fill in missing values.
# For numbers (Age, Salary), we'll use the mean.
# For text (Country), we'll use the most frequent value.

# Separate features (X) from the target variable (y)
X = df.iloc[:, :-1].values # All rows, all columns except the last
y = df.iloc[:, -1].values  # All rows, only the last column

# Impute missing numerical data (Age, Salary are at index 1 and 2)
num_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = num_imputer.fit_transform(X[:, 1:3])

# Impute missing categorical data (Country is at index 0)
cat_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X[:, 0:1] = cat_imputer.fit_transform(X[:, 0:1])

print("\n----------- Data After Imputation -----------")
# We convert back to a DataFrame for easy viewing
df_imputed = pd.DataFrame(X, columns=['Country', 'Age', 'Salary'])
print(df_imputed)

# 3. Encoding Categorical Data
# Models need numbers, not text. We'll convert the 'Country' column into
# multiple numerical columns using One-Hot Encoding.
# 'France' -> [1, 0, 0], 'Germany' -> [0, 1, 0], 'Spain' -> [0, 0, 1]

# The ColumnTransformer is a powerful tool to apply different transformations to different columns.
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print("\n----------- Data After One-Hot Encoding -----------")
print("Shape of X:", X.shape)
print("Note: The first 3 columns now represent France, Germany, and Spain.")
print(X[:5]) # Show first 5 rows

# 4. Feature Scaling
# The 'Age' and 'Salary' columns are on different scales.
# Scaling brings them to a comparable range, which helps many algorithms perform better.
# We will use StandardScaler which transforms data to have a mean of 0 and standard deviation of 1.

# Note: We don't scale the one-hot encoded columns (the first 3 columns).
scaler = StandardScaler()
X[:, 3:] = scaler.fit_transform(X[:, 3:]) # Scale the Age and Salary columns

print("\n----------- Final Preprocessed Data (Ready for a Model!) -----------")
print(pd.DataFrame(X, columns=['France', 'Germany', 'Spain', 'Age_Scaled', 'Salary_Scaled']))
```

## Summary

- **Data preprocessing is a mandatory step** in any serious machine learning project. Garbage in, garbage out.
- **Handling Missing Data:** You can fill (impute) missing values using strategies like the mean, median, or most frequent value.
- **Encoding Categorical Data:** Convert text labels into a numerical format (like One-Hot Encoding) so that algorithms can process them.
- **Feature Scaling:** Standardize the range of your numerical features so that no single feature can dominate the learning process. This is crucial for algorithms that are sensitive to distance, like SVM and KNN.

With our data now clean and prepared, we can move on to the next creative step: Feature Engineering.
