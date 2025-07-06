# Chapter 7: Logistic Regression

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Despite its name, Logistic Regression is not a regression algorithm. It is a supervised learning algorithm used for **classification** tasks. It's one of the most popular and widely used algorithms for solving binary classification problems, where the output is one of two classes (e.g., Yes/No, True/False, 1/0).

So, how does it work? It's actually very similar to Linear Regression, but with an extra step. It calculates a weighted sum of the input features (like Linear Regression) but then runs that result through a special function called the **Sigmoid function**.

The Sigmoid function is an S-shaped curve that squashes any number it's given into a value between 0 and 1. This is incredibly useful because we can interpret this output as a **probability**.

> **Analogy: The Pass/Fail Predictor**
> Imagine you want to predict if a student will pass or fail an exam based on hours studied.
> - **Linear Regression** would predict a score, like 85 or 45.
> - **Logistic Regression** would predict the *probability* of passing, like 0.95 (95% chance of passing) or 0.30 (30% chance of passing).
>
> We can then set a threshold (usually 0.5). If the predicted probability is greater than 0.5, we classify the outcome as "Pass" (1). If it's less than 0.5, we classify it as "Fail" (0).

### The Sigmoid Function

This is the magic ingredient. It ensures the model's output is always between 0 and 1, making it perfect for probability estimation.

![A plot of the Sigmoid function, an S-shaped curve that goes from 0 to 1.](./images/sigmoid_function.png)

## Python Code Example

Let's build a model to predict whether a person will purchase a product based on their age and estimated salary.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a dummy dataset
from sklearn.datasets import make_classification
X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    flip_y=0.1,
    random_state=42
)

# 2. Preprocessing and Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature Scaling is important for Logistic Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 4. Make predictions
y_pred = model.predict(X_test)

# You can also get the probabilities
y_pred_proba = model.predict_proba(X_test)

print("----------- Model Evaluation -----------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n----------- Example Predictions -----------")
print("First 5 predictions (Class):", y_pred[:5])
print("First 5 probabilities (for Class 0 and Class 1):")
print(y_pred_proba[:5].round(2))
```

## Visual Explanation

A key concept in classification is the **decision boundary**. It's the line or surface that separates the different classes. For Logistic Regression, this boundary is linear.

!A scatter plot showing two classes of data points (e.g., red and blue) separated by a straight line, which is the decision boundary.
*Logistic Regression learns a linear decision boundary to separate the classes.*

## Summary

- **Logistic Regression** is a go-to algorithm for **binary classification**.
- Despite its name, it predicts probabilities, not continuous values.
- It uses the **Sigmoid function** to map any real-valued number into a probability between 0 and 1.
- The line that separates the classes is called the **decision boundary**, and for logistic regression, it is linear.
- It's a simple, fast, and highly interpretable algorithm.