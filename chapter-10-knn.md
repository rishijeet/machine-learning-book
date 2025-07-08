# Chapter 10: K-Nearest Neighbors (KNN)

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

K-Nearest Neighbors (KNN) is one of the simplest and most intuitive machine learning algorithms. It's a supervised learning algorithm that can be used for both classification and regression, but it's most commonly used for classification.

KNN is often called a **"lazy learner"** because it doesn't actually learn a "model" in the traditional sense. Instead of building a function from the training data, it just memorizes the entire training dataset.

When it's time to make a prediction for a new, unseen data point, KNN follows a simple rule:

1.  It calculates the distance from the new point to every single point in the training data.
2.  It finds the 'K' closest points (the "neighbors").
3.  For classification, it takes a majority vote: the new point is assigned the class that is most common among its K neighbors.

> **Analogy: Guessing a House Price by Looking at the Neighbors**
> Imagine you're trying to guess if a house in a new city is expensive or affordable. You don't have a complex formula. You just walk around the house and look at its 5 closest neighbors (K=5). If 4 of them are expensive mansions and 1 is a modest home, you'd probably guess the new house is also expensive. You're classifying it based on the majority class of its nearest neighbors.

The choice of **K** is crucial. A small K can make the model sensitive to noise, while a large K can be computationally expensive and might over-simplify the decision boundary.

**Important Note:** Because KNN is based on distance, **feature scaling is mandatory**. If one feature (like salary) is on a much larger scale than another (like age), it will dominate the distance calculation, and the smaller-scale feature will be ignored.

## Python Code Example

Let's build a KNN classifier. We'll also show a simple way to find a good value for K.

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Create a dummy dataset and split it
X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Feature Scaling (Crucial for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train a KNN model with a specific K
k = 5
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train_scaled, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test_scaled)
print(f"----------- Model Performance with K={k} -----------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 5. Find the best value for K
error_rate = []
k_range = range(1, 40)

for i in k_range:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_scaled, y_train)
    pred_i = knn.predict(X_test_scaled)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10, 6))
plt.plot(k_range, error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

best_k = k_range[np.argmin(error_rate)]
print(f"\nBest K value found: {best_k} with an error rate of {min(error_rate):.4f}")
```

## Summary

- **K-Nearest Neighbors (KNN)** is a simple, non-parametric, "lazy" learning algorithm.
- It classifies new data based on the majority vote of its 'K' nearest neighbors in the training set.
- **Feature scaling is absolutely essential** for KNN to work correctly.
- The choice of **K is a critical hyperparameter** that needs to be tuned.
- KNN can be computationally slow and memory-intensive for large datasets because it needs to store all training data and calculate distances for every new prediction.
