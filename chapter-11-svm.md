# Chapter 11: Support Vector Machines (SVM)

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Support Vector Machines (SVMs) are powerful and flexible supervised learning models used for classification, regression, and outlier detection. For classification, the core idea of an SVM is to find the "best" possible line or boundary (called a **hyperplane**) that separates the different classes in your data.

But what does "best" mean? An SVM defines the best hyperplane as the one that has the **maximum margin**—the largest possible distance—between the hyperplane and the nearest data point from either class.

> **Analogy: The Widest Possible Road**
> Imagine you have two neighborhoods (two classes of data points) and you want to build a straight road that separates them. You could draw many possible roads.
> - A bad road might be too close to one neighborhood, leaving little room for error.
> - An SVM finds the road that is as wide as possible, maximizing the "buffer zone" or "margin" on both sides. The houses (data points) that are right on the edge of this wide road are the "support vectors"—they are the critical points that support the entire structure of the road. If you moved one of these houses, the road would have to be redrawn.

### The Kernel Trick: Handling Non-Linear Data

What if the data can't be separated by a straight line? This is where SVMs truly shine. They use a clever mathematical technique called the **kernel trick**.

The kernel trick takes your data, which is in a low-dimensional space (e.g., 2D), and projects it into a much higher-dimensional space. In this higher dimension, the data often becomes linearly separable. The SVM can then easily draw a hyperplane to separate the classes. When this hyperplane is projected back down to the original dimension, it looks like a complex, non-linear curve.

Common kernels include:
- **'linear'**: For data that is already linearly separable.
- **'rbf' (Radial Basis Function)**: A very popular and powerful kernel that can handle complex, non-linear relationships. It's often the default choice.
- **'poly'**: Creates polynomial decision boundaries.

## Python Code Example

Let's see the SVM in action, especially how the RBF kernel can handle data that a linear model cannot.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Create a non-linearly separable dataset
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

# 2. Preprocessing and Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling is important for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train an SVM with a Linear Kernel
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)

# 4. Train an SVM with an RBF Kernel
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

# 5. Compare Performance
print("----------- Model Performance -----------")
print(f"Linear Kernel SVM Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}")
print(f"RBF Kernel SVM Accuracy: {accuracy_score(y_test, y_pred_rbf):.4f}")

# 6. Visualize the decision boundaries (simplified for illustration)
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(title)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_decision_boundary(svm_linear, X_test_scaled, y_test, "Linear Kernel Decision Boundary")
plt.subplot(1, 2, 2)
plot_decision_boundary(svm_rbf, X_test_scaled, y_test, "RBF Kernel Decision Boundary")
plt.show()
```

## Summary

- **Support Vector Machines (SVMs)** are powerful classifiers that work by finding the optimal separating hyperplane with the maximum margin.
- The data points that define this hyperplane are called **support vectors**.
- The **kernel trick** allows SVMs to solve complex, non-linear problems by projecting data into higher dimensions. The **RBF kernel** is a very powerful and common choice.
- **Feature scaling is essential** for SVMs to perform correctly.
- SVMs are effective in high-dimensional spaces and are memory-efficient because they only use a subset of training points (the support vectors) in the decision function.

---

[< Previous: Chapter 10: KNN](./chapter-10-knn.md) | [Next: Chapter 12: Clustering >](./chapter-12-clustering.md)