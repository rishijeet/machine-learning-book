# Chapter 5: Train/Test Split and Cross Validation

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

So far, we've learned how to prepare data and create new features. But how do we train a model and, more importantly, how do we know if it has actually *learned* anything useful?

A common mistake is to train a model on all your data and then test it on that same data. This leads to a critical problem called **overfitting**.

> **Analogy: The Student and the Exam**
> Imagine a student who is given the exact questions and answers for an upcoming exam. The student can memorize them perfectly and get 100% on the test. But have they actually learned the subject? No. If you give them a *different* exam with new questions on the same topics, they will likely fail.
>
> - **Training Data:** The practice questions and answers.
> - **The Model:** The student.
> - **Overfitting:** The student memorizes the answers instead of learning the concepts.
> - **The Goal:** We want our student (model) to perform well on a *final, unseen exam* (new, real-world data).

To solve this, we split our data into two parts:
1.  **Training Set:** The majority of the data, used to teach the model.
2.  **Testing Set:** A smaller, held-back portion of the data that the model never sees during training. We use this to evaluate its true performance.

### The Problem with a Single Split

A single train/test split is good, but what if you get "lucky" or "unlucky" with your split? Maybe the test set accidentally contains all the easy examples, making your model look better than it is.

To get a more reliable estimate of performance, we use **Cross-Validation**. The most common method is **K-Fold Cross-Validation**.

1.  Split the data into K "folds" (e.g., 5 or 10).
2.  In the first round, hold out Fold 1 as the test set and train the model on the other K-1 folds.
3.  In the second round, hold out Fold 2 as the test set and train on the rest.
4.  Repeat this K times, with each fold getting a turn to be the test set.
5.  The final performance is the average of the scores from all K rounds.

This gives us a much more robust and trustworthy measure of our model's ability to generalize to new data.

## Python Code Example

Let's see how to implement both a simple train/test split and K-fold cross-validation in Python.

```python
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 1. Create a dummy dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# 2. Simple Train/Test Split
print("----------- Simple Train/Test Split -----------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Train a model on the training set
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate on the test set
test_score = model.score(X_test, y_test)
print(f"Score on single test set: {test_score:.4f}\n")

# 3. K-Fold Cross-Validation
print("----------- K-Fold Cross-Validation -----------")
# We use the *entire* dataset (X, y) for cross-validation, as it handles its own splits.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_score does the entire loop for us: splitting, training, and scoring.
cv_scores = cross_val_score(model, X, y, cv=kf)

print(f"Scores for each of the 5 folds: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.4f}")
print(f"Standard Deviation of CV Scores: {np.std(cv_scores):.4f}")
```

## Summary

- **Never test your model on the same data it was trained on.** This leads to an overly optimistic and misleading performance score.
- **Overfitting** occurs when a model memorizes the training data instead of learning the general patterns.
- A **train/test split** is the minimum requirement for proper model evaluation.
- **K-Fold Cross-Validation** is a more robust technique that provides a more reliable estimate of how your model will perform on unseen data by training and testing on multiple different splits of the data.

---

[< Previous: Chapter 4: Feature Engineering](./chapter-04-feature-engineering.md) | [Next: Chapter 6: Linear Regression >](./chapter-06-linear-regression.md)