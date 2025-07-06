# Chapter 15: Hyperparameter Tuning

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Think of a machine learning model as a machine with many knobs and dials. The settings of these dials can drastically change how well the machine performs. These settings, which we choose *before* the model starts learning from the data, are called **hyperparameters**.

**Hyperparameter tuning** is the process of finding the best combination of these settings to get the most accurate and robust model possible.

> **Analogy: Tuning a Radio**
> Your data is like the radio waves being broadcast. The model is your radio. The hyperparameters are the knobs for `volume`, `tuning` (frequency), and `bass/treble`. You can't change the radio waves (the data), but you can adjust the knobs (hyperparameters) to find the clearest possible signal (the best model performance). If the tuning is slightly off, you get static (an inaccurate model).

### Parameters vs. Hyperparameters

It's crucial not to confuse these two:
-   **Parameters:** These are values that the model *learns* from the data during training. Examples include the weights in a linear regression or the feature importances in a decision tree. We don't set these manually.
-   **Hyperparameters:** These are values we, the data scientists, set *before* training. Examples include the `K` in K-Nearest Neighbors, the `n_estimators` (number of trees) in a Random Forest, or the `C` (regularization strength) in an SVM.

### Common Tuning Techniques

1.  **Grid Search:** This method exhaustively tries every single possible combination of the hyperparameters you specify. It's thorough but can be extremely slow if you have many hyperparameters or a wide range of values.
2.  **Random Search:** Instead of trying every combination, this method tries a fixed number of random combinations from the hyperparameter space. It's often much faster than Grid Search and can yield surprisingly good results, as it's more likely to find good combinations for hyperparameters that have a larger effect on performance.
3.  **Bayesian Optimization:** A more advanced technique that uses the results from previous trials to inform which combination to try next. It builds a probabilistic model of the objective function and uses it to select the most promising hyperparameters to evaluate.

## Python Code Example

Let's use Grid Search and Random Search to find the best hyperparameters for a Random Forest model.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# 1. Create a dummy dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Define the hyperparameter grid
# This is a dictionary where keys are the hyperparameter names
# and values are the settings to try.
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 3. Set up and run Grid Search
print("----------- Running Grid Search -----------")
# We use a smaller subset of the grid for demonstration to keep it fast.
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={'n_estimators': [50, 100], 'max_depth': [10, 20]},
    cv=3, # 3-fold cross-validation
    n_jobs=-1, # Use all available CPU cores
    verbose=2
)
grid_search.fit(X_train, y_train)

print("\nBest parameters found by Grid Search:", grid_search.best_params_)
print(f"Best score from Grid Search: {grid_search.best_score_:.4f}")

# 4. Set up and run Random Search
print("\n----------- Running Random Search -----------")
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_grid, # Use the full grid here
    n_iter=20, # Number of random combinations to try
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)
random_search.fit(X_train, y_train)

print("\nBest parameters found by Random Search:", random_search.best_params_)
print(f"Best score from Random Search: {random_search.best_score_:.4f}")
```

## Visual Explanation

!A diagram showing a 2D grid for two hyperparameters (e.g., n_estimators and max_depth). Grid Search checks every point on the grid, while Random Search checks a random subset of points.
*Grid Search is exhaustive but slow. Random Search is a more efficient way to explore the hyperparameter space.*

## Summary

- **Hyperparameters** are the settings of a model that we define before training. **Parameters** are what the model learns during training.
- **Hyperparameter tuning** is the process of finding the optimal settings to maximize model performance.
- **Grid Search** is a brute-force method that tries all possible combinations. It's thorough but can be computationally expensive.
- **Random Search** is often more efficient, as it samples random combinations, which can quickly find good-performing models.
- Tuning is an essential step to move a model from a "good enough" baseline to a production-ready, high-performance state.