# Chapter 14: Model Evaluation Metrics

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

So you've built a model. Congratulations! But how do you know if it's any good? Is it ready to be shown to the world, or does it need more work? This is where model evaluation comes in. We need objective numbers—metrics—to measure our model's performance.

### Why Not Just Use Accuracy?

Accuracy (the percentage of correct predictions) seems simple and intuitive, but it can be very misleading, especially with **imbalanced datasets**.

> **Analogy: The Rare Disease Detector**
> Imagine you build a model to detect a rare disease that affects only 1 in 1000 people. If your model is lazy and just predicts "no disease" for everyone, it will be **99.9% accurate**! But it's completely useless because it never finds the people who actually need help.

This is why we need more nuanced metrics for classification.

### The Confusion Matrix

To understand these metrics, we first need to understand the **Confusion Matrix**. It's a table that shows where our model got things right and where it got them wrong.

-   **True Positives (TP):** You predicted YES, and the actual answer was YES. (Correctly identified a sick patient).
-   **True Negatives (TN):** You predicted NO, and the actual answer was NO. (Correctly identified a healthy patient).
-   **False Positives (FP):** You predicted YES, but the actual answer was NO. (A "false alarm." Told a healthy patient they were sick).
-   **False Negatives (FN):** You predicted NO, but the actual answer was YES. (A "miss." Told a sick patient they were healthy - often the worst kind of error!).

### Key Metrics for Classification

1.  **Precision:** Of all the times you predicted YES, how often were you correct?
    -   **Formula:** `TP / (TP + FP)`
    -   **When to use it:** When the cost of a False Positive is high. For example, in spam detection, you don't want to accidentally mark an important email as spam (a false positive). High precision is key.

2.  **Recall (or Sensitivity):** Of all the actual YES cases, how many did you find?
    -   **Formula:** `TP / (TP + FN)`
    -   **When to use it:** When the cost of a False Negative is high. For example, in medical diagnosis, you absolutely want to find every sick patient (avoid false negatives). High recall is critical.

3.  **F1-Score:** A single score that balances Precision and Recall. It's the "harmonic mean" of the two.
    -   **Formula:** `2 * (Precision * Recall) / (Precision + Recall)`
    -   **When to use it:** When you need a good balance between Precision and Recall and don't have a specific reason to prioritize one over the other.

### Key Metrics for Regression

For regression tasks where we predict a number, we use different metrics:

1.  **Mean Absolute Error (MAE):** The average of the absolute differences between the predicted and actual values. It's easy to understand because it's in the same units as the target variable.
2.  **Mean Squared Error (MSE):** The average of the squared differences. This metric penalizes larger errors more heavily than smaller ones.
3.  **Root Mean Squared Error (RMSE):** The square root of the MSE. This brings the metric back to the original units of the target variable, making it more interpretable than MSE.

## Python Code Example

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)
import numpy as np

# --- Classification Metrics Example ---
print("----------- Classification Metrics -----------")
y_true_class = [0, 1, 1, 0, 1, 0, 0, 1, 1, 1] # 0 = No, 1 = Yes
y_pred_class = [0, 1, 0, 0, 1, 1, 0, 1, 1, 0] # Our model's guesses

print(f"Accuracy: {accuracy_score(y_true_class, y_pred_class):.2f}")
print(f"Precision: {precision_score(y_true_class, y_pred_class):.2f}") # 4 / (4+1) = 0.80
print(f"Recall: {recall_score(y_true_class, y_pred_class):.2f}")     # 4 / (4+2) = 0.67
print(f"F1-Score: {f1_score(y_true_class, y_pred_class):.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_class, y_pred_class))
# [[TN FP]
#  [FN TP]]
# [[3 1]
#  [2 4]]

# --- Regression Metrics Example ---
print("\n----------- Regression Metrics -----------")
y_true_reg = [100, 150, 200, 250, 300]
y_pred_reg = [110, 145, 215, 240, 290]

mae = mean_absolute_error(y_true_reg, y_pred_reg)
mse = mean_squared_error(y_true_reg, y_pred_reg)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```

## Summary

- **Accuracy is not always enough**, especially for imbalanced datasets where one class is much more frequent than another.
- For **classification**, use **Precision** when False Positives are costly, **Recall** when False Negatives are costly, and the **F1-Score** for a balanced view.
- For **regression**, **MAE** gives you the average error in the original units, while **MSE** and **RMSE** penalize large errors more severely. Choosing the right metric depends entirely on your project's business goal.