# Chapter 6: Linear Regression

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Linear Regression is often the first algorithm people learn in machine learning. It's a fundamental building block and a fantastic way to understand the relationship between variables. It's a supervised learning algorithm used for **regression** tasks, meaning its goal is to predict a continuous numerical value.

Imagine you're trying to predict a house's price. You have data on different houses: their size (in square feet) and their price. If you plot this data, you might see a trend: as the size increases, the price generally increases too.

Linear Regression is the technique of drawing a straight line that **best fits** this data. This line represents the mathematical relationship between the house size (the **feature** or **independent variable**) and its price (the **target** or **dependent variable**).

Once you have this line, you can use it to predict the price of a new house just by knowing its size!

> **Analogy: The Plant Growth Line**
> You measure a plant's height every day for a few weeks. If you plot the days on the x-axis and the height on the y-axis, you'll see a series of dots. You can draw a straight line through these points to show the general growth trend. Linear regression is the mathematical way to find the *perfect* line that minimizes the total distance from the line to all the actual data points.

### The Math (Simplified)

The equation for a straight line is `y = mx + c`. In machine learning, we write it a bit differently, but the concept is identical:

`y = w1*x1 + w0`

- `y` is the prediction (e.g., house price).
- `x1` is the input feature (e.g., house size).
- `w1` is the **weight** or **coefficient** (the slope of the line, `m`). It tells us how much `y` changes for a one-unit increase in `x1`.
- `w0` is the **bias** or **intercept** (where the line crosses the y-axis, `c`). It's the baseline value of `y` when `x1` is zero.

The goal of "training" a linear regression model is to find the best possible values for `w1` and `w0` that make the line fit the data as closely as possible.

## Python Code Example

Let's build a simple linear regression model to predict a student's exam score based on the number of hours they studied.

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a dummy dataset
data = {
    'Hours_Studied': [1, 2, 2.5, 3, 4, 4.5, 5, 6, 7, 8],
    'Exam_Score': [55, 60, 62, 68, 75, 78, 82, 88, 90, 97]
}
df = pd.DataFrame(data)

# 2. Define features (X) and target (y)
# In scikit-learn, features (X) are expected to be a 2D array-like structure.
X = df[['Hours_Studied']]
y = df['Exam_Score']

# 3. Initialize and train the model
# In a real project, you would use your training set here (from a train/test split).
model = LinearRegression()
model.fit(X, y)

print(f"Model Intercept (w0): {model.intercept_:.2f}")
print(f"Model Coefficient for Hours_Studied (w1): {model.coef_[0]:.2f}")

# 4. Make a prediction
hours_to_predict = [[5.5]] # Predict score for 5.5 hours of study
predicted_score = model.predict(hours_to_predict)
print(f"\nPredicted score for {hours_to_predict[0][0]} hours: {predicted_score[0]:.2f}")

# 5. Visualize the results
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Hours_Studied', y='Exam_Score', data=df, s=100, label='Actual Data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.title('Hours Studied vs. Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.legend()
plt.grid(True)
plt.show()
```

## Summary

- **Linear Regression** is a supervised learning algorithm used to predict a continuous numerical output.
- It works by finding the best-fitting straight line (or hyperplane in higher dimensions) that describes the relationship between features and the target.
- The model "learns" the optimal **intercept (bias)** and **coefficients (weights)** for the features during training.
- It's a simple, interpretable algorithm that serves as a great foundation for understanding more complex models.

---

[< Previous: Chapter 5: Train Test Split Cross Validation](./chapter-05-train-test-split-cross-validation.md) | [Next: Chapter 7: Logistic Regression >](./chapter-07-logistic-regression.md)