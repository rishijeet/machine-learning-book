# Chapter 4: Feature Engineering

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

If data preprocessing is the "kitchen prep," then **feature engineering** is the art of being a creative chef. It's the process of using your domain knowledge to transform raw data into features that better represent the underlying problem to the machine learning model.

A "feature" is just an input variable—a column in your dataset. Feature engineering is about creating *new* columns from existing ones to help your model learn more effectively. A well-engineered feature can be the difference between a mediocre model and a highly accurate one.

> **Analogy: Describing a House to a Buyer**
> You have a dataset with `house_length` and `house_width`. These are your raw features. A model could learn from them, but it might be slow.
> What if you create a new feature called `house_area` by multiplying `length * width`? This single, engineered feature is much more informative and directly related to the house's value.
> What if you also have `year_built`? You could create a feature called `house_age` (current year - year_built), which is often more predictive of maintenance issues than the build year itself.

Common feature engineering techniques include:
- **Creating Interaction Features:** Combining two or more features (e.g., `area = length * width`).
- **Polynomial Features:** Creating squared or cubed versions of features to capture non-linear relationships.
- **Extracting Information from Dates:** Breaking down a date column into `year`, `month`, `day_of_week`, or `is_weekend`.

## Python Code Example

Let's engineer some new features from a simple dataset of online purchases.

```python
import pandas as pd

# 1. Create a dummy dataset
data = {
    'user_id': [1, 2, 3, 4, 5],
    'signup_date': ['2022-01-15', '2022-02-20', '2022-03-01', '2022-04-10', '2022-05-25'],
    'last_purchase_date': ['2023-01-20', '2022-08-15', '2023-03-05', '2022-04-11', '2023-06-01'],
    'total_purchases': [15, 5, 2, 1, 25],
    'total_spent': [300, 150, 50, 10, 1200]
}
df = pd.DataFrame(data)

# Ensure date columns are in datetime format
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])

print("----------- Original Data -----------")
print(df)

# 2. Engineer New Features

# Feature 1: 'days_as_customer' - How long has the user been registered?
df['days_as_customer'] = (df['last_purchase_date'] - df['signup_date']).dt.days

# Feature 2: 'avg_purchase_value' - What is the average amount spent per purchase?
# We add a small number (1e-6) to avoid division by zero for users with 0 purchases.
df['avg_purchase_value'] = df['total_spent'] / (df['total_purchases'] + 1e-6)

# Feature 3: 'signup_month' - Is there a seasonal pattern to signups?
df['signup_month'] = df['signup_date'].dt.month

print("\n----------- Data with Engineered Features -----------")
print(df[['user_id', 'days_as_customer', 'avg_purchase_value', 'signup_month']])
```

## Summary

- **Feature engineering is a highly creative process** that blends domain knowledge with data manipulation.
- Its goal is to create features that make the underlying patterns in the data more obvious to the machine learning algorithm.
- Simple transformations, like combining columns or extracting parts of a date, can significantly boost model performance.
- Always think about what information would be most useful for making a prediction and see if you can create it from the data you already have.

---

[< Previous: Chapter 3: Data Preprocessing](./chapter-03-data-preprocessing.md) | [Next: Chapter 5: Train Test Split Cross Validation >](./chapter-05-train-test-split-cross-validation.md)