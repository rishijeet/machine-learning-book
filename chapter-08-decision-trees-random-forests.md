# Chapter 8: Decision Trees & Random Forests

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Decision Trees are one of the most intuitive models in machine learning. They work by making a series of "if/else" questions about the data, ultimately leading to a decision. Their structure is just like a flowchart, which makes them very easy to understand and interpret.

> **Analogy: The Loan Application Flowchart**
> Imagine a bank deciding whether to approve a loan. They might use a process like this:
> 1.  Is the applicant's credit score > 700?
>     - **Yes:** Is their annual income > $50,000?
>       - **Yes:** Approve Loan.
>       - **No:** Is their debt-to-income ratio < 40%?
>         - **Yes:** Approve Loan.
>         - **No:** Deny Loan.
>     - **No:** Deny Loan.
>
> This flowchart is a decision tree. The model learns the best questions to ask (the splits) and in what order to best separate the data into the final outcomes (the leaves).

### The Problem with a Single Tree: Overfitting

A single decision tree is powerful, but it has a major weakness: it can easily **overfit** the training data. It can keep growing and creating new branches to perfectly classify every single data point it has seen, creating a very complex tree that doesn't generalize well to new, unseen data. It's like the student who memorizes the practice exam perfectly but can't answer new questions.

### The Solution: Random Forests

This brings us to the **Random Forest**. If one tree is good, then a whole forest of them must be better!

A Random Forest is an **ensemble** model, which means it's a model made up of many smaller models (in this case, many decision trees). It builds hundreds or thousands of slightly different decision trees and then makes a final prediction by taking a majority vote from all of them.

> **Analogy: A Committee of Experts**
> Instead of asking one single expert (a single tree) for their opinion, you ask a large committee of diverse experts. Each expert has slightly different knowledge and biases. While any single expert might be wrong, the collective "wisdom of the crowd" is usually very accurate and robust.

The "random" part comes from two key ideas:
1.  **Random Data Samples:** Each tree in the forest is trained on a different random subset of the original data (this is called bagging or bootstrap aggregating).
2.  **Random Feature Subsets:** At each split in a tree, only a random subset of the total features is considered. This forces the trees to be different from each other and prevents one very strong feature from dominating every tree.

This randomness makes the overall model much less prone to overfitting and generally much more accurate than a single decision tree.

## Python Code Example

Let's build both a single Decision Tree and a Random Forest to see the difference. We'll also visualize the single tree to see how interpretable it is.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Create a dummy dataset
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train a Single Decision Tree
single_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
single_tree.fit(X_train, y_train)
y_pred_tree = single_tree.predict(X_test)

# 3. Train a Random Forest
random_forest = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_forest = random_forest.predict(X_test)

# 4. Compare Accuracy
print("----------- Model Performance -----------")
print(f"Single Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.4f}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_forest):.4f}")

# 5. Visualize the Single Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(
    single_tree,
    filled=True,
    rounded=True,
    class_names=['Class 0', 'Class 1'],
    feature_names=[f'Feature_{i}' for i in range(X.shape[1])]
)
plt.title("Visualization of a Single Decision Tree")
plt.show()
```

## Visual Explanation

![8497b926-592b-40c3-a0a9-e74484c7d95b](https://github.com/user-attachments/assets/2bd3f27c-d2b5-4607-acbb-6974a0effdbb)

*A Random Forest combines the outputs of many individual trees to make a more accurate and stable prediction.*

## Summary

- **Decision Trees** are intuitive, flowchart-like models that are easy to visualize and interpret.
- Their main weakness is a tendency to **overfit** the training data.
- **Random Forests** are an ensemble of many decision trees. They overcome overfitting by using randomness and the "wisdom of the crowd" (majority voting).
- Random Forests are one of the most powerful and widely used "off-the-shelf" classification and regression algorithms due to their high accuracy and robustness.
