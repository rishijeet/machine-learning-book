# Chapter 13: Dimensionality Reduction (PCA, t-SNE)

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

In machine learning, we often deal with datasets that have a huge number of features (or dimensions). While more data seems better, too many features can lead to the **"Curse of Dimensionality"**—a phenomenon where models perform worse because the data becomes too sparse and difficult to work with.

**Dimensionality reduction** is the process of reducing the number of features in a dataset while trying to retain as much of the important information as possible.

> **Analogy: Summarizing a Long Book**
> Imagine a 1000-page book (a high-dimensional dataset). Reading the whole thing is time-consuming and you might get lost in the details. Instead, you could read a 10-page summary (the reduced dataset). This summary loses some detail, but it captures the main plot points and characters (the most important information), making the story much easier to understand.

We'll look at two popular techniques:
1.  **Principal Component Analysis (PCA):** A linear technique used for data compression and simplification.
2.  **t-SNE (t-Distributed Stochastic Neighbor Embedding):** A non-linear technique used primarily for data visualization.

### 1. Principal Component Analysis (PCA)

PCA is the most common dimensionality reduction algorithm. It works by identifying the "principal components" of the data. These are new, artificial features that are combinations of the old features. The first principal component is the direction in the data that accounts for the most variance. The second principal component accounts for the second-most variance, and so on.

By keeping only the first few principal components, we can reduce the number of features while preserving the bulk of the data's variance (i.e., its information).

### 2. t-SNE

t-SNE is a more modern and powerful technique, but it's almost exclusively used for **visualization**, typically reducing data down to 2 or 3 dimensions. It's a non-linear method that tries to arrange the data points in a low-dimensional space such that similar points are modeled closely together and dissimilar points are modeled far apart. It's fantastic for exploring the underlying structure of your data.

## Python Code Example

Let's take a high-dimensional dataset and see how PCA and t-SNE can help us understand it.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. Load a high-dimensional dataset
# The digits dataset has 64 features (8x8 pixel images of handwritten digits).
digits = load_digits()
X = digits.data
y = digits.target

print(f"Original data shape: {X.shape}")

# 2. Scale the data (important for PCA)
X_scaled = StandardScaler().fit_transform(X)

# 3. Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"Data shape after PCA: {X_pca.shape}")
print(f"Explained variance by 2 components: {sum(pca.explained_variance_ratio_):.2f}")

# 4. Apply t-SNE to reduce to 2 components
# t-SNE can be slow on large datasets.
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled) # Use scaled data for consistency

print(f"Data shape after t-SNE: {X_tsne.shape}")

# 5. Visualize the results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
plt.title('PCA of Digits Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
plt.title('t-SNE of Digits Dataset')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.colorbar()

plt.show()
```

## Summary

- **Dimensionality Reduction** is used to reduce the number of features, which can improve model performance and speed up training.
- **PCA** is a linear method best used for data compression and removing redundant features. It creates new features (principal components) that are ordered by the amount of variance they explain.
- **t-SNE** is a non-linear method best used for data visualization. It excels at revealing the underlying cluster structure in high-dimensional data.
- Always scale your data before applying PCA.

---

[< Previous: Chapter 12: Clustering](./chapter-12-clustering.md) | [Next: Chapter 14: Model Evaluation >](./chapter-14-model-evaluation.md)