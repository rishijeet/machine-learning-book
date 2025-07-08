# Chapter 12: Clustering (K-Means, DBSCAN, Hierarchical)

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Welcome to the world of **unsupervised learning**! Until now, we've been working with labeled data (supervised learning). Clustering is our first foray into problems where we don't have an answer key. The goal of clustering is to find natural groupings or "clusters" in unlabeled data.

> **Analogy: The Grocery Store Organizer**
> Imagine you're given thousands of grocery items and told to organize them. You have no pre-defined categories. You would naturally start grouping them: fruits with other fruits, cleaning supplies together, dairy products in another section. You are discovering the hidden structure in the data. This is exactly what clustering algorithms do.

We will explore three popular clustering algorithms:
1.  **K-Means:** A simple and fast algorithm that groups data into a pre-specified number of spherical clusters.
2.  **Hierarchical Clustering:** An algorithm that creates a tree of clusters, which is great for visualization.
3.  **DBSCAN:** A density-based algorithm that can find arbitrarily shaped clusters and identify noise.

### 1. K-Means Clustering

This is the most well-known clustering algorithm. It aims to partition the data into 'K' distinct, non-overlapping clusters.

- **How it works:** It iteratively assigns each data point to the nearest cluster center (centroid) and then updates the centroid to be the mean of the assigned points. This process repeats until the centroids stop moving.
- **Main Challenge:** You must tell the algorithm the number of clusters (K) you want to find beforehand. A common technique to find the optimal K is the **Elbow Method**.

### 2. Hierarchical Clustering

This method creates a hierarchy of clusters, which can be visualized as a tree-like diagram called a **dendrogram**.

- **How it works (Agglomerative):** It starts by treating each data point as its own cluster. Then, it repeatedly merges the two closest clusters until only one cluster (containing all data points) remains.
- **Main Advantage:** You don't need to specify the number of clusters upfront. You can look at the dendrogram and decide where to "cut" the tree to form your clusters.

### 3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

DBSCAN is a powerful algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.

- **How it works:** It defines clusters as continuous regions of high density. It can find clusters of any shape and is robust to noise.
- **Main Advantage:** It doesn't require you to specify the number of clusters and can identify outliers automatically.

## Python Code Example

Let's implement K-Means and use the Elbow Method to determine the optimal number of clusters.

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Create a dummy dataset
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.80, random_state=42)

# It's good practice to scale data for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Use the Elbow Method to find the optimal K
wcss = [] # Within-Cluster Sum of Squares
k_range = range(1, 11)
for i in k_range:
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(k_range, wcss, marker='o', linestyle='--')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# From the plot, the "elbow" is clearly at K=4.

# 3. Train the K-Means model with the optimal K
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# 4. Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(X_scaled[y_kmeans == 0, 0], X_scaled[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X_scaled[y_kmeans == 1, 0], X_scaled[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X_scaled[y_kmeans == 2, 0], X_scaled[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X_scaled[y_kmeans == 3, 0], X_scaled[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', marker='*', label='Centroids')

plt.title('Clusters of customers')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.legend()
plt.grid(True)
plt.show()
```

## Summary

- **Clustering** is an unsupervised learning task for finding hidden groups in unlabeled data.
- **K-Means** is a fast and simple algorithm but requires you to specify the number of clusters (K) and assumes clusters are spherical.
- **Hierarchical Clustering** builds a tree of clusters (a dendrogram) and doesn't require a pre-specified K.
- **DBSCAN** is a density-based method that can find arbitrarily shaped clusters and identify noise, without needing K to be specified.