# Chapter 18: TensorFlow & PyTorch Basics

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

While Scikit-learn's `MLPClassifier` is great for an introduction, serious deep learning requires more specialized, powerful, and flexible tools. The two undisputed kings of the deep learning world are **TensorFlow** (developed by Google) and **PyTorch** (developed by Meta/Facebook).

These frameworks provide two crucial advantages:
1.  **GPU Acceleration:** They can leverage the massive parallel processing power of Graphics Processing Units (GPUs) to train complex models exponentially faster.
2.  **Automatic Differentiation:** They automatically calculate the complex gradients needed for backpropagation, freeing you from the difficult calculus.

> **Analogy: Building a Car**
> - **Scikit-learn:** This is like buying a high-quality, reliable, pre-built car (like a Toyota Camry). It's easy to use and works great for most everyday tasks.
> - **TensorFlow/PyTorch:** These are like being given a high-performance engine, chassis, and all the individual parts to build your own custom Formula 1 race car. It's more complex to assemble, but the resulting performance and control are in a completely different league.

### TensorFlow (and Keras)

**TensorFlow** is a powerful, production-ready framework. For beginners, the best way to use it is through its high-level API, **Keras**. Keras provides a simple, modular way to build and train neural networks, abstracting away much of the underlying complexity.

### PyTorch

**PyTorch** is famous for its simplicity and "Pythonic" feel. It's extremely popular in the research community because it offers a more direct and flexible way to define models and write custom training loops.

## Python Code Example

The best way to understand the difference is to build the exact same simple neural network in both frameworks.

```python
# --- Common Setup ---
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Create a dataset and scale it
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- TensorFlow / Keras Example ---
print("----------- TensorFlow / Keras -----------")
import tensorflow as tf

# 2. Build the model using the simple Keras Sequential API
model_tf = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(20,)), # Hidden layer 1
    tf.keras.layers.Dense(16, activation='relu'),                   # Hidden layer 2
    tf.keras.layers.Dense(1, activation='sigmoid')                  # Output layer
])

# 3. Compile the model (specify optimizer and loss function)
model_tf.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

# 4. Train the model
model_tf.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# 5. Evaluate the model
loss, accuracy = model_tf.evaluate(X_test, y_test, verbose=0)
print(f"TensorFlow Model Accuracy: {accuracy:.4f}")


# --- PyTorch Example ---
print("\n----------- PyTorch -----------")
import torch
import torch.nn as nn
import torch.optim as optim

# Convert data to PyTorch Tensors
X_train_pt = torch.tensor(X_train, dtype=torch.float32)
y_train_pt = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test_pt = torch.tensor(X_test, dtype=torch.float32)
y_test_pt = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

# 2. Build the model by defining a class
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(20, 32)
        self.layer2 = nn.Linear(32, 16)
        self.layer3 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))
        return x

model_pt = SimpleNet()

# 3. Define loss function and optimizer
criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizer = optim.Adam(model_pt.parameters(), lr=0.001)

# 4. The explicit training loop
for epoch in range(10):
    optimizer.zero_grad()      # Clear old gradients
    outputs = model_pt(X_train_pt) # Forward pass
    loss = criterion(outputs, y_train_pt) # Calculate loss
    loss.backward()            # Backpropagation
    optimizer.step()           # Update weights

# 5. Evaluate the model
with torch.no_grad(): # Deactivate autograd for evaluation
    y_pred = model_pt(X_test_pt)
    predicted_classes = (y_pred > 0.5).float()
    accuracy = (predicted_classes == y_test_pt).float().mean()
    print(f"PyTorch Model Accuracy: {accuracy.item():.4f}")
```

## Visual Explanation

<img src="https://github.com/user-attachments/assets/c561c1a4-1db1-4e81-9d89-bfa34ee7afe4" width="600">

*TensorFlow (with Keras) and PyTorch are the two dominant frameworks for modern deep learning.*

## Summary

- **TensorFlow** and **PyTorch** are essential for serious deep learning due to their flexibility and GPU support.
- **Keras** provides a user-friendly, high-level API for TensorFlow, making it easy to quickly build and iterate on standard models.
- **PyTorch** offers a more "Pythonic" and explicit approach, giving you fine-grained control over the model and training loop, which is why it's a favorite in research.
- The core concepts (layers, activation functions, optimizers, loss functions) are the same in both. Learning one makes it much easier to pick up the other.
