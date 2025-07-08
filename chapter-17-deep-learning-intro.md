# Chapter 17: Intro to Deep Learning

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Welcome to the most talked-about subfield of machine learning: **Deep Learning**. Deep learning is the engine behind many of the most impressive AI achievements, from self-driving cars to realistic language translation and image generation.

At its core, deep learning is a type of machine learning based on **Artificial Neural Networks (ANNs)**. The "deep" in deep learning simply refers to the use of many layers in the network.

> **Analogy: The Human Brain**
> ANNs are inspired by the structure of the human brain. Our brains are made of billions of interconnected neurons that process information. An ANN is a simplified model of this.
> - **Neuron:** A single processing unit that receives signals and passes them on.
> - **Network:** A large collection of these neurons organized into layers.
> - **Learning:** The process of strengthening or weakening the connections between neurons based on experience (data).

### The Building Block: The Perceptron (Neuron)

The simplest unit in a neural network is a **perceptron**, or neuron. It's a simple computational unit that:
1.  Takes one or more inputs (your features).
2.  Multiplies each input by a **weight** (the strength of the connection).
3.  Sums up all the weighted inputs.
4.  Passes the result through an **activation function**, which decides whether the neuron should "fire" and what signal to pass on.

### Artificial Neural Networks (ANNs)

A single neuron isn't very powerful. The magic happens when we stack them together in layers:

-   **Input Layer:** Receives the raw data. There is one neuron for each feature in your dataset.
-   **Hidden Layers:** These are the layers between the input and output. This is where the real "learning" happens. The network learns to recognize increasingly complex patterns in these layers.
-   **Output Layer:** Produces the final prediction. For classification, it might have one neuron for each class.

A "shallow" neural network might have only one hidden layer. A **"deep"** neural network has multiple hidden layers, allowing it to learn very complex, hierarchical patterns.

### How Do They Learn?

The network learns by adjusting the weights of the connections between neurons. It does this through a process called **backpropagation** and an optimization algorithm like **gradient descent**. In simple terms, the network makes a guess, checks how wrong its guess was (calculates the "error"), and then works backward through the network, slightly adjusting each weight to reduce the error. This process is repeated thousands or millions of time until the network's predictions are as accurate as possible.

## Python Code Example

While full-fledged deep learning is usually done with specialized libraries like TensorFlow or PyTorch (which we'll cover in the next chapter), `scikit-learn` provides a simple implementation called `MLPClassifier` (Multi-layer Perceptron) that is perfect for an introduction.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# 1. Create a dataset
X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Scale the features
# Neural networks are very sensitive to feature scaling.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Initialize and train the MLP Classifier
# hidden_layer_sizes=(100, 50) means two hidden layers:
# - The first hidden layer has 100 neurons.
# - The second hidden layer has 50 neurons.
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500, # Number of training epochs
    random_state=42,
    verbose=False # Set to True to see training progress
)

print("Training a Multi-layer Perceptron...")
model.fit(X_train_scaled, y_train)
print("Training complete.")

# 4. Evaluate the model
y_pred = model.predict(X_test_scaled)
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

## Summary

- **Deep Learning** uses Artificial Neural Networks (ANNs) with many layers (hence "deep") to learn complex patterns.
- The basic unit is a **neuron** (or perceptron), which processes weighted inputs and uses an activation function to produce an output.
- Neurons are organized into an **input layer**, one or more **hidden layers**, and an **output layer**.
- The network "learns" by adjusting the **weights** of the connections between neurons through a process called **backpropagation**.
- Deep learning excels at tasks involving unstructured data like images, sound, and text.