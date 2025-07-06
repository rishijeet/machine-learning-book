# Chapter 16: Model Deployment Basics

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

Building a highly accurate model is a great achievement, but it's only half the battle. A model is useless if it just sits on your computer. **Model deployment** is the process of integrating your trained model into a real-world application so that other people and systems can use it to make predictions.

> **Analogy: The Master Baker's Recipe**
> You (the data scientist) have spent weeks perfecting a secret recipe for the world's best cake (the model).
> - **Training:** Perfecting the recipe in your kitchen.
> - **Deployment:** Opening a bakery (the application) where customers can come and buy your cake.
>
> To do this, you need two things:
> 1.  A way to save your recipe so you don't forget it (`pickle`).
> 2.  A storefront with a counter where customers can place orders (a `Flask` API).

In this chapter, we'll cover the simplest way to deploy a model: saving it with Python's `pickle` library and creating a basic web API with `Flask`.

### Step 1: Saving and Loading Your Model

Once you've trained a model, it exists only in your computer's memory. We need to save its learned parameters to a file. `pickle` is a Python library that serializes a Python object into a byte stream, which we can save to a file and load back later.

### Step 2: Creating a Web API with Flask

A web API (Application Programming Interface) is a way for different computer programs to talk to each other. We'll use **Flask**, a lightweight Python web framework, to create a simple server that:
1.  Loads our saved model file.
2.  Listens for incoming requests with new data.
3.  Uses the model to make a prediction on that data.
4.  Sends the prediction back as a response.

## Python Code Example

This process involves two separate scripts.

#### Script 1: `train_and_save_model.py`

First, we train our model and save it to a file.

```python
# train_and_save_model.py

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Create and train a model
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Model trained successfully.")

# 2. Save the trained model to a file
filename = 'random_forest_model.pkl'
with open(filename, 'wb') as file: # 'wb' means 'write bytes'
    pickle.dump(model, file)

print(f"Model saved to {filename}")
```

#### Script 2: `app.py` (The Flask API)

Now, we create the web server to serve our model.

```python
# app.py

from flask import Flask, request, jsonify
import pickle
import numpy as np

# 1. Initialize the Flask app
app = Flask(__name__)

# 2. Load the saved model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# 3. Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Convert the data into a numpy array for the model
    # Assumes the client sends a list of feature values, e.g., [0.1, 0.2, ...]
    features = np.array(data['features']).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction[0])})

# 4. Run the app
if __name__ == '__main__':
    # To run: `python app.py` in your terminal
    app.run(debug=True, port=5000)
```

### How to Test Your API

1.  Run `python train_and_save_model.py` to create the `random_forest_model.pkl` file.
2.  Run `python app.py` to start the Flask server.
3.  Open a new terminal and use a tool like `curl` to send a POST request with some feature data:

```bash
# This sends a JSON payload with 10 features to our running API
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [0.1, -0.5, 1.2, -0.8, 0.3, 0.9, -1.1, 0.4, -0.2, 1.5]}'
```

You should get a response like: `{"prediction":1}`

## Visual Explanation

!A flowchart showing the deployment process: User sends JSON data via an HTTP POST request to the Flask API. The API receives the request, loads the pickled model from disk, uses the model to make a prediction, and returns the prediction as a JSON response.
*A simple deployment pipeline for serving a machine learning model.*

## Summary

- **Model Deployment** is the process of making your trained model available to users and other applications.
- A common and simple method is to **serialize** (save) the model using `pickle`.
- A lightweight web framework like **Flask** can be used to create an **API endpoint** that loads the pickled model and serves predictions over the internet.
- This is a basic approach. The field of **MLOps** (Machine Learning Operations) deals with more robust, scalable, and automated ways to deploy and manage models in production.