# Chapter 9: Naive Bayes

_This chapter is part of the book **Mastering Machine Learning — From Scratch to Advanced**_

## Overview

The Naive Bayes classifier is a simple but surprisingly powerful probabilistic algorithm. It's based on **Bayes' Theorem** and is particularly popular for text classification tasks like spam filtering and sentiment analysis.

The "naive" part of the name comes from a core assumption the model makes: that every feature is **independent** of every other feature. This means the model assumes that the presence of one word in an email has no bearing on the presence of another word, which isn't really true in language. However, even with this "naive" assumption, the algorithm performs remarkably well in practice.

> **Analogy: The Naive Spam Detective**
> Imagine a detective trying to identify a spam email. They have a list of suspicious words: "Viagra," "free," "money," "winner."
> - A normal detective would understand that the words "free" and "money" often appear together.
> - A **naive** detective treats each word as a separate piece of evidence. They calculate the probability of an email being spam given the word "free," and separately, the probability given the word "money." They then combine these probabilities to make a final judgment.
>
> Even though the assumption of independence is wrong (these words are related!), the combined weight of evidence is often enough for the naive detective to correctly classify the email as spam.

### How It Works

At its heart, Naive Bayes calculates the probability of a certain class (e.g., "spam") given a set of features (e.g., the words in the email). It uses Bayes' Theorem to do this:

`P(Class | Features) = (P(Features | Class) * P(Class)) / P(Features)`

In simple terms: The probability of it being *spam* given these *words* depends on how often these *words* appear in previous *spam* emails.

## Python Code Example

Let's build a classic spam filter using Naive Bayes. We'll teach it to distinguish between spam and non-spam ("ham") messages.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Create a dummy text dataset
data = {
    'message': [
        'Free money! Click here to win.',
        'You are a winner, claim your prize now!',
        'Exclusive offer just for you, free gift inside.',
        'Hi Mom, how are you doing? See you on Sunday.',
        'Can we reschedule our meeting for tomorrow?',
        'Don\'t forget to pick up the groceries on your way home.',
        'Congratulations, you won a free vacation!',
        'Let\'s catch up for lunch next week.'
    ],
    'label': ['spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'spam', 'ham']
}
df = pd.DataFrame(data)

# 2. Preprocessing: Convert text to a matrix of token counts
# CountVectorizer will turn each message into a vector of word counts.
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

# 3. Split data and train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# 4. Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred, labels=['ham', 'spam']))

# 5. Test with new messages
new_messages = [
    'Can you send me the report?',
    'Claim your free money now'
]
new_messages_transformed = vectorizer.transform(new_messages)
predictions = model.predict(new_messages_transformed)

print("\n--- New Predictions ---")
for msg, pred in zip(new_messages, predictions):
    print(f"'{msg}' -> {pred}")
```

## Summary

- **Naive Bayes** is a fast, simple, and effective probabilistic classifier.
- It's based on **Bayes' Theorem** with the "naive" assumption that all features are independent.
- It excels at **text classification** problems like spam detection and sentiment analysis.
- Because of its simplicity, it works well even with very high-dimensional data (like a vocabulary of thousands of words) and can perform well with relatively small amounts of training data.

---

[< Previous: Chapter 8: Decision Trees Random Forests](./chapter-08-decision-trees-random-forests.md) | [Next: Chapter 10: KNN >](./chapter-10-knn.md)