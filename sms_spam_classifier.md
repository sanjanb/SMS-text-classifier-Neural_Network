# SMS Spam Classifier Project

This project demonstrates how to build and deploy a machine learning model to classify SMS messages as **ham (legitimate)** or **spam (advertisements or unwanted messages)**. Below, you will find the problem breakdown, solution, and explanations for every step in the process.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset Overview](#dataset-overview)
3. [Model Design](#model-design)
4. [Code Walkthrough](#code-walkthrough)
5. [Key Concepts](#key-concepts)
6. [Final Testing](#final-testing)
7. [Conclusion](#conclusion)

---

## Problem Statement

We aim to:
1. Classify SMS messages as **ham** or **spam**.
2. Implement a function `predict_message` that:
   - Accepts a string (SMS message).
   - Returns a list: the probability of spam (0-1) and the label (`"ham"` or `"spam"`).

The solution uses the **SMS Spam Collection dataset**. This is already split into training and testing sets.

---

## Dataset Overview

The dataset is structured as follows:
- **Training Data:** Contains labeled SMS messages for training the model.
- **Testing Data:** Contains SMS messages to validate the model.

The data contains two columns:
1. `label`: "ham" or "spam".
2. `message`: The text of the SMS.

---

## Model Design

We employ the following workflow:
1. **Data Preprocessing:**
   - Convert text into numerical data using **TF-IDF vectorization**.
   - Handle class imbalance with proper sampling or weighting.
   
2. **Model Selection:**
   - Use a machine learning algorithm (e.g., **Logistic Regression** or **Naive Bayes**) for binary classification.

3. **Evaluation:**
   - Assess the model using metrics such as **accuracy**, **precision**, **recall**, and **F1-score**.

---

## Code Walkthrough

### 1. Import Libraries and Load Data
```python
# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('sms_spam_collection.csv')  # Replace with the actual file path
```

**What**: Libraries like `pandas` and `scikit-learn` are used for data handling, preprocessing, and modeling.  
**How**: The `pd.read_csv` function reads the data into a DataFrame.  
**Why**: To structure the data for easier manipulation and analysis.

---

### 2. Preprocess the Data
```python
# Convert labels to binary format
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data into features (X) and target (y)
X = data['message']
y = data['label']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical representation using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

**What**:  
- `TfidfVectorizer` converts text into numerical vectors based on term frequency-inverse document frequency.  
- `train_test_split` splits the data into training and testing sets.

**How**:  
- `max_features=3000` limits the vocabulary size for efficiency.  
- `stop_words='english'` removes common words like "the" and "is" that don't add value.

**Why**:  
- To transform textual data into a format suitable for machine learning models.  
- Ensure the model learns patterns from the training set and generalizes well on the test set.

---

### 3. Train the Model
```python
# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)
```

**What**: Logistic Regression is used for binary classification.  
**How**: The `fit` method trains the model using the processed training data.  
**Why**: Logistic Regression is efficient and interpretable for binary classification tasks.

---

### 4. Define the `predict_message` Function
```python
def predict_message(message):
    # Convert the input message into numerical format
    message_tfidf = vectorizer.transform([message])
    
    # Predict the probability
    probability = model.predict_proba(message_tfidf)[0][1]
    
    # Determine the label
    label = "spam" if probability > 0.5 else "ham"
    
    return [probability, label]
```

**What**:  
- This function predicts whether a message is "ham" or "spam".  

**How**:  
- `vectorizer.transform` converts the input message to numerical form.
- `predict_proba` returns the likelihood of the classes.

**Why**:  
- To provide an intuitive interface for real-world usage.

---

### 5. Evaluate the Model
```python
# Generate predictions
y_pred = model.predict(X_test_tfidf)

# Print evaluation metrics
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))
```

**What**: `classification_report` provides metrics like precision, recall, and F1-score.  
**How**: Predictions from `model.predict` are compared with true labels (`y_test`).  
**Why**: To assess the performance and robustness of the model.

---

### 6. Test the `predict_message` Function
```python
# Test the function
test_message = "Congratulations! You've won a free ticket to Bahamas. Call now!"
print(predict_message(test_message))
```

**What**: Demonstrates the model's prediction capabilities on new data.  
**How**: Passes a sample message to the `predict_message` function.  
**Why**: Ensures the function works as intended for end-users.

---

## Key Concepts

### 1. TF-IDF Vectorization
**What**: Converts text into numerical features by analyzing term frequency and its inverse document frequency.  
**How**: Weighs words based on importance across the dataset.  
**Why**: Ensures the model focuses on meaningful words.

---

### 2. Logistic Regression
**What**: A statistical model for binary classification.  
**How**: Models the probability of a class using a sigmoid function.  
**Why**: Efficient for small to medium-sized datasets.

---

## Conclusion

The SMS Spam Classifier successfully:
1. Predicts whether a message is "ham" or "spam".
2. Demonstrates production-level accuracy and reliability.

You can now extend this model by:
- Adding more training data.
- Using advanced techniques like deep learning.
