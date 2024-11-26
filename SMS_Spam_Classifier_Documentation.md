# **SMS Spam Classification - Complete Documentation**

## **Table of Contents**
1. [Introduction](#1-introduction)  
2. [Problem Statement](#2-problem-statement)  
3. [Complete Concepts Used](#3-complete-concepts-used)  
    - [Data Loading](#data-loading)  
    - [Label Encoding](#label-encoding)  
    - [Text Tokenization](#text-tokenization)  
    - [Padding and Sequence Formatting](#padding-and-sequence-formatting)  
    - [Deep Learning Fundamentals](#deep-learning-fundamentals)  
    - [Embedding Layers](#embedding-layers)  
    - [Global Average Pooling](#global-average-pooling)  
    - [Dense Layers](#dense-layers)  
    - [Model Compilation](#model-compilation)  
    - [Model Training](#model-training)  
    - [Evaluation and Metrics](#evaluation-and-metrics)  
    - [Text Prediction and Testing](#text-prediction-and-testing)  
4. [Step-by-Step Workflow](#4-step-by-step-workflow)  
5. [Conclusion](#5-conclusion)

---

## **1. Introduction**

Spam messages clutter inboxes and reduce productivity. This project classifies SMS messages into **ham** (legitimate) or **spam** (unwanted promotional messages) using **Natural Language Processing (NLP)** and **Deep Learning**.

---

## **2. Problem Statement**

**What:** Build a model that classifies SMS messages as **ham** or **spam**.  
**Why:** Automating spam detection saves time and protects users from unsolicited messages.  
**How:** By training a neural network on labeled text data and deploying it for predictions.

---

## **3. Complete Concepts Used**

### **Data Loading**

**What:** Load datasets in `.tsv` format using **Pandas**, a powerful library for data manipulation.  
**Why:** Efficiently process tabular data for training and evaluation.  
**How:**  
1. Use `pd.read_csv()` to load files.
2. Assign column names `label` and `message`.

```python
import pandas as pd
train_data = pd.read_csv("train-data.tsv", sep="\t", header=None, names=["label", "message"])
test_data = pd.read_csv("valid-data.tsv", sep="\t", header=None, names=["label", "message"])
```

---

### **Label Encoding**

**What:** Convert text labels (`ham`, `spam`) into binary numbers (`0`, `1`).  
**Why:** Neural networks process numeric data, not text.  
**How:** Use `map()` to transform labels:
```python
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
```

---

### **Text Tokenization**

**What:** Split text into smaller units (tokens) and map them to numeric indices.  
**Why:** Models can’t process raw text, but they can interpret numeric representations.  
**How:**  
1. Use TensorFlow's `Tokenizer` to generate tokens.  
2. Create a word-to-index mapping.  
3. Handle unseen words with `<OOV>` (Out Of Vocabulary) tokens.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(train_data['message'])
```

Example:  
Message: `"Win cash now!"` → Tokens: `[1, 23, 45]`

---

### **Padding and Sequence Formatting**

**What:** Make sequences uniform by padding shorter ones and truncating longer ones.  
**Why:** Neural networks require inputs of consistent dimensions.  
**How:** Use TensorFlow’s `pad_sequences()` to pad or truncate messages.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train_sequences = tokenizer.texts_to_sequences(train_data['message'])
X_train_padded = pad_sequences(X_train_sequences, maxlen=120, padding='post')
```

Example:  
Input: `[1, 23]` → Padded: `[1, 23, 0, 0]` (max length = 4)

---

### **Deep Learning Fundamentals**

**What:** Neural networks mimic the brain to process and learn from data.  
**Why:** Enable computers to solve complex problems like text classification.  
**How:** Layers of neurons process data, applying mathematical functions to generate outputs.

Key terms:
- **Epochs:** Number of times the model sees the training data.
- **Forward pass:** Input flows through the network.
- **Backward pass:** Network adjusts weights to minimize error.

---

### **Embedding Layers**

**What:** Transform words into dense, low-dimensional vectors that represent semantic meaning.  
**Why:** Capture relationships between words (e.g., "king" and "queen" are related).  
**How:** Use TensorFlow's `Embedding` layer.

```python
tf.keras.layers.Embedding(input_dim=10000, output_dim=16, input_length=120)
```

Example:  
Word: `"hello"` → Vector: `[0.2, -0.1, 0.4, 0.9]`

---

### **Global Average Pooling**

**What:** Reduce the dimensions of word embeddings by averaging all vectors in a sequence.  
**Why:** Summarize the sequence meaningfully without retaining positional details.  
**How:** Use TensorFlow’s `GlobalAveragePooling1D`.

```python
tf.keras.layers.GlobalAveragePooling1D()
```

---

### **Dense Layers**

**What:** Fully connected layers in a neural network.  
**Why:** Combine features and make predictions.  
**How:**  
1. Use **ReLU activation** to introduce non-linearity.  
2. Use **Sigmoid activation** in the output layer for binary classification.

```python
tf.keras.layers.Dense(16, activation='relu')
tf.keras.layers.Dense(1, activation='sigmoid')
```

---

### **Model Compilation**

**What:** Define how the model learns.  
**Why:** Specify the optimizer, loss function, and evaluation metrics.  
**How:** Use TensorFlow’s `compile()` method.

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

Components:
- **Optimizer (Adam):** Adjusts weights during training.
- **Loss (Binary Crossentropy):** Measures error for binary outputs.
- **Metrics (Accuracy):** Tracks correct predictions.

---

### **Model Training**

**What:** Teach the model to map inputs to outputs by minimizing loss.  
**Why:** Allow the model to generalize patterns in the training data.  
**How:** Use `model.fit()` with training data and validation sets.

```python
history = model.fit(X_train_padded, y_train, epochs=10, validation_data=(X_test_padded, y_test))
```

---

### **Evaluation and Metrics**

**What:** Assess the model's performance on unseen data.  
**Why:** Ensure the model generalizes well.  
**How:** Use `model.evaluate()` to compute accuracy and loss.

```python
model.evaluate(X_test_padded, y_test)
```

---

### **Text Prediction and Testing**

**What:** Predict whether a new SMS message is spam or ham.  
**Why:** Validate the model’s ability to classify real-world messages.  
**How:**  
1. Tokenize and pad the input message.  
2. Predict probabilities using `model.predict()`.  
3. Assign a label (`ham` or `spam`) based on the probability threshold (0.5).

```python
def predict_message(pred_text):
    pred_sequence = tokenizer.texts_to_sequences([pred_text])
    pred_padded = pad_sequences(pred_sequence, maxlen=120, padding='post')
    probability = model.predict(pred_padded)[0][0]
    label = "spam" if probability > 0.5 else "ham"
    return [probability, label]
```

---

## **4. Step-by-Step Workflow**

1. **Load Data:** Read `.tsv` files into Pandas DataFrames.  
2. **Preprocess:** Tokenize text, pad sequences, and encode labels.  
3. **Build Model:** Create a neural network with embedding, pooling, and dense layers.  
4. **Train Model:** Train with labeled data to minimize loss.  
5. **Evaluate Model:** Measure accuracy on validation data.  
6. **Test Predictions:** Classify unseen messages as spam or ham.

---

## **5. Conclusion**

This documentation explains every concept used in SMS classification, from data loading to text prediction. Understanding **what** each step does, **why** it’s necessary, and **how** it’s implemented ensures a deep grasp of text classification workflows.
