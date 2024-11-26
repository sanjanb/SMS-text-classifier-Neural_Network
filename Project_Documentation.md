 **SMS Spam Classification Project Documentation**

### **Introduction**
This project involves building a machine learning model using deep learning to classify SMS messages as either "spam" or "ham" (not spam). The model uses a binary classification approach with TensorFlow and Keras, leveraging Natural Language Processing (NLP) techniques.

---

## **Project Workflow**

### **1. Libraries and Dependencies**
We begin by installing and importing necessary libraries:
- **`tensorflow`**: Core library for building and training deep learning models.
- **`pandas`**: For loading and processing tabular data.
- **`numpy`**: Used for numerical computations.
- **`matplotlib`**: To visualize training and validation performance.

### **Code:**
```python
try:
    !pip install tf-nightly
except Exception:
    pass

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

### **Why?**
- These libraries provide essential tools for text processing, tokenization, and deep learning model training.

---

### **2. Data Loading**
We download and load the training and test datasets:
- **Datasets:** Tab-separated files (`.tsv`) containing SMS messages labeled as "ham" or "spam."

### **Code:**
```python
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=["label", "message"])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=["label", "message"])
```

### **Why?**
- The dataset forms the foundation for training and testing the model. Loading it into a structured format (`pandas DataFrame`) allows for easy manipulation.

---

### **3. Data Preprocessing**
To prepare the dataset:
1. **Map labels to binary format:** Convert "ham" to 0 and "spam" to 1.
2. **Split into features and labels:**
   - Features (`X`): SMS text.
   - Labels (`y`): Binary classification labels.

### **Code:**
```python
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

X_train = train_data['message'].values
y_train = train_data['label'].values
X_test = test_data['message'].values
y_test = test_data['label'].values
```

### **Why?**
- Machine learning models operate on numerical data, so categorical labels must be converted to numeric form.

---

### **4. Text Tokenization**
Tokenization converts text into sequences of numeric tokens:
1. **Tokenizer:** Create a tokenizer that handles vocabulary size (10,000 words) and unknown words (`<OOV>`).
2. **Text to sequences:** Convert text to numeric sequences.
3. **Padding:** Ensure all sequences have uniform length (120).

### **Code:**
```python
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_sequences, maxlen=120, padding='post')
X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_sequences, maxlen=120, padding='post')
```

### **Why?**
- Tokenization converts text data into a format understandable by neural networks.
- Padding ensures uniform sequence length, which is required for efficient model training.

---

### **5. Model Architecture**
The deep learning model comprises the following layers:
1. **Embedding Layer:** Maps token indices to dense vectors of fixed size (16 dimensions).
2. **Global Average Pooling Layer:** Reduces the output of the embedding layer into a single vector by averaging.
3. **Dense Layer (ReLU):** Applies a fully connected layer with 16 neurons and ReLU activation.
4. **Dense Layer (Sigmoid):** Outputs a single value (probability of being spam).

### **Code:**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=120),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### **Why?**
- The architecture balances simplicity and effectiveness for text classification.

---

### **6. Model Compilation and Training**
The model is compiled with:
- **Optimizer:** Adam (adaptive learning rate).
- **Loss Function:** Binary Crossentropy (suitable for binary classification).
- **Metrics:** Accuracy.

### **Code:**
```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train_padded,
    y_train,
    epochs=10,
    validation_data=(X_test_padded, y_test),
    verbose=2
)
```

### **Why?**
- Training the model optimizes weights to minimize loss and improve accuracy over 10 epochs.

---

### **7. Prediction Function**
A function to classify new messages as "ham" or "spam."

### **Code:**
```python
def predict_message(pred_text):
    pred_sequence = tokenizer.texts_to_sequences([pred_text])
    pred_padded = tf.keras.preprocessing.sequence.pad_sequences(pred_sequence, maxlen=120, padding='post')
    probability = model.predict(pred_padded)[0][0]
    label = "spam" if probability > 0.5 else "ham"
    return [probability, label]
```

### **Why?**
- This function converts text input into a numeric sequence, feeds it into the model, and interprets the result.

---

### **8. Testing the Model**
The `test_predictions` function evaluates the model against predefined messages to validate its generalization.

### **Code:**
```python
def test_predictions():
    test_messages = [...]
    test_answers = [...]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")
test_predictions()
```

---

## **Conclusion**
This project demonstrates building a spam classifier using NLP techniques and TensorFlow. Key concepts like tokenization, padding, and embedding were utilized to process text data and train an efficient deep learning model. The model achieves accurate predictions, making it a robust solution for SMS spam classification.
