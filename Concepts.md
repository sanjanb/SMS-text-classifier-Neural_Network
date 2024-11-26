# **SMS Spam Classification with Deep Learning**

## **Table of Contents**
1. [Introduction](#introduction)  
2. [Problem Statement](#problem-statement)  
3. [Concepts and Techniques](#concepts-and-techniques)  
4. [Tools and Libraries](#tools-and-libraries)  
5. [Project Workflow](#project-workflow)  
6. [Data Preprocessing](#data-preprocessing)  
7. [Deep Learning Concepts](#deep-learning-concepts)  
8. [Model Architecture](#model-architecture)  
9. [Model Training](#model-training)  
10. [Evaluation and Testing](#evaluation-and-testing)  
11. [Deployment](#deployment)  
12. [Conclusion](#conclusion)

---

## **1. Introduction**

This project demonstrates how to classify SMS messages as **spam** or **ham** using **Natural Language Processing (NLP)** and **Deep Learning**.  
- **Spam**: Unsolicited or promotional messages.  
- **Ham**: Legitimate (non-spam) messages.  

The solution involves building a **binary classification model** using text data.

---

## **2. Problem Statement**

Given an SMS message, determine whether it is spam or ham.  
For example:  
- Message: `"Congratulations! You've won a $1000 gift card!"` → `Spam`  
- Message: `"Let's meet for lunch tomorrow."` → `Ham`

---

## **3. Concepts and Techniques**

### **Text Data Preprocessing**  
- **Label Encoding**: Converts text labels (ham/spam) to numeric form.  
- **Tokenization**: Breaks text into smaller units (tokens).  
- **Text-to-Sequence Conversion**: Maps tokens to numeric indices.  
- **Padding**: Ensures all text sequences have the same length by adding zeros.

### **Deep Learning for Text**  
- **Embedding Layer**: Converts words to dense vector representations.  
- **GlobalAveragePooling**: Reduces sequence dimensions while retaining key information.  
- **Dense Layers**: Learn relationships between input features and the target variable.

---

## **4. Tools and Libraries**

- **TensorFlow**: An open-source library for building and training deep learning models.  
- **Pandas**: For data manipulation and analysis.  
- **Matplotlib**: For visualizing training performance.  
- **Keras**: A high-level API for TensorFlow to build neural networks easily.

---

## **5. Project Workflow**

### Overview of Steps:
1. **Data Collection**: Load and explore SMS datasets.
2. **Data Preprocessing**: Tokenize, sequence, and pad the text data.
3. **Model Design**: Build a neural network architecture.
4. **Model Training**: Train the model on labeled data.
5. **Evaluation**: Assess model accuracy on test data.
6. **Deployment**: Use the model to classify new messages.

---

## **6. Data Preprocessing**

Preprocessing text is essential for making it usable in a deep learning model.

### **1. Label Encoding**
- Labels (`ham` and `spam`) are converted into numerical values:
  - `ham` → 0
  - `spam` → 1  
```python
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
```

### **2. Tokenization**
- Converts words into tokens (unique numerical IDs).  
  - Example:  
    `"You have won"` → `[1, 7, 54]`  
```python
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train_sequences = tokenizer.texts_to_sequences(X_train)
```

### **3. Text Padding**
- Ensures uniform sequence lengths by adding zeros to shorter texts.  
  - Example:  
    `[1, 7, 54]` → `[1, 7, 54, 0, 0]`  
```python
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_sequences, maxlen=120, padding='post')
```

---

## **7. Deep Learning Concepts**

### **1. Embedding Layer**
- Converts tokens into dense vectors of fixed size.  
- Captures **semantic relationships** between words.  
  - Example:  
    - "king" and "queen" might have embeddings close to each other.  
    - `"Hello"` → `[0.25, 0.12, -0.43, ...]`

### **2. Global Average Pooling**
- Reduces sequence dimensions to a single vector by averaging embeddings.  
- Simplifies input for subsequent layers.  

### **3. Dense Layers**
- Fully connected layers for extracting patterns and relationships in the data.  
- The last dense layer uses **sigmoid activation** to output probabilities between 0 and 1.

---

## **8. Model Architecture**

The model is built using **TensorFlow/Keras Sequential API**.

| **Layer**                     | **Purpose**                                                               |
|-------------------------------|---------------------------------------------------------------------------|
| **Embedding**                 | Converts word indices to dense vector representations.                   |
| **GlobalAveragePooling1D**    | Reduces sequence dimensions to a single vector by averaging.              |
| **Dense (ReLU)**              | Extracts meaningful patterns from the data.                              |
| **Dense (Sigmoid)**           | Outputs a probability for binary classification.                         |

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=120),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

---

## **9. Model Training**

### **Compilation**
- **Optimizer**: Adam (adaptive learning rates).  
- **Loss**: Binary crossentropy (suitable for binary classification).  
- **Metrics**: Accuracy.  

```python
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### **Training**
- Train the model over **10 epochs** with training and validation datasets.  

```python
history = model.fit(
    X_train_padded,
    y_train,
    epochs=10,
    validation_data=(X_test_padded, y_test),
    verbose=2
)
```

---

## **10. Evaluation and Testing**

### **Visualizing Training Performance**
- Plot training and validation accuracy over epochs to check for overfitting.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
```

### **Predicting New Messages**
- Use the trained model to classify SMS messages.
- Output:  
  - **Probability**: Likelihood of being spam.  
  - **Label**: `ham` or `spam`.

```python
def predict_message(pred_text):
    pred_sequence = tokenizer.texts_to_sequences([pred_text])
    pred_padded = tf.keras.preprocessing.sequence.pad_sequences(pred_sequence, maxlen=120, padding='post')
    probability = model.predict(pred_padded)[0][0]
    label = "spam" if probability > 0.5 else "ham"
    return [probability, label]
```

---

## **11. Deployment**

Deploy the model in real-world applications, such as:
- SMS filtering systems.
- Email spam detection tools.

---

## **12. Conclusion**

This project combines **NLP** and **Deep Learning** to build a robust SMS spam classifier. You’ve learned:
- How to preprocess text data.  
- How to build and train a neural network.  
- How to deploy a predictive model for classification tasks.

> 🧠 **Keep experimenting!** Try using different architectures, datasets, and techniques to further enhance your understanding.
