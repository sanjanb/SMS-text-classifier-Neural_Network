# **📨 SMS Spam Classification Using Deep Learning**

## **🌟 Objective**
The purpose of this project is to build a **machine learning model** using **Natural Language Processing (NLP)** techniques to classify SMS messages as either:
- **Ham (Non-Spam):** Legitimate messages.  
- **Spam:** Unsolicited or promotional messages.

This is achieved by:
1. Tokenizing the text into sequences.
2. Building and training a **neural network model** in TensorFlow.  
3. Evaluating and deploying the model to predict new messages.

---

## **🧠 What Will You Learn?**
This documentation provides a comprehensive understanding of:
- Text preprocessing for NLP tasks.
- Building neural networks for text data.
- Practical deployment of spam classifiers.
- Key deep learning concepts applied to real-world problems.

---

## **🔄 Workflow**

### Overview of Project Workflow:
1. **Install Libraries**: Set up dependencies.  
2. **Data Collection**: Download and preprocess datasets.  
3. **Data Cleaning**: Tokenize, sequence, and pad the text.  
4. **Model Building**: Define the architecture and compile the model.  
5. **Model Training**: Train the model and monitor performance.  
6. **Evaluation**: Test the model on unseen data.  
7. **Deployment**: Create a prediction function for real-time use.

---

## **📋 Data Description**

We use two datasets for this project:
- **Training Data**: Contains SMS messages with labels (`ham` or `spam`).  
- **Validation Data**: Used to test and validate model performance.

### **Dataset Structure**
| **Column** | **Description**                          |
|------------|------------------------------------------|
| `label`    | `ham` (0) or `spam` (1)                  |
| `message`  | SMS content (e.g., "Congrats! You won.") |

### Why Tab-Separated Values (TSV)?
- TSV files are lightweight and straightforward to parse.  
- They’re ideal for plain-text datasets like SMS messages.

---

## **🔧 Preprocessing Steps**

### 1. **Label Encoding**
Convert textual labels into numerical values:
- `ham` → `0`
- `spam` → `1`

### 2. **Tokenization**
- Breaks text into smaller units (words or tokens).
- Assigns each token a unique numerical index.

### 3. **Text-to-Sequence Conversion**
Transforms the text into a sequence of integers based on token indices.

### 4. **Padding**
- Ensures uniform input size by adding zeroes to shorter sequences.
- Essential for batch processing in neural networks.

---

### **Key Code**
```python
# Map labels to binary values
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

# Sequence Conversion and Padding
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_sequences, maxlen=120, padding='post')
```

> 💡 **Why Tokenization and Padding?**  
> Neural networks process numerical data. By tokenizing text and standardizing input lengths, we transform raw SMS messages into a format suitable for training.

---

## **📐 Model Architecture**

This project uses a **Sequential Neural Network** with the following components:

| **Layer**                     | **Purpose**                                                               |
|-------------------------------|---------------------------------------------------------------------------|
| **Embedding**                 | Converts word indices to dense vectors of fixed size (word embeddings).  |
| **GlobalAveragePooling1D**    | Reduces sequence dimensions to a single vector by averaging.              |
| **Dense (ReLU)**              | Extracts meaningful patterns from the data.                              |
| **Dense (Sigmoid)**           | Outputs a single probability (spam or ham).                              |

### **Key Code**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=120),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

> ⚙️ **Key Concepts**:
> - **Embedding Layer**: Captures relationships between words.  
> - **GlobalAveragePooling**: Reduces dimensionality while preserving key features.  
> - **Dense Layers**: Perform feature extraction and classification.

---

## **🔨 Model Training**

### Training Configuration:
| **Parameter**   | **Value**            |
|------------------|----------------------|
| **Optimizer**   | Adam (adaptive learning rates). |
| **Loss**        | Binary crossentropy (for binary classification). |
| **Metric**      | Accuracy.            |

### **Code**
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

> 🎯 **Tips for Better Training**:
> - Monitor both training and validation accuracy to avoid overfitting.  
> - Use early stopping for large datasets to save time and resources.

---

## **📈 Visualizing Performance**
You can plot the training and validation metrics over epochs to understand model behavior.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.show()
```

> 📊 **Insights**: Ensure the validation accuracy aligns with training accuracy to confirm generalization.

---

## **🔍 Predictions**

The model predicts whether an SMS is spam based on its content.  
It outputs:
1. **Probability**: Spam likelihood.  
2. **Label**: `ham` or `spam`.

### **Key Code**
```python
def predict_message(pred_text):
    pred_sequence = tokenizer.texts_to_sequences([pred_text])
    pred_padded = tf.keras.preprocessing.sequence.pad_sequences(pred_sequence, maxlen=120, padding='post')
    probability = model.predict(pred_padded)[0][0]
    label = "spam" if probability > 0.5 else "ham"
    return [probability, label]
```

> **Example**:  
> Input: *"Congratulations! You won a free iPhone."*  
> Output: `[0.93, "spam"]`

---

## **✅ Validation and Testing**

The model is tested on unseen SMS examples to validate its accuracy.

### Test Dataset
| **Message**                                   | **Expected Output** |
|-----------------------------------------------|----------------------|
| "how are you doing today?"                    | ham                 |
| "sale today! to stop texts call 98912460324." | spam                |
| "you have won $1000. call now!"               | spam                |

---

## **✨ Conclusion**

This project showcases how to use **NLP and deep learning** to solve real-world problems like spam detection. By leveraging:
- **Text preprocessing** techniques.
- **Efficient neural network architectures.**
- **Intuitive visualization tools.**

It achieves robust performance and paves the way for deployment in messaging systems.

---

> 🌟 *Feel free to contribute or suggest improvements via GitHub!*
