# SMS Spam Classifier 📩🚫

This project is a machine learning application that classifies SMS messages as either **ham** (regular messages) or **spam** (unwanted messages). It uses a neural network model built with TensorFlow and Python.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Description](#model-description)
- [Setup Instructions](#setup-instructions)
- [How It Works](#how-it-works)
- [File Breakdown](#file-breakdown)
  - [code-breakdown.md](#code-breakdownmd)
  - [sms_spam_classifier.md](#sms_spam_classifierrmd)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

The SMS Spam Classifier is designed to predict whether an SMS message is:
- **Ham**: A normal message.
- **Spam**: Unwanted advertisement or malicious content.

### Key Features
- Utilizes the SMS Spam Collection dataset.
- Preprocesses text data using tokenization and padding.
- Implements a simple neural network with an embedding layer for text classification.
- Provides a `predict_message` function to classify any SMS message.

---

## Dataset

The project uses the **SMS Spam Collection dataset**, provided as:
- `train-data.tsv`: Training dataset.
- `valid-data.tsv`: Validation dataset.

Each dataset contains two columns:
1. `label`: Indicates whether the message is "ham" or "spam."
2. `message`: The actual SMS content.

---

## Model Description

### Architecture
The model is built using TensorFlow and includes:
1. **Embedding Layer**: Converts words into dense vectors.
2. **Global Average Pooling Layer**: Aggregates word vectors into a fixed-size representation.
3. **Dense Layers**: Two fully connected layers for feature extraction and classification.

### Compilation
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy

### Evaluation
The model is tested with a custom function, ensuring correct classification of predefined test cases.

---

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sms-spam-classifier.git
   cd sms-spam-classifier
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the datasets:
   ```bash
   wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
   wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv
   ```

4. Run the project notebook or script:
   ```bash
   python sms_spam_classifier.py
   ```

5. Test predictions:
   - Modify `pred_text` in the script or notebook.
   - Run `test_predictions()` to validate the model.

---

## How It Works

The project works in the following stages:
1. **Data Preprocessing**: Converts text messages into numerical format using tokenization and padding.
2. **Model Training**: Trains a neural network on the processed data.
3. **Prediction**: Uses the trained model to classify new messages via the `predict_message` function.
4. **Evaluation**: Tests the model against predefined messages to validate accuracy.

---

## File Breakdown

### [code-breakdown.md](./code-breakdown.md)
This document provides an in-depth explanation of the project's codebase:
- Code structure
- Functionality of each component
- How each block contributes to the overall workflow

### [sms_spam_classifier.md](./sms_spam_classifier.md)
This document includes:
- A detailed walkthrough of the SMS Spam Classifier implementation.
- Explanations of the key concepts, including:
  - Data preprocessing
  - Neural network architecture
  - Model evaluation

---

## Contributing

We welcome contributions! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Add feature"`).
4. Push the branch (`git push origin feature-name`).
5. Open a pull request.

---

## License

This project is licensed under the [MIT License](./LICENSE).

---

Happy coding! 🚀
