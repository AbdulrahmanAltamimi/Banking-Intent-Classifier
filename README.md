# Banking Intent Classification

A multi-class intent classification system for banking customer support queries using BERT and traditional machine learning models.

<p align="center">
  <a href="https://colab.research.google.com/drive/1sMfKzbwhV3Q2_IhXXfINbGGUgRkLXAL_?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
  <a href="https://huggingface.co/spaces/abdulrahmanMoh/banking-intent-classifier">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces"/>
  </a>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Transformers-4.35+-yellow.svg" alt="Transformers"/>
</p>

---

## Overview

This project implements a comprehensive intent classification system that categorizes banking customer queries into **77 distinct intents**. The system compares multiple machine learning approaches, from traditional models to state-of-the-art transformers.

### Key Features

- **77 Banking Intent Classes** covering cards, transfers, top-ups, verification, and more
- **Multiple Model Comparison**: Naive Bayes, Logistic Regression, SVM, Random Forest, MLP, Ensemble methods, and BERT
- **BERT Transformer** achieving 92.5% F1-Score
- **GPU-Accelerated Training** with mixed precision (FP16)
- **Production-Ready Pipeline** with saved models for deployment
- **Interactive Demo** on Hugging Face Spaces

---

## Demo

Try the live demo on Hugging Face Spaces:

<p align="center">
  <a href="https://huggingface.co/spaces/abdulrahmanMoh/banking-intent-classifier">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Banking%20Intent%20Classifier-blue?style=for-the-badge" alt="Live Demo"/>
  </a>
</p>

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **BERT** | **92.47%** | **92.85%** | **92.47%** | **92.48%** |
| Stacking Classifier | 87.47% | 87.79% | 87.47% | 87.45% |
| SVM (LinearSVC) | 85.39% | 85.85% | 85.39% | 85.39% |
| Voting Classifier | 84.42% | 85.29% | 84.42% | 84.34% |
| MLP Classifier | 84.09% | 85.00% | 84.09% | 84.04% |
| Logistic Regression | 83.05% | 84.16% | 83.05% | 82.96% |
| Random Forest | 82.82% | 83.48% | 82.82% | 82.78% |
| Naive Bayes | 79.32% | 81.01% | 79.32% | 78.41% |

---

## Dataset

**Banking77 Dataset** from PolyAI

| Split | Samples | Classes |
|-------|---------|---------|
| Training | 10,003 | 77 |
| Test | 3,080 | 77 |

### Intent Categories

The 77 intents cover various banking operations:

- **Card Management**: activation, arrival, delivery, lost/stolen, declined payments
- **Transfers**: failed, pending, fees, timing, cancellation
- **Top-ups**: by card, bank transfer, cash, failed, limits
- **Verification**: identity, source of funds, top-up verification
- **Account**: balance updates, termination, personal details
- **ATM**: support, cash withdrawal issues, wrong amounts
- **And more...**

---

## Project Structure

```
banking-intent-classification/
|-- banking_intent_classification.ipynb  # Main training notebook
|-- Data/
|   |-- banking_train.csv                # Training data
|   |-- banking_test.csv                 # Test data
|   |-- download_data.py                 # Data download script
|-- streamlit_app/
|   |-- streamlit_app.py                 # Streamlit demo app
|   |-- requirements.txt
|   |-- README.md
|-- README.md
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/banking-intent-classification.git
cd banking-intent-classification

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
transformers>=4.35.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
nltk>=3.8.0
contractions>=0.1.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
```

---

## Usage

### Training

Open the notebook in Google Colab or run locally:

<a href="https://colab.research.google.com/drive/1sMfKzbwhV3Q2_IhXXfINbGGUgRkLXAL_?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Inference with BERT

```python
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load model
model = BertForSequenceClassification.from_pretrained('abdulrahmanMoh/bert_Banking77')
tokenizer = BertTokenizer.from_pretrained('abdulrahmanMoh/bert_Banking77')

# Predict
text = "I want to transfer money to another account"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

print(f"Predicted Intent ID: {prediction}")
```

### Inference with Sklearn Pipeline

```python
import joblib

# Load pipeline and encoder
pipeline = joblib.load('saved_models/best_sklearn_model.pkl')
label_encoder = joblib.load('saved_models/label_encoder.pkl')

# Predict
text = "My card was declined at the store"
prediction = pipeline.predict([text])[0]
intent = label_encoder.inverse_transform([prediction])[0]

print(f"Predicted Intent: {intent}")
```

### Run Streamlit App Locally

```bash
cd streamlit_app
streamlit run app.py
```

---

## Model Architecture

### BERT Configuration

- **Base Model**: `bert-base-uncased`
- **Max Sequence Length**: 128 tokens
- **Training Epochs**: 10 (with early stopping)
- **Batch Size**: 16 (train), 32 (eval)
- **Learning Rate**: 5e-5 with warmup
- **Optimizer**: AdamW with weight decay 0.01

### Training Details

- GPU: Tesla T4 (Google Colab)
- Training Time: ~9 minutes
- Mixed Precision: FP16 enabled
- Early Stopping: Patience of 2 epochs

---

## Results Visualization

### Model Comparison

The BERT model significantly outperforms traditional ML approaches, achieving a 5% improvement in F1-score over the best ensemble method.

### Confusion Matrix

The confusion matrix shows strong diagonal dominance, indicating good classification across all 77 intent classes.

### Training Curves

- Training and validation loss converge smoothly
- Validation metrics (accuracy, F1, precision, recall) improve consistently across epochs

---

## References

- [Banking77 Dataset](https://github.com/PolyAI-LDN/task-specific-datasets)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---


## Author

**Abdulrahman Mohammed Al-tamimi**

<p align="center">
  <a href="https://github.com/AbdulrahmanAltamimi">
    <img src="https://img.shields.io/badge/GitHub-abdulrahmanMoh-black?style=flat&logo=github" alt="GitHub"/>
  </a>
  <a href="https://huggingface.co/abdulrahmanMoh">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-abdulrahmanMoh-yellow" alt="Hugging Face"/>
  </a>
</p>


