---
title: Banking Intent Classifier
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
models:
  - abdulrahmanMoh/bert_Banking77
---

# Banking Intent Classifier

A BERT-based intent classification model for banking customer support queries.

## Model Details

- **Model**: BERT (bert-base-uncased) fine-tuned for sequence classification
- **Dataset**: Banking77 dataset
- **Classes**: 77 banking-related intents
- **Task**: Multi-class text classification

## Usage

Enter a banking-related query and the model will predict the customer intent.

## Example Queries

- "I want to transfer money to another account"
- "My card was declined at the store"
- "How do I change my PIN number?"
- "I lost my credit card yesterday"

## Intent Categories

The model classifies queries into 77 intents including:
- Card-related (activation, lost/stolen, declined, etc.)
- Transfer-related (failed, pending, fees, etc.)
- Account-related (balance, verification, etc.)
- Top-up related (failed, limits, charges, etc.)
- And more...
