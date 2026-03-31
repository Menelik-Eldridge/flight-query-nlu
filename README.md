# Natural Language Understanding: Intent Classification & Slot Filling

TensorFlow/Keras implementation of an end-to-end NLU pipeline performing both **intent classification** and **token-level slot extraction** on natural language flight queries.

## Project Overview

This project implements an end-to-end **Natural Language Understanding (NLU) pipeline** capable of performing:

- Intent Classification (sentence-level prediction)
- Slot Filling (token-level classification)

The system processes natural language flight queries and extracts structured meaning by combining sentence-level context with word-level features using neural network architectures built in TensorFlow / Keras.

### 1. Intent

The overall goal of the sentence.

Example:

"I want a flight to Boston"

Intent:

flight_booking

### 2. Slots

Specific structured pieces of information contained within the sentence.

Example:

"I want a flight to Boston"

Slot labels:

O O O O B-city

Where:

O = Not a slot  
B-city = Beginning of a city slot

This project builds models that automatically learn to perform both tasks.

## System Architecture

The pipeline consists of three main stages:

Data Preprocessing
        ↓
Intent Classification Model
        ↓
Word-to-Slot Classification Model

## 1. Data Preprocessing

The dataset contains three columns:

| Column | Description |
|------|------|
| query | Natural language sentence |
| intent | Sentence-level class label |
| slot filling | Token-level slot annotations |

### Sentence-to-Token Transformation

Slot classification requires token-level training examples.

A custom transformation function converts sentence data into word-level samples while preserving alignment with the original sentence.

Example transformation:

Sentence: "fly to Boston"  
Slots:    "O O B-city"

Becomes:

| Word | Sentence | Slot |
|----|----|----|
| fly | fly to Boston | O |
| to | fly to Boston | O |
| Boston | fly to Boston | B-city |

This allows the model to predict slot labels for each individual token while still seeing the full sentence context.

## 2. Text Vectorization

Text is converted into numerical feature vectors using **Keras TextVectorization**.

### Sentence Representation

Sentence queries are encoded using:

multi_hot encoding  
ngrams = 2

This creates unigram and bigram features that capture word co-occurrence patterns.

### Word Representation

Individual tokens are vectorized using a separate vocabulary:

output_mode = multi_hot  
ngrams = 1

### Slot Label Encoding

Slot labels are converted to integer class IDs using a dedicated vectorization layer.

Example mapping:

O → 1  
B-city → 2  
B-time → 3

## 3. Intent Classification Model

The intent classifier predicts the overall purpose of a query.

### Architecture

Input: multi-hot sentence vector  
↓  
Dense layer  
↓  
Dropout (0.5)  
↓  
Softmax output

## 4. Word-to-Slot Classification Model

The slot filling model predicts a slot label for each word in the sentence.

### Dual Input Architecture

The model uses two feature sources:

- Sentence-level context
- Word-level representation

These are combined using feature concatenation.

### Architecture

Sentence vector  
        ↓  
        ┐  
         → Concatenate → Dense → Dropout → Softmax  
        ┘  
Word vector

## Key NLP Techniques Used

- Text vectorization using Keras TextVectorization
- Unigram and bigram feature engineering
- Sparse multi-hot text encoding
- Token-level slot labeling
- Vocabulary construction and management
- Sentence-to-token dataset transformation
- Feature fusion via tensor concatenation
- Dual-input neural architectures
- Categorical label encoding
- Dropout regularization

## Machine Learning Techniques

- Feedforward neural networks
- Softmax multi-class classification
- Cross-entropy loss optimization
- Tensor shape management
- Multi-input model training
- Train / validation / test dataset splitting

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas

## What This Project Demonstrates

This project demonstrates the ability to:

- Build complete NLP preprocessing pipelines
- Transform unstructured text into structured ML features
- Design neural architectures for language understanding
- Implement token-level classification systems
- Combine global sentence context with local token features
- Manage tensor shapes and vocabulary construction in deep learning models

## Example Use Case

Input query:

"book a flight to Boston tomorrow"

Predicted output:

Intent:

flight_booking

Slots:

O O O O B-city B-date

Structured output:

{
  "intent": "flight_booking",
  "city": "Boston",
  "date": "tomorrow"
}

## Key Technical Contributions

### Custom Token-Level Dataset Construction

Implemented preprocessing functions to transform sentence-level annotations into aligned token-level training samples.

### Dual-Input Neural Architecture for Context-Aware Slot Prediction

Designed a neural network architecture combining sentence-level and token-level representations.

### Feature Engineering Using Unigrams and Bigrams

Incorporated bigram features to capture contextual relationships between words.

### Sparse High-Dimensional Text Representations

Implemented multi-hot encoding to represent queries as sparse feature vectors.

### Structured Vocabulary Management

Built separate vocabularies for sentence inputs, word inputs, and slot labels.

### Feature Fusion via Tensor Concatenation

Merged token and sentence feature spaces using the Keras Concatenate layer.

### End-to-End Natural Language Understanding Pipeline

Built a complete pipeline consisting of:

1. Data preprocessing  
2. Feature extraction  
3. Intent classification  
4. Slot classification  

## Future Improvements

- Sequence models (LSTM / Transformer architectures)
- Word embeddings instead of multi-hot vectors
- Attention-based slot prediction
- Joint intent + slot multitask training
