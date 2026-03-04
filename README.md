# Natural Language Understanding: Intent Classification & Slot Filling

TensorFlow/Keras implementation of an end-to-end NLP pipeline performing both **intent classification** and **token-level slot extraction** on natural language flight queries.

Natural Language Understanding: Intent Classification & Slot Filling

This project implements an end-to-end Natural Language Understanding (NLU) pipeline capable of performing:

Intent Classification (sentence-level prediction)

Slot Filling (token-level classification)

The system processes natural language flight queries and extracts structured meaning by combining sentence-level context with word-level features using neural network architectures built in TensorFlow / Keras.

Project Overview

Natural language requests often contain two types of information:

1. Intent

The overall goal of the sentence.

Example:

"I want a flight to Boston"

Intent:

flight_booking
2. Slots

Specific structured pieces of information contained within the sentence.

Example:

"I want a flight to Boston"

Slot labels:

O O O O B-city

Where:

O = Not a slot

B-city = Beginning of a city slot

This project builds models that automatically learn to perform both tasks.

System Architecture

The pipeline consists of three main stages:

Data Preprocessing
        ↓
Intent Classification Model
        ↓
Word-to-Slot Classification Model
1. Data Preprocessing

The dataset contains three columns:

Column	Description
query	Natural language sentence
intent	Sentence-level class label
slot filling	Token-level slot annotations

Example:

Query	Intent	Slot Filling
"fly to Boston tomorrow"	flight_query	O O B-city B-date
Sentence-to-Token Transformation

Slot classification requires token-level training examples.

A custom transformation function converts sentence data into word-level samples while preserving alignment with the original sentence.

Example transformation:

Sentence: "fly to Boston"
Slots:    "O O B-city"

Becomes:

Word	Sentence	Slot
fly	fly to Boston	O
to	fly to Boston	O
Boston	fly to Boston	B-city

This allows the model to predict slot labels for each individual token while still seeing the full sentence context.

2. Text Vectorization

Text is converted into numerical feature vectors using Keras TextVectorization.

Sentence Representation

Sentence queries are encoded using:

multi_hot encoding
ngrams = 2

This creates unigram and bigram features that capture word co-occurrence patterns.

Example:

"I want flight"

Features include:

I
want
flight
I want
want flight
Word Representation

Individual tokens are vectorized using a separate vocabulary:

output_mode = multi_hot
ngrams = 1

Each word becomes a sparse vector representing its presence in the vocabulary.

Slot Label Encoding

Slot labels are converted to integer class IDs using a dedicated vectorization layer.

Example mapping:

O       → 1
B-city  → 2
B-time  → 3
3. Intent Classification Model

The intent classifier predicts the overall purpose of a query.

Architecture
Input: multi-hot sentence vector
        ↓
Dense layer
        ↓
Dropout (0.5)
        ↓
Softmax output

This model learns to map sentence-level features to intent categories.

4. Word-to-Slot Classification Model

The slot filling model predicts a slot label for each word in the sentence.

Dual Input Architecture

The model uses two feature sources:

Sentence-level context

Word-level representation

These are combined using feature concatenation.

Architecture
Sentence vector
        ↓
        ┐
         → Concatenate → Dense → Dropout → Softmax
        ┘
Word vector

This allows the model to learn relationships such as:

Boston → likely B-city
tomorrow → likely B-date

while still considering sentence context.

Key NLP Techniques Used

Text vectorization using Keras TextVectorization

Unigram and bigram feature engineering

Sparse multi-hot text encoding

Token-level slot labeling

Vocabulary construction and management

Sentence-to-token dataset transformation

Feature fusion via tensor concatenation

Dual-input neural architectures

Categorical label encoding

Dropout regularization

Machine Learning Techniques

Feedforward neural networks

Softmax multi-class classification

Cross-entropy loss optimization

Tensor shape management

Multi-input model training

Train / validation / test dataset splitting

Technologies Used

Python

TensorFlow

Keras

NumPy

Pandas

What This Project Demonstrates

This project demonstrates the ability to:

Build complete NLP preprocessing pipelines

Transform unstructured text into structured ML features

Design neural architectures for language understanding

Implement token-level classification systems

Combine global sentence context with local token features

Manage tensor shapes and vocabulary construction in deep learning models

Example Use Case

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

Key Technical Contributions
Custom Token-Level Dataset Construction

Implemented custom preprocessing functions to transform sentence-level annotations into aligned token-level training samples. This enables the model to perform slot classification for each individual word while preserving the full sentence context required for accurate prediction.

Dual-Input Neural Architecture for Context-Aware Slot Prediction

Designed a neural network architecture that combines two independent feature sources:

sentence-level query representation

token-level word representation

These features are fused using tensor concatenation, allowing the model to leverage both global sentence context and local token identity when predicting slot labels.

Feature Engineering Using Unigrams and Bigrams

Extended traditional bag-of-words modeling by incorporating bigram features during vectorization. This allows the model to capture short-range contextual relationships between words that improve intent classification performance.

Sparse High-Dimensional Text Representations

Implemented multi-hot text encoding to represent queries as high-dimensional sparse feature vectors. This approach preserves vocabulary-level information while remaining computationally efficient for feedforward neural models.

Structured Vocabulary Management

Built separate vocabularies for:

sentence-level query representation

token-level word inputs

slot label classification

This separation allows each component of the pipeline to be optimized for its specific modeling task.

Feature Fusion via Tensor Concatenation

Implemented feature fusion using the Keras Concatenate layer to merge token-level and sentence-level feature spaces into a single representation used for slot classification.

End-to-End Natural Language Understanding Pipeline

Developed a full NLP pipeline consisting of:

Data preprocessing and token alignment

Feature extraction via text vectorization

Intent classification model

Token-level slot classification model

This pipeline demonstrates how unstructured text can be converted into structured semantic information.

Future Improvements

Potential extensions include:

Sequence models (LSTM / Transformer architectures)

Word embeddings instead of multi-hot vectors

Attention-based slot prediction

Joint intent + slot multitask training
