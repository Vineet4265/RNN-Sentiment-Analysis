# RNN Sentiment Analysis on IMDB Movie Reviews

## Project Overview
This project classifies movie reviews as positive or negative using a Recurrent Neural Network (RNN) built with PyTorch. Text is cleaned with NLP preprocessing, converted into TF-IDF features and passed to the network for binary sentiment classification.

## Problem Statement
Reading large volumes of customer or audience reviews by hand is slow. This project builds a deep learning model that predicts the sentiment of a review automatically.

## Dataset
- IMDB movie reviews dataset with two columns: `review` (text) and `sentiment` (positive / negative)
- Total records: 50,000 (49,582 after removing duplicates)
- No missing values
- The dataset is not included in this repository. Download the IMDB dataset (`IMDB Dataset.csv`) and place it in the project folder to run the notebook.

## Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- PyTorch

## Project Workflow
1. Data loading and cleaning (missing values, duplicates)
2. Text preprocessing: lowercasing, removing URLs, punctuation, HTML tags and stopwords, and Porter stemming
3. Label encoding of the target (negative = 0, positive = 1)
4. TF-IDF vectorization (top 5,000 features)
5. Train-test split (80% train: 39,665 reviews, 20% test: 9,917 reviews)
6. PyTorch datasets and data loaders (batch size 64)
7. Building and training the RNN
8. Evaluation on the test set

## Model
- Framework: PyTorch
- Architecture: RNN layer (1 layer, 128 hidden units) followed by a fully connected layer with a sigmoid output
- Loss function: Binary Cross-Entropy
- Optimizer: Adam
- Epochs: 10

## Results
- Test accuracy: **85.16%** on 9,917 held-out reviews

## Limitations and Future Improvements
- Each review is passed to the RNN as a single TF-IDF vector (sequence length of 1), so the model does not read the words in order. A next step is to use word embeddings with an LSTM or GRU over the token sequence.
- Only accuracy was measured. Adding precision, recall, F1 score and a confusion matrix would give a fuller evaluation.
- The TF-IDF vectorizer was fitted before the train-test split. Fitting it on the training data only would avoid leakage.

## Author
Vineet Kumar

GitHub: https://github.com/Vineet4265

LinkedIn: https://www.linkedin.com/in/vineet-kumar-b08a90239/


