# Fake Review Detection using SVM

This project detects fake product reviews using Natural Language Processing and Support Vector Machine (SVM). It processes over 40,000 Amazon product reviews and classifies them as genuine or fake.

## Project Overview

- **Dataset Size**: 40,000+ reviews
- **Features Used**: Review text, category, rating
- **Target**: Label (`CG` = Genuine, `OR` = Fake)

## Notebooks

- `Dataset_Preprocessing.ipynb`: 
  - Handles text cleaning, lemmatization, stopword removal
  - Performs EDA and visualizations
  - Saves the cleaned dataset

- `Model_SVM.ipynb`: 
  - Converts text to features using TF-IDF
  - Balances dataset using SMOTE
  - Trains a linear SVM classifier
  - Evaluates and exports the model

## Techniques Used

- Text preprocessing (regex, stopwords, lemmatization)
- TF-IDF vectorization with unigrams and bigrams
- SMOTE for handling class imbalance
- SVM with linear kernel for classification

## Model Performance

- **Accuracy**: 90%
- **F1-Score**: 0.90
- Balanced performance across both classes

## Outputs

- `svm_model.pkl`: Trained SVM model
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer

## How to Use

1. Run the preprocessing notebook to clean the raw data.
2. Execute the model notebook to train, evaluate, and export the model.
3. Use the exported model and vectorizer for predictions on new data.

---

This project demonstrates a complete NLP pipeline from preprocessing to model deployment-ready artifacts.
