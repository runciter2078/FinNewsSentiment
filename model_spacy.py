#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_spacy.py

This script uses spaCy for text preprocessing and builds a Random Forest classifier pipeline
with TF-IDF features for sentiment analysis of financial news headlines.
It evaluates the performance (accuracy, F1-score) and displays confusion matrices.

Requirements:
  - 'FinancES_phase_2_train_public.csv' dataset
  - spaCy Spanish model: 'es_core_news_sm'
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import spacy
import unicodedata

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess(text, nlp):
    text = remove_accents(text.lower())
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def main():
    nlp = spacy.load('es_core_news_sm')
    df_train = pd.read_csv('FinancES_phase_2_train_public.csv')
    
    # Filter rows based on allowed sentiments
    allowed_sentiments = ['negative', 'positive', 'neutral']
    for col in ['target_sentiment', 'companies_sentiment', 'consumers_sentiment']:
        df_train = df_train[df_train[col].isin(allowed_sentiments)]
    
    # Preprocess text column (joining tokens to form a string)
    df_train['processed_text'] = df_train['text'].apply(lambda x: ' '.join(preprocess(x, nlp)))
    
    # Split data: 60% train, 20% test, 20% validation
    train_data, temp_data = train_test_split(df_train, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Build a simple pipeline for target_sentiment classification
    vectorizer = TfidfVectorizer()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('clf', clf)
    ])
    
    X_train = train_data['processed_text']
    y_train = train_data['target_sentiment']
    pipeline.fit(X_train, y_train)
    
    X_test = test_data['processed_text']
    y_test = test_data['target_sentiment']
    predictions = pipeline.predict(X_test)
    
    f1_val = f1_score(y_test, predictions, average='weighted')
    print(f"F1-score for target_sentiment: {f1_val}")
    
    cm = confusion_matrix(y_test, predictions, labels=['positive', 'neutral', 'negative'])
    plt.title('Confusion Matrix for target_sentiment')
    plt.imshow(cm, cmap='Oranges', interpolation='nearest')
    plt.colorbar()
    categories = ['positive', 'neutral', 'negative']
    plt.xticks([0, 1, 2], categories)
    plt.yticks([0, 1, 2], categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='black')
    plt.show()
    
if __name__ == "__main__":
    main()
