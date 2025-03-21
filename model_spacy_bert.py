#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_spacy_bert.py

This script combines spaCy for text preprocessing with a BERT-based sentiment analysis model.
It uses the 'nlptown/bert-base-multilingual-uncased-sentiment' pipeline from Hugging Face to predict sentiment.
The script evaluates performance using F1-score and displays a confusion matrix.

Requirements:
  - 'FinancES_phase_2_train_public.csv' dataset
  - spaCy Spanish model: 'es_core_news_sm'
"""

import pandas as pd
import spacy
import unicodedata
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import pipeline
import matplotlib.pyplot as plt

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
    df_train = df_train.sample(frac=1, random_state=42)
    
    allowed_sentiments = ['negative', 'positive', 'neutral']
    for col in ['target_sentiment', 'companies_sentiment', 'consumers_sentiment']:
        df_train = df_train[df_train[col].isin(allowed_sentiments)]
    
    df_train['processed_text'] = df_train['text'].apply(lambda x: preprocess(x, nlp))
    df_train['processed_target'] = df_train['target'].apply(lambda x: preprocess(x, nlp))
    
    def extract_keywords(text):
        tokens = preprocess(text, nlp)
        if not tokens:
            return None
        return pd.Series(tokens).value_counts().idxmax()
    
    df_train['predicted_target'] = df_train['text'].apply(extract_keywords)
    df_train['processed_target_str'] = df_train['processed_target'].apply(lambda x: ' '.join(x) if x else '')
    
    acc = accuracy_score(df_train['processed_target_str'].fillna(''), df_train['predicted_target'].fillna(''))
    print(f"Accuracy for predicted_target: {acc}")
    
    # Load the BERT-based sentiment analysis pipeline
    classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
    
    def stars_to_sentiment(stars_label):
        stars = int(stars_label[0])
        if stars in [1, 2]:
            return 'negative'
        elif stars == 3:
            return 'neutral'
        elif stars in [4, 5]:
            return 'positive'
    
    def predict_sentiment(processed_text, predicted_target):
        processed_text_str = ' '.join(processed_text) if processed_text else ''
        predicted_target_str = predicted_target if predicted_target else ''
        combined_text = processed_text_str + ' ' + predicted_target_str
        result = classifier(combined_text)
        if not result:
            return 'neutral'
        stars_label = result[0]['label']
        return stars_to_sentiment(stars_label)
    
    df_train['predicted_sentiment'] = df_train.apply(
        lambda row: predict_sentiment(row['processed_text'], row['predicted_target']),
        axis=1
    )
    
    f1_val = f1_score(df_train['target_sentiment'], df_train['predicted_sentiment'], average='weighted')
    print("F1-Score for target_sentiment: ", f1_val)
    
    cm = confusion_matrix(df_train['target_sentiment'], df_train['predicted_sentiment'], labels=['positive', 'neutral', 'negative'])
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
