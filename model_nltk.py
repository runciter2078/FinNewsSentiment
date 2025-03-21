#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_nltk.py

This script uses NLTK for text preprocessing and feature extraction.
It extracts the main economic subject using frequency-based methods and builds Random Forest classifiers
for sentiment analysis (target, companies, and consumers). Additionally, it exports the top feature importances to a Word document.

Requirements:
  - 'FinancES_phase_2_train_public.csv' dataset
  - NLTK resources ('stopwords' and 'punkt') will be downloaded automatically.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import unicodedata
from docx import Document

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess(text):
    text = remove_accents(text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def extract_keywords(text):
    tokens = preprocess(text)
    if not tokens:
        return None
    freq = pd.Series(tokens).value_counts()
    return freq.idxmax() if not freq.empty else None

def get_processed_text(data):
    return data['processed_text'].apply(' '.join)

def get_predicted_target(data):
    return data['predicted_target'].fillna('')

def main():
    df_train = pd.read_csv('FinancES_phase_2_train_public.csv')
    
    # Filter rows with allowed sentiments
    allowed_sentiments = ['negative', 'positive', 'neutral']
    for col in ['target_sentiment', 'companies_sentiment', 'consumers_sentiment']:
        df_train = df_train[df_train[col].isin(allowed_sentiments)]
    
    df_train['processed_text'] = df_train['text'].apply(preprocess)
    df_train['processed_target'] = df_train['target'].apply(preprocess)
    
    df_train['predicted_target'] = df_train['text'].apply(extract_keywords)
    df_train['processed_target_str'] = df_train['processed_target'].apply(lambda x: ' '.join(x) if x else '')
    
    acc = accuracy_score(df_train['processed_target_str'].fillna(''), df_train['predicted_target'].fillna(''))
    print(f"Accuracy for predicted_target: {acc}")
    
    # Split data: 60% train, 20% test, 20% validation
    train_data, temp_data = train_test_split(df_train, test_size=0.4, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    from sklearn.preprocessing import FunctionTransformer
    get_processed_text_ft = FunctionTransformer(lambda x: get_processed_text(x), validate=False)
    get_predicted_target_ft = FunctionTransformer(lambda x: get_predicted_target(x), validate=False)
    
    combined_features = FeatureUnion([
        ('processed_text', Pipeline([
            ('selector', get_processed_text_ft),
            ('tfidf', TfidfVectorizer())
        ])),
        ('predicted_target', Pipeline([
            ('selector', get_predicted_target_ft),
            ('tfidf', TfidfVectorizer())
        ]))
    ])
    
    all_features_df = pd.DataFrame(columns=['Feature', 'Importance', 'Sentiment'])
    
    parameters_dict = {
        'target_sentiment': {'clf__max_depth': None, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 5, 'clf__n_estimators': 150, 'tfidf__max_df': 1.0, 'tfidf__min_df': 2, 'tfidf__ngram_range': (1, 3)},
        'companies_sentiment': {'clf__max_depth': None, 'clf__min_samples_leaf': 2, 'clf__min_samples_split': 5, 'clf__n_estimators': 50, 'tfidf__max_df': 0.5, 'tfidf__min_df': 1, 'tfidf__ngram_range': (1, 2)},
        'consumers_sentiment': {'clf__max_depth': None, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 10, 'clf__n_estimators': 150, 'tfidf__max_df': 0.75, 'tfidf__min_df': 3, 'tfidf__ngram_range': (1, 1)}
    }
    
    for col in ['target_sentiment', 'companies_sentiment', 'consumers_sentiment']:
        parameters = parameters_dict[col]
        tfidf = TfidfVectorizer(max_df=parameters['tfidf__max_df'],
                                min_df=parameters['tfidf__min_df'],
                                ngram_range=parameters['tfidf__ngram_range'])
        clf = RandomForestClassifier(max_depth=parameters['clf__max_depth'],
                                       min_samples_leaf=parameters['clf__min_samples_leaf'],
                                       min_samples_split=parameters['clf__min_samples_split'],
                                       n_estimators=parameters['clf__n_estimators'])
        pipeline = Pipeline([('tfidf', tfidf), ('clf', clf)])
        pipeline.fit(get_processed_text(train_data), train_data[col])
        
        preds = pipeline.predict(get_processed_text(test_data))
        f1_scr = f1_score(test_data[col], preds, average='weighted')
        print(f"F1-score for {col}: {f1_scr}")
        
        final_model = pipeline
        X_combined = pd.concat([get_processed_text(train_data), get_processed_text(test_data)])
        y_combined = pd.concat([train_data[col], test_data[col]])
        final_model.fit(X_combined, y_combined)
        preds_val = final_model.predict(get_processed_text(val_data))
        f1_val = f1_score(val_data[col], preds_val, average='weighted')
        print(f"F1-score for {col} on validation data: {f1_val}")
        
        cm = confusion_matrix(val_data[col], preds_val, labels=['positive', 'neutral', 'negative'])
        plt.title(f'Confusion Matrix for {col}')
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
        
        feature_importances = clf.feature_importances_
        features = tfidf.get_feature_names_out()
        feature_importance_dict = dict(zip(features, feature_importances))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)[:20]
        temp_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
        temp_df['Sentiment'] = col
        all_features_df = pd.concat([all_features_df, temp_df], ignore_index=True)
    
    # Export feature importances to a Word document
    document = Document()
    table = document.add_table(rows=1, cols=len(all_features_df.columns))
    hdr_cells = table.rows[0].cells
    for i, column in enumerate(all_features_df.columns):
        hdr_cells[i].text = str(column)
    for index, row in all_features_df.iterrows():
        row_cells = table.add_row().cells
        for i, value in enumerate(row):
            row_cells[i].text = str(value)
    document.save('important_features.docx')
    
if __name__ == "__main__":
    main()
