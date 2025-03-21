#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
explaning_lime.py

This script uses LIME to explain the predictions of a sentiment analysis model.
It loads a pre-trained pipeline and a validation dataset, selects a sample instance,
and generates an explanation for the predicted sentiment.

Requirements:
  - A CSV file 'validation_data.csv' with a column 'processed_text'
  - A saved model pipeline 'model_pipeline.pkl' that contains a TfidfVectorizer (named 'tfidf')
    and a classifier (named 'clf')

Note: Adjust file names and column names as needed.
"""

import joblib
import pandas as pd
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

def predict_proba_func(texts, model_pipeline):
    """Return predicted probabilities for a list of texts using the model pipeline."""
    return model_pipeline.predict_proba(texts)

def main():
    # Load validation data and model pipeline
    try:
        val_data = pd.read_csv('validation_data.csv')
    except Exception as e:
        print("Error loading validation_data.csv:", e)
        return

    try:
        model_pipeline = joblib.load('model_pipeline.pkl')
    except Exception as e:
        print("Error loading model_pipeline.pkl:", e)
        return

    # Set sample index and processed text column name
    fila = 4595
    col = 'processed_text'
    
    if fila >= len(val_data):
        print(f"Index {fila} is out of range. Using the last row instead.")
        fila = len(val_data) - 1

    selected_row = val_data.iloc[fila]
    # If the column contains a list, join its elements; otherwise, convert to string
    if isinstance(selected_row[col], list):
        selected_text = ' '.join(selected_row[col])
    else:
        selected_text = str(selected_row[col])
    
    print("Selected text:")
    print(selected_text)
    
    # Initialize LimeTextExplainer with class names
    explainer = LimeTextExplainer(class_names=['positive', 'neutral', 'negative'])
    
    # Generate explanation using a lambda to pass the model pipeline to predict_proba_func
    exp = explainer.explain_instance(selected_text, lambda texts: predict_proba_func(texts, model_pipeline))
    
    # Display explanation in notebook (if applicable) and as a matplotlib figure
    exp.show_in_notebook(text=True)
    fig = exp.as_pyplot_figure()
    plt.show()
    
    # Additionally, print predicted probabilities and classifier classes
    probs = model_pipeline.predict_proba([selected_text])
    print("Predicted probabilities:", probs)
    print("Classifier classes:", model_pipeline.named_steps['clf'].classes_)

if __name__ == "__main__":
    main()
