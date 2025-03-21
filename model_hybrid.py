#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_hybrid.py

Hybrid model combining OpenAI's GPT-3 for extracting the main economic subject (POE)
from financial news headlines with a Random Forest classifier for sentiment analysis.
It operates on a sample of the dataset for demonstration purposes.

Note: Configure your OpenAI API key before running.
"""

import openai
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def load_data(csv_path='FinancES_phase_2_train_public.csv', sample_frac=0.1, random_state=42):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=sample_frac, random_state=random_state)
    return df

def main():
    # Configure OpenAI API key (replace "your_api_key" with your actual key)
    api_key = "your_api_key"
    openai.api_key = api_key
    
    # Load dataset sample
    df_train = load_data()
    
    predictions = []
    
    # Define prompt for extracting main economic subject (POE)
    prompt_base = (
        "Contesta con una única palabra o palabra compuesta. Te voy a dar tres ejemplos de titulares de noticias financieras y de cuál es su principal objeto económico. "
        "Ejemplo 1: 'Bayer presenta un ERE para 75 personas en Sant Joan Despí (Barcelona)'. En este primer titular, el principal ente económico es 'Bayer'. "
        "Ejemplo 2: 'Banc Sabadell vende su gestora a Amundi con 351M en plusvalías'. En este segundo titular, el principal objeto económico es 'Banc Sabadell'. "
        "Ejemplo 3: 'Los datos sobre el uso de los ascensores arrojan una caída del 45% en la afluencia a la oficina por ómicron'. En este tercer ejemplo, el principal objeto es 'uso de los ascensores'. "
        "Ahora debes tú extraer el principal objeto económico de este titular: '{}'. Basado en los ejemplos previos, responde siempre con una única palabra o palabra compuesta."
    )
    
    for index, row in df_train.iterrows():
        prompt = prompt_base.format(row['text'])
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=10,
            temperature=0.2,
            top_p=0.9
        )
        prediction = response.choices[0].text.strip()
        predictions.append(prediction)
    
    df_train['target_prediction'] = predictions
    df_train['target'] = df_train['target'].fillna('').str.lower()
    df_train['target_prediction'] = df_train['target_prediction'].fillna('').str.lower()
    
    accuracy_val = accuracy_score(df_train['target'], df_train['target_prediction'])
    print(f"La precisión es: {accuracy_val}")
    
    ###############################################################################
    # Sentiment analysis for the POE
    sentiment_predictions = []
    sentiment_prompt_base = (
        "El titular es: '{}'. El principal objeto económico identificado es: '{}'. "
        "¿Cuál es el sentimiento del titular respecto al principal objeto económico? "
        "Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."
    )
    
    for index, row in df_train.iterrows():
        sentiment_prompt = sentiment_prompt_base.format(row['text'], row['target_prediction'])
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=sentiment_prompt,
            max_tokens=3,
            temperature=0.2,
            top_p=0.9
        )
        sentiment_prediction = response.choices[0].text.strip().lower()
        sentiment_predictions.append(sentiment_prediction)
    
    df_train['sentiment_prediction'] = sentiment_predictions
    
    # Ensure target_sentiment is in lowercase
    df_train['target_sentiment'] = df_train['target_sentiment'].fillna('').str.lower()
    
    sentiment_mapping = {
        "pos": "positive",
        "neg": "negative",
        "p": "positive",
        "neutral": "neutral",
    }
    df_train['sentiment_prediction'] = df_train['sentiment_prediction'].map(sentiment_mapping).fillna(df_train['sentiment_prediction'])
    
    f1_val = f1_score(df_train['target_sentiment'], df_train['sentiment_prediction'], average='weighted')
    print(f"El F1 Score para la predicción de sentimiento es: {f1_val}")
    
    # Plot confusion matrix
    cm = confusion_matrix(df_train['target_sentiment'], df_train['sentiment_prediction'], labels=["positive", "neutral", "negative"])
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
