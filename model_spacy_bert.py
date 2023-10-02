import pandas as pd
import spacy
import re
import unicodedata
from sklearn.metrics import f1_score
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

nlp = spacy.load('es_core_news_sm')

# Cargar el dataframe
df_train = pd.read_csv('FinancES_phase_2_train_public.csv')
df_train = df_train.sample(frac=1, random_state=42)

allowed_sentiments = ['negative', 'positive', 'neutral']
for col in ['target_sentiment', 'companies_sentiment', 'consumers_sentiment']:
    df_train = df_train[df_train[col].isin(allowed_sentiments)]

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess(text):
    text = remove_accents(text.lower())
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens

df_train['processed_text'] = df_train['text'].apply(preprocess)
df_train['processed_target'] = df_train['target'].apply(preprocess)

def extract_keywords(text):
    tokens = preprocess(text)
    if not tokens:
        return None
    most_common = pd.Series(tokens).value_counts().idxmax()
    return most_common

df_train['predicted_target'] = df_train['text'].apply(extract_keywords)
df_train['processed_target_str'] = df_train['processed_target'].apply(lambda x: ' '.join(x) if x else None)

accuracy = accuracy_score(df_train['processed_target_str'].fillna(''), df_train['predicted_target'].fillna(''))
print(f"Accuracy for predicted_target: {accuracy}")

# Cargar el clasificador de sentimiento
classifier = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

def stars_to_sentiment(stars_label):
    # Suponiendo que 1 y 2 estrellas es 'negative', 3 es 'neutral' y 4 y 5 son 'positive'
    stars = int(stars_label[0])
    if stars in [1, 2]:
        return 'negative'
    elif stars == 3:
        return 'neutral'
    elif stars in [4, 5]:
        return 'positive'

def predict_sentiment(processed_text, predicted_target):
    # Convertir None a cadena vacía si es necesario
    processed_text_str = ' '.join(processed_text) if processed_text else ''
    predicted_target_str = predicted_target if predicted_target else ''
    
    combined_text = processed_text_str + ' ' + predicted_target_str
    result = classifier(combined_text)
    if not result:
        return 'neutral'
    stars_label = result[0]['label']
    sentiment = stars_to_sentiment(stars_label)
    return sentiment


df_train['predicted_sentiment'] = df_train.apply(
    lambda row: predict_sentiment(row['processed_text'], row['predicted_target']),
    axis=1)

# Calcular el f1-score
f1 = f1_score(df_train['target_sentiment'], df_train['predicted_sentiment'], average='weighted')
#print(f"F1-Score: {f1:.2f}")
print("F1-Score for target_sentiment: ", f1)

# Matriz de confusión
cm = confusion_matrix(df_train['target_sentiment'], df_train['predicted_sentiment'], labels=['positive', 'neutral', 'negative'])
plt.title('Confusion Matrix for target_sentiment')
plt.imshow(cm, cmap='Oranges', interpolation='nearest')
plt.colorbar()

# Etiquetas para los ejes
categories = ['positive', 'neutral', 'negative']
plt.xticks([0, 1, 2], categories)
plt.yticks([0, 1, 2], categories)

plt.xlabel('Predicted')
plt.ylabel('True')
# Poner los números dentro de las casillas
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='black')
plt.show()



















