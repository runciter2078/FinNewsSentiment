import openai
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import unicodedata
import spacy
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import numpy as np

# Configurar la clave de API de OpenAI
api_key = "api"
openai.api_key = api_key

# Leer el dataset y tomar una muestra
df_train = pd.read_csv('FinancES_phase_2_train_public.csv')
df_train = df_train.sample(frac=0.1, random_state=42)

# Inicializar una lista vacía para almacenar las predicciones
predictions = []

# Preparar el prompt base
#prompt_base = """Te voy a dar un titular de una noticia financiera. Necesito que me digas cuál es el principal objeto económico de dicho titular. Para que te sirva como ejemplo, te voy a dar tres titulares y te voy a decir cuál es el principal objeto económico en cada uno de ellos. Ejemplo 1: 'Bayer presenta un ERE para 75 personas en Sant Joan Despí (Barcelona)'. En este primer titular, el principal ente económico es Bayer. Ejemplo 2: 'Banc Sabadell vende su gestora a Amundi con 351M en plusvalías'. En este segundo titular, el principal agente económico es Banc Sabadell. Ejemplo 3: 'Los datos sobre el uso de los ascensores arrojan una caída del 45% en la afluencia a la oficina por ómicron'. En este tercer y último ejemplo, el principal objeto es 'uso de los ascensores'. Ahora debes tú extraer el principal objeto económico de  este titular: '{}'. Basado en los ejemplos previos sobre titulares financieros y sus objetos económicos principales, identifica el principal objeto económico del anterior titular y responde con una única palabra."""
prompt_base = """Contesta con una única palabra o palabra compuesta. Te voy a dar tres ejemplos de titulares de noticias financieras y de cuál es su principal objeto económico. Ejemplo 1: 'Bayer presenta un ERE para 75 personas en Sant Joan Despí (Barcelona)'. En este primer titular, el principal ente económico es 'Bayer'. Ejemplo 2: 'Banc Sabadell vende su gestora a Amundi con 351M en plusvalías'. En este segundo titular, el principal objeto económico es 'Banc Sabadell'. Ejemplo 3: 'Los datos sobre el uso de los ascensores arrojan una caída del 45% en la afluencia a la oficina por ómicron'. En este tercer y último ejemplo, el principal objeto es 'uso de los ascensores'. Ahora debes tú extraer el principal objeto económico de este titular: '{}'. Basado en los ejemplos previos sobre titulares financieros y sus objetos económicos principales, identifica el principal objeto económico del anterior titular y responde siempre con una única palabra o palabra compuesta."""


# Iterar sobre cada fila del DataFrame
for index, row in df_train.iterrows():
    prompt = prompt_base.format(row['text'])
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=10,  # Puedes ajustar este valor
        temperature=0.2,  # Baja temperatura para respuestas más focales
        top_p=0.9, # Ajustar este valor según tus necesidades
    )
    prediction = response.choices[0].text.strip()
    #print(response.choices[0])
    predictions.append(prediction)


# Añadir las predicciones al DataFrame original
df_train['target_prediction'] = predictions

df_train['target'] = df_train['target'].fillna('')
df_train['target_prediction'] = df_train['target_prediction'].fillna('')
df_train['target'] = df_train['target'].str.lower()
df_train['target_prediction'] = df_train['target_prediction'].str.lower()

# Calcular la precisión
accuracy = accuracy_score(df_train['target'], df_train['target_prediction'])
print(f"La precisión es: {accuracy}")

###############################################################################
# Sentimiento POE

# Inicializar una lista vacía para almacenar las predicciones de sentimiento
sentiment_predictions = []

# Preparar el prompt de sentimiento
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular respecto al principal objeto económico? Responde con 'positivo', 'neutral' o 'negativo'."""
# Ajustar el prompt de sentimiento
sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular respecto al principal objeto económico? Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""


# Iterar sobre cada fila del DataFrame
for index, row in df_train.iterrows():
    sentiment_prompt = sentiment_prompt_base.format(row['text'], row['target_prediction'])
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=sentiment_prompt,
        max_tokens=3,  # Ajusta este valor según tus necesidades
        temperature=0.2,  # Baja temperatura para respuestas más focales y determinadas
        top_p=0.9, # Ajustar este valor según tus necesidades
    )
    sentiment_prediction = response.choices[0].text.strip().lower()
    sentiment_predictions.append(sentiment_prediction)

# Añadir las predicciones de sentimiento al DataFrame original
df_train['sentiment_prediction'] = sentiment_predictions

# Asegurar que la columna 'target_sentiment' no tenga NaN
df_train['target_sentiment'] = df_train['target_sentiment'].fillna('')
df_train['target_sentiment'] = df_train['target_sentiment'].str.lower()

# Mapear abreviaciones o versiones cortas a las etiquetas correctas
sentiment_mapping = {
    "pos": "positive",
    "neg": "negative",
    "p": "positive",
    "neutral": "neutral",
    # Añade cualquier otra abreviatura que observes
}

# Aplicar mapeo a las predicciones
df_train['sentiment_prediction'] = df_train['sentiment_prediction'].map(sentiment_mapping).fillna(df_train['sentiment_prediction'])

# Calcular el F1 Score
f1 = f1_score(df_train['target_sentiment'].str.lower(), df_train['sentiment_prediction'].str.lower(), average='weighted')
print(f"El F1 Score para la predicción de sentimiento es: {f1}")


# Matriz de confusión
cm = confusion_matrix(df_train['target_sentiment'], df_train['sentiment_prediction'], labels=["positive", "neutral", "negative"])
plt.title(f'Confusion Matrix for target_sentiment')
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

###############################################################################
# spaCy

nlp = spacy.load('es_core_news_sm')

# Asegúrate de que df_train esté definido en tu código
df = df_train[['text', 'target_prediction', 'companies_sentiment', 'consumers_sentiment']].copy()

allowed_sentiments = ['negative', 'positive', 'neutral']
for col in ['companies_sentiment', 'consumers_sentiment']:
    df = df[df[col].isin(allowed_sentiments)]

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess(text, words_to_remove=[]):
    text = remove_accents(text.lower())
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop and token.text not in words_to_remove]
    return ' '.join(tokens)  # Convirtiendo listas de tokens a strings directamente

df['processed_text'] = df['text'].apply(preprocess)
df['processed_target'] = df['target_prediction'].apply(preprocess)

words_to_remove_dict = {
    'companies_sentiment': ['xxx'],
    'consumers_sentiment': ['millones', 'pib', 'espana', 'gobierno']
}

for col, words_to_remove in words_to_remove_dict.items():
    df[f'processed_text_{col}'] = df['text'].apply(lambda x: preprocess(x, words_to_remove))

X_train, X_test, y_train_companies, y_test_companies, idx_train, idx_test = train_test_split(
    df[['processed_text', 'processed_target']],
    df['companies_sentiment'],
    df.index,
    test_size=0.2,
    random_state=42
)

y_train_consumers = df.loc[idx_train, 'consumers_sentiment'].copy()
y_test_consumers = df.loc[idx_test, 'consumers_sentiment'].copy()

X_train = X_train.dropna(subset=['processed_text', 'processed_target'])
y_train_consumers = y_train_consumers.loc[X_train.index].copy()  # Actualizado después de dropna

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

param_grid = {
    'tfidf__max_df': [0.85, 0.9, 0.95],
    'tfidf__min_df': [2, 3, 5],
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [None, 10, 20, 30],
}

grid_companies = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', cv=5, verbose=2)
grid_consumers = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', cv=5, verbose=2)

grid_companies.fit(X_train['processed_text'] + ' ' + X_train['processed_target'], y_train_companies)

print("Mejores parámetros para grid_companies:")
print(grid_companies.best_params_)

y_pred_companies = grid_companies.predict(X_test['processed_text'] + ' ' + X_test['processed_target'])
print("f1-score companies: ", f1_score(y_test_companies, y_pred_companies, average='weighted'))

cm = confusion_matrix(y_test_companies, y_pred_companies, labels=['positive', 'neutral', 'negative'])
plt.figure(figsize=(10, 7))
plt.title(f'Confusion Matrix for companies_sentiment')
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['positive', 'neutral', 'negative'],
            yticklabels=['positive', 'neutral', 'negative'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

grid_consumers.fit(X_train['processed_text'] + ' ' + X_train['processed_target'], y_train_consumers)

print("Mejores parámetros para grid_consumers:")
print(grid_consumers.best_params_)

y_pred_consumers = grid_consumers.predict(X_test['processed_text'] + ' ' + X_test['processed_target'])
print("f1-score consumers: ", f1_score(y_test_consumers, y_pred_consumers, average='weighted'))

cm_consumers = confusion_matrix(y_test_consumers, y_pred_consumers, labels=['positive', 'neutral', 'negative'])
plt.figure(figsize=(10, 7))
plt.title('Confusion Matrix for consumers_sentiment')
sns.heatmap(cm_consumers, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['positive', 'neutral', 'negative'],
            yticklabels=['positive', 'neutral', 'negative'])
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()




   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





















