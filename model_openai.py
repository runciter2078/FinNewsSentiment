import openai
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Configurar la clave de API de OpenAI
api_key = "api"
openai.api_key = api_key

# Leer el dataset y tomar una muestra
df_train = pd.read_csv('FinancES_phase_2_train_public.csv')
df_train = df_train.sample(frac=0.05, random_state=42)

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
# Sentimiento Empresas

# Inicializar una lista vacía para almacenar las predicciones de sentimiento
sentiment_predictions = []

# Preparar el prompt de sentimiento
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular respecto al principal objeto económico? Responde con 'positivo', 'neutral' o 'negativo'."""
# Ajustar el prompt de sentimiento
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular respecto al principal objeto económico? Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen) en general (no específicamente a la o las empresas nombradas en el titular)? Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera como afecta a las empresas en general, no específicamente a la empresa nombrada en el titular. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera el sentimiento con respecto a las empresas en general, no específicamente a la empresa nombrada en el titular. Si no hay una relación directa entre el titular y el sentimiento que provoca en las empresas en general, devuelve la respuesta ‘neutral’. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera el sentimiento con respecto a las empresas en general, no específicamente a la empresa nombrada en el titular. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'. Si no hay una relación directa entre el titular y el sentimiento que provoca en las empresas en general, devuelve la respuesta ‘neutral’."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera el sentimiento con respecto a las empresas en general, no específicamente a la empresa nombrada en el titular. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'. Si no hay una relación clara y directa entre el titular y el sentimiento que provoca en las empresas en general, devuelve la respuesta ‘neutral’."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera el sentimiento con respecto a las empresas en general, no específicamente a la empresa nombrada en el titular. Te pongo un ejemplo: el titular “Sharp podría reducir un tercio su plantilla” debería etiquetarse como “neutral”, porque, aunque es evidentemente “negativo” para la empresa Sharp, es un evento neutral para las empresas en general. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'. Si no hay una relación clara entre el titular y el sentimiento que provoca en las empresas en general, devuelve la respuesta ‘neutral’."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera el sentimiento con respecto a las empresas en general, no específicamente a la empresa nombrada en el titular. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'. Si no hay una relación clara entre el titular y el sentimiento que provoca en las empresas en general, devuelve la respuesta ‘neutral’. Te pongo un ejemplo: el titular “Sharp podría reducir un tercio su plantilla” debería etiquetarse como “neutral”, porque, aunque es evidentemente “negativo” para la empresa Sharp, es un evento neutral para las empresas en general. """
sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera el sentimiento con respecto a las empresas en general, no específicamente a la empresa nombrada en el titular, adoptando una postura descriptiva y neutral. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'. Si no hay una relación clara entre el titular y el sentimiento que provoca en las empresas en general, devuelve la respuesta ‘neutral’. Te pongo un ejemplo: el titular “Sharp podría reducir un tercio su plantilla” debería etiquetarse como “neutral”, porque, aunque es evidentemente “negativo” para la empresa Sharp, es un evento neutral para las empresas en general. """

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
df_train['sentiment_prediction_companies'] = sentiment_predictions

# Asegurar que la columna 'target_sentiment' no tenga NaN
df_train['companies_sentiment'] = df_train['companies_sentiment'].fillna('')
df_train['companies_sentiment'] = df_train['companies_sentiment'].str.lower()

# Mapear abreviaciones o versiones cortas a las etiquetas correctas
sentiment_mapping = {
    "pos": "positive",
    "neg": "negative",
    "p": "positive",
    "neutral": "neutral",
    # Añade cualquier otra abreviatura que observes
}

# Aplicar mapeo a las predicciones
df_train['sentiment_prediction_companies'] = df_train['sentiment_prediction_companies'].map(sentiment_mapping).fillna(df_train['sentiment_prediction_companies'])

# Calcular el F1 Score
f1 = f1_score(df_train['companies_sentiment'].str.lower(), df_train['sentiment_prediction_companies'].str.lower(), average='weighted')
print(f"El F1 Score para la predicción de sentimiento es: {f1}")

# Matriz de confusión
cm = confusion_matrix(df_train['companies_sentiment'], df_train['sentiment_prediction_companies'], labels=["positive", "neutral", "negative"])
plt.title(f'Confusion Matrix for companies_sentiment')
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
# Sentimiento Consumidores

# Inicializar una lista vacía para almacenar las predicciones de sentimiento
sentiment_predictions = []

# Preparar el prompt de sentimiento
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular respecto al principal objeto económico? Responde con 'positivo', 'neutral' o 'negativo'."""
# Ajustar el prompt de sentimiento
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular respecto al principal objeto económico? Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen) en general (no específicamente a la o las empresas nombradas en el titular)? Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera como afecta a las empresas en general, no específicamente a la empresa nombrada en el titular. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a las empresas (definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen)? Considera el sentimiento con respecto a las empresas en general, no específicamente a la empresa nombrada en el titular. Si no hay una relación directa entre el titular y el sentimiento que provoca en las empresas en general, devuelve la respuesta ‘neutral’. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'."""
#sentiment_prompt_base = """El titular es: '{}'. El principal objeto económico identificado es: '{}'. ¿Cuál es el sentimiento del titular con respecto a los consumidores (definiendo a los consumidores como los hogares e individuos que consumen lo que producen las empresas)? Considera como afecta a los consumidores en general, no específicamente a las personas nombradas en el titular si las hubiese. Responde con la palabra exacta 'positivo', 'neutral' o 'negativo'. Si no hay una relación directa entre el titular y el sentimiento que provoca en los consumidores en general, devuelve la respuesta ‘neutral’."""
sentiment_prompt_base = """Dado este titular de una noticia financiera : '{}' y su principal objeto económico identificado: '{}' necesito que me digas cuál es el sentimiento del titular con respecto a las empresas, definiendo a las empresas como aquellas entidades que producen los bienes y servicios que otros consumen. Debes contestar con la palabra 'positivo', 'neutral' o 'negativo', ninguna palabra más. Adopta una postura descriptiva y neutral, si no hay una relación clara y directa entre el titular y el sentimiento que provoca en las empresas en general, devuelve el sentimiento ‘neutral’."""


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
df_train['sentiment_prediction_consumers'] = sentiment_predictions

# Asegurar que la columna 'target_sentiment' no tenga NaN
df_train['consumers_sentiment'] = df_train['consumers_sentiment'].fillna('')
df_train['consumers_sentiment'] = df_train['consumers_sentiment'].str.lower()

# Mapear abreviaciones o versiones cortas a las etiquetas correctas
sentiment_mapping = {
    "pos": "positive",
    "neg": "negative",
    "p": "positive",
    "neutral": "neutral",
    # Añade cualquier otra abreviatura que observes
}

# Aplicar mapeo a las predicciones
df_train['sentiment_prediction_consumers'] = df_train['sentiment_prediction_consumers'].map(sentiment_mapping).fillna(df_train['sentiment_prediction_consumers'])

# Calcular el F1 Score
f1 = f1_score(df_train['consumers_sentiment'].str.lower(), df_train['sentiment_prediction_consumers'].str.lower(), average='weighted')
print(f"El F1 Score para la predicción de sentimiento es: {f1}")

# Matriz de confusión
cm = confusion_matrix(df_train['consumers_sentiment'], df_train['sentiment_prediction_consumers'], labels=["positive", "neutral", "negative"])
plt.title(f'Confusion Matrix for consumers_sentiment')
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




















