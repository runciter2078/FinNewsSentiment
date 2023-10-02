import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nltk import FreqDist, word_tokenize
from nltk.corpus import stopwords
import nltk
import unicodedata

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Leer el dataset
df_train = pd.read_csv('FinancES_phase_2_train_public.csv')

# Limpiar columnas de sentimientos
allowed_sentiments = ['negative', 'positive', 'neutral']
for col in ['target_sentiment', 'companies_sentiment', 'consumers_sentiment']:
    df_train = df_train[df_train[col].isin(allowed_sentiments)]

# Funciones de preprocesamiento
stop_words = set(stopwords.words('spanish'))

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

def preprocess(text):
    text = remove_accents(text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Preprocesar el texto y el target
df_train['processed_text'] = df_train['text'].apply(preprocess)
df_train['processed_target'] = df_train['target'].apply(preprocess)

# Función para extraer el token más común
def extract_keywords(text):
    tokens = preprocess(text)
    if not tokens:
        return None
    fdist = FreqDist(tokens)
    most_common = fdist.most_common(1)
    return most_common[0][0] if most_common else None

# Extraer ente económico principal
df_train['predicted_target'] = df_train['text'].apply(extract_keywords)
df_train['processed_target_str'] = df_train['processed_target'].apply(lambda x: ' '.join(x) if x else None)

# Métricas para el ente económico principal
accuracy = accuracy_score(df_train['processed_target_str'].fillna(''), df_train['predicted_target'].fillna(''))
print(f"Accuracy for predicted_target: {accuracy}")

# Dividir los datos: 60% para entrenamiento, 20% para prueba y 20% para validación
train_data, temp_data = train_test_split(df_train, test_size=0.4, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Funciones para seleccionar columnas
def get_processed_text(data):
    return data['processed_text'].apply(' '.join)

def get_predicted_target(data):
    return data['predicted_target'].fillna('')

# Transformers personalizados
get_processed_text_ft = FunctionTransformer(lambda x: get_processed_text(x), validate=False)
get_predicted_target_ft = FunctionTransformer(lambda x: get_predicted_target(x), validate=False)

# Crear FeatureUnion
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

# Inicializar un DataFrame para almacenar las características importantes de todos los clasificadores
all_features_df = pd.DataFrame(columns=['Feature', 'Importance', 'Sentiment'])

# Definir un pipeline y un grid de búsqueda para cada columna de sentimientos
for col in ['target_sentiment', 'companies_sentiment', 'consumers_sentiment']:
    pipeline = Pipeline([
        ('features', combined_features),
        ('clf', RandomForestClassifier())
    ])
    
    # parameters = {
    # 'features__processed_text__tfidf__max_df': [0.5, 0.75, 1.0],
    # 'features__processed_text__tfidf__min_df': [1, 2, 3],
    # 'features__processed_text__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    # 'features__predicted_target__tfidf__max_df': [0.5, 0.75, 1.0],
    # 'features__predicted_target__tfidf__min_df': [1, 2, 3],
    # 'features__predicted_target__tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    # 'clf__n_estimators': [50, 100, 150],
    # 'clf__max_depth': [None, 10, 20, 30],
    # 'clf__min_samples_split': [2, 5, 10],
    # 'clf__min_samples_leaf': [1, 2, 4]}

    # parameters = {
    #     'features__processed_text__tfidf__max_df': [0.5, 1.0],
    #     'features__processed_text__tfidf__min_df': [1, 2],
    #     'features__processed_text__tfidf__ngram_range': [(1, 1), (1, 2)],
    #     'features__predicted_target__tfidf__max_df': [0.5, 1.0],
    #     'features__predicted_target__tfidf__min_df': [1, 2],
    #     'features__predicted_target__tfidf__ngram_range': [(1, 1), (1, 2)],
    #     'clf__n_estimators': [50, 100],
    #     'clf__max_depth': [None, 20],
    #     'clf__min_samples_split': [2, 5],
    #     'clf__min_samples_leaf': [1, 2],
    # }
    
    parameters = {
        'features__processed_text__tfidf__max_df': [0.5],
        'features__processed_text__tfidf__min_df': [1],
        'features__processed_text__tfidf__ngram_range': [(1, 2)],
        'features__predicted_target__tfidf__max_df': [0.5],
        'features__predicted_target__tfidf__min_df': [2],
        'features__predicted_target__tfidf__ngram_range': [(1, 2)],
        'clf__n_estimators': [100],
        'clf__max_depth': [None],
        'clf__min_samples_split': [5],
        'clf__min_samples_leaf': [2],
    }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='f1_weighted')
    grid_search.fit(train_data, train_data[col])
    #print("Best parameters found: ", grid_search.best_params_)
    
    # Métricas F1 para los datos de prueba
    predictions = grid_search.predict(test_data)
    f1 = f1_score(test_data[col], predictions, average='weighted')
    print(f"F1-score for {col}: {f1}")

    # Entrenar con datos de entrenamiento + test
    final_model = grid_search.best_estimator_
    final_model.fit(pd.concat([train_data, test_data]), pd.concat([train_data, test_data])[col])
    
    # Métricas F1 para los datos de validación
    predictions_val = final_model.predict(val_data)
    f1_val = f1_score(val_data[col], predictions_val, average='weighted')
    print(f"F1-score for {col} on validation data: {f1_val}")
        
    # Matriz de confusión
    cm = confusion_matrix(val_data[col], predictions_val, labels=['positive', 'neutral', 'negative'])
    plt.title(f'Confusion Matrix for {col}')
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
    
    # Añadir las importancias de las características del clasificador actual a all_features_df
    feature_importances_df = pd.DataFrame(columns=['Feature', 'Importance', 'Sentiment'])
    
    # Obtén el pipeline final del GridSearchCV
    final_pipeline = grid_search.best_estimator_
    
    # Obtén el clasificador Random Forest del pipeline
    clf = final_pipeline.named_steps['clf']
    
    # Obtén el FeatureUnion del pipeline
    feature_union = final_pipeline.named_steps['features']
    
    # Obtén los nombres de las características de los transformadores dentro del FeatureUnion
    feature_names = []
    for transformer_name, transformer in feature_union.transformer_list:
        if transformer_name == 'processed_text':
            feature_names.extend(transformer.named_steps['tfidf'].get_feature_names_out())
        elif transformer_name == 'predicted_target':
            feature_names.extend(transformer.named_steps['tfidf'].get_feature_names_out())
    
    # Obtén las importancias de las características del Random Forest y añádelas al DataFrame
    importances = clf.feature_importances_
    for i in range(len(importances)):
        temp_df = pd.DataFrame({'Feature': [feature_names[i]], 'Importance': [importances[i]], 'Sentiment': [col]})
        feature_importances_df = pd.concat([feature_importances_df, temp_df], ignore_index=True)
    
    all_features_df = pd.concat([all_features_df, feature_importances_df], ignore_index=True)


    # Procesar y mostrar las características más importantes para cada sentimiento
    grouped_df = all_features_df.groupby('Sentiment').apply(lambda x: x.nlargest(20, 'Importance')).reset_index(drop=True)
    pivot_df = grouped_df.pivot_table(index=grouped_df.groupby('Sentiment').cumcount(), columns='Sentiment', values=['Feature', 'Importance'], aggfunc='first')
    pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
    top_features_df = pivot_df.reset_index(drop=True)
        
#%%
import pandas as pd
from docx import Document

# Suponiendo que important_features_df es tu DataFrame
# important_features_df = pd.DataFrame( ... )

document = Document()

# Añadir el DataFrame fila por fila
table = document.add_table(rows=1, cols=len(top_features_df.columns))
hdr_cells = table.rows[0].cells

# Añadir títulos
for i, column in enumerate(top_features_df.columns):
    hdr_cells[i].text = str(column)

# Añadir datos
for index, row in top_features_df.iterrows():
    row_cells = table.add_row().cells
    for i, value in enumerate(row):
        row_cells[i].text = str(value)

# Guardar el documento
document.save('mi_documento2.docx')

    

#%%
import pandas as pd
from docx import Document

# Suponiendo que important_features_df es tu DataFrame
# important_features_df = pd.DataFrame( ... )

document = Document()

# Añadir el DataFrame fila por fila
table = document.add_table(rows=1, cols=len(top_features_df.columns))
hdr_cells = table.rows[0].cells

# Añadir títulos
for i, column in enumerate(top_features_df.columns):
    hdr_cells[i].text = str(column)

# Añadir datos
for index, row in top_features_df.iterrows():
    row_cells = table.add_row().cells
    for i, value in enumerate(row):
        row_cells[i].text = str(value)

# Guardar el documento
document.save('mi_documento2.docx')
