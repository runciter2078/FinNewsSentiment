from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

fila = 4595

# Define la función que LIME necesita
def predict_proba_func(texts):
    transformed_texts = tfidf.transform(texts)
    return clf.predict_proba(transformed_texts)

# Inicializa el explainer de LIME
explainer = LimeTextExplainer(class_names=['positive', 'neutral', 'negative'])

# Selecciona una fila específica
selected_row = val_data.loc[fila]
selected_text = ' '.join(selected_row[f'processed_text_{col}'])
print(selected_text)

# Obtiene las explicaciones
#exp = explainer.explain_instance(selected_text, predict_proba_func)
exp = explainer.explain_instance(selected_text, predict_proba_func)

#exp = explainer.explain_instance(selected_text, predict_proba_func, labels=[1])  # Para la clase 1 si es "positive"

# Muestra las explicaciones
exp.show_in_notebook(text=True)
fig = exp.as_pyplot_figure()
plt.show()

# Para revisar las probabilidades de las clases para una instancia específica:
probs = final_model.predict_proba(get_processed_text(val_data.loc[[fila]], col))
print(probs)
print(final_model.named_steps['clf'].classes_)
