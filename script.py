import random
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

## funcion para identificar las etiquetas de los clusteres que se crean.
# Sigue en proceso, falla en el most_common_label_num por un out of range
def identificar_clusters(text, labels, ye, encod):
    cluster_labels = {}
    for cluster in range(text['clf'].n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        cluster_true_labels = ye[cluster_indices]
        most_common_label_num = Counter(cluster_true_labels).most_common(1)[0][0]
        most_common_label_str = encod.inverse_transform([[most_common_label_num]])[0][0]
        cluster_labels[cluster] = most_common_label_str

    return cluster_labels


# Trabajar con datos de Texto en scikit-learn: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# Visualizar clusters con t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html


dataset = 'news_reducido.csv'

# Leer los datos en formato csv
data = pd.read_csv(dataset)

# Nos quedamos con el texto (puedes quedarte con más información si quieres)
X = data['text'].astype(str).to_numpy()

# Ahora, procesamos las etiquetas, para cada clase, le asignamos un valor numérico entre 0 y el número de clases
enc = OrdinalEncoder()
y = enc.fit_transform(np.reshape(data['category'], (-1, 1))).reshape(-1)

# Hacemos la partición train-test con Validacion cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(X, y)


random_states = [0, 42, 100, 200, 300]
iterador = iter(random_states)

# Definir aquí los pipelines necesarios para cada problema (clustering, clasificación, etc.)
#pipeline para clustering
text_binary = Pipeline([
    ('vect', CountVectorizer(binary=True)),
    ('clf', KMeans(n_clusters=4, random_state=random.choice(random_states))),
])

text_frecuency = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', KMeans(n_clusters=4, random_state=random.choice(random_states))),
])

text_tfidf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', KMeans(n_clusters=4, random_state=random.choice(random_states))),
])

#pipeline para clasificacion
text_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Ahora, para cada fold:
etiquetas_usadas = {}
accuracies = np.zeros(5)
for i, (tra, tst) in enumerate(skf.split(X,y)):
        
    fit_clustering = True
    fit_classification = True
    text_tfidf['clf'].random_state = next(iterador)
    
    # Clustering
    if fit_clustering:
        # Entrenamiento
        text_binary.fit(X[tra])
        text_frecuency.fit(X[tra])
        text_tfidf.fit(X[tra])
        
        # Test
        labels1 = text_binary.predict(X[tst])
        labels2 = text_frecuency.predict(X[tst])
        labels3 = text_tfidf.predict(X[tst])

        print(identificar_clusters(text_tfidf, labels3, ye=y[tst], encod=enc))
        folds = "Fold "+str(i)
        etiquetas_usadas[folds] = identificar_clusters(text_tfidf, labels3, y[tst], enc)

        # Calculo de metricas
        acc = np.mean(labels1 == y[tst])
        #decoded_labels = enc.inverse_transform(y[tst].reshape(-1, 1)).reshape(-1)
        #print(f'Labels: {decoded_labels}')
        print(f'Binary: {acc}')
        acc = np.mean(labels2 == y[tst])
        print(f'Frecuency: {acc}')
        acc = np.mean(labels3 == y[tst])
        print(f'TF-IDF: {acc}')


    # Clasificacion
    if fit_classification:
        # Entrenamiento
        text_sgd.fit(X[tra], y[tra])

        # Test (obtener predicciones)
        predictedsgd = text_sgd.predict(X[tst])
        
        # Calculo de metricas de calidad (ahora, solo accuracy)
        acc_sgd = np.mean(predictedsgd == y[tst])

        print(f'Clasificacion: {acc_sgd}')
        accuracies[i] = acc_sgd
        
# Tras el K-Fold, hay que mostrar la precision media obtenida ( o cualquier otra metrica de interes, pero promediada)
avg_acc = np.average(accuracies)
print(f'Precision media = {avg_acc}')

print(etiquetas_usadas)

