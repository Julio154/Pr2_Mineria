import random
from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
        if len(cluster_indices) == 0:
            cluster_labels[cluster] = "Sin datos"
        else:
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
    ('clf', KMeans(n_clusters=4, random_state=200)),
])

text_frecuency = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', KMeans(n_clusters=4, random_state=200)),
])

# Pipeline optima para clustering #
text_tfidf = Pipeline([
    ('vect', CountVectorizer(max_df=0.8, min_df=2, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', KMeans(init='k-means++', max_iter=300, n_clusters=4)),
])

#pipeline para clasificacion
text_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# Pipeline optima para clasificacion #
text_SGDC = Pipeline([
    ('count', CountVectorizer(max_df=0.9, min_df=5, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', SGDClassifier(random_state=42, alpha=0.0001, max_iter=1000, penalty='l2')),
])
text_SGDC = Pipeline([
    ('count', CountVectorizer(max_df=0.9, min_df=5, ngram_range=(1, 2))),
    ('tfidf', TfidfTransformer(norm='l2')),
    ('clf', KNeighborsClassifier()),
])


param_grid_knn = {
    'clf__n_neighbors': [3, 5, 9],
    'clf__weights': ['uniform', 'distance'],
    'clf__metric': ['euclidean', 'manhattan'],
    'clf__p': [1, 2],
    'clf__algorithm': ['auto']
}

grid_search_tfidf = GridSearchCV(
    text_tfidf,
    param_grid_knn,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',  # Adjust scoring if needed
    n_jobs=-1,
    verbose=1
)
grid_search_tfidf.fit(X, y)

print("mejores parametros: ", grid_search_tfidf.best_params_)
print("mejor score: ", grid_search_tfidf.best_score_)

# Ahora, para cada fold:
etiquetas_usadas = {}
accuracies = np.zeros(5)
for i, (tra, tst) in enumerate(skf.split(X, y)):
        
    fit_clustering = True
    fit_classification = True
    #text_tfidf['clf'].random_state = next(iterador)
    
    # Clustering
    if fit_clustering:
        # Entrenamiento
        #text_binary.fit(X[tra])
        #text_frecuency.fit(X[tra])
        text_tfidf.fit(X[tra])
        
        # Test
        #labels1 = text_binary.predict(X[tst])
        #labels2 = text_frecuency.predict(X[tst])
        labels3 = text_tfidf.predict(X[tst])

        folds = "Fold "+str(i)
        etiquetas_usadas[folds] = identificar_clusters(text_tfidf, labels3, y[tst], enc)

        # Calculo de metricas
        #acc = np.mean(labels1 == y[tst])
        #decoded_labels = enc.inverse_transform(y[tst].reshape(-1, 1)).reshape(-1)
        #print(f'Labels: {decoded_labels}')
        #print(f'Binary: {acc}')
        #acc = np.mean(labels2 == y[tst])
        #print(f'Frecuency: {acc}')
        acc = np.mean(labels3 == y[tst])
        print(f'TF-IDF {folds}: {acc}')

        '''
        centroids = text_binary['clf'].cluster_centers_
        iterations = text_binary['clf'].n_iter_

        if iterations < text_binary['clf'].max_iter:
            print('Centroides estables (converged)')
        else:
            print('Centroides inestables (not converged)')
        '''

        '''
        # Preparacion de datos para t-SNE
        transformed = text_frecuency.transform(X[tst])
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(transformed)
        # Visualizacion de los clusteres
        plt.figure(figsize=(10, 6))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels2, cmap='viridis', marker='o')
        plt.title('t-SNE Clustering')
        plt.colorbar()
        plt.show()
        '''

    '''
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
'''



print(etiquetas_usadas)

