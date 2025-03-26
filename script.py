import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
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


# Trabajar con datos de Texto en scikit-learn: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# Visualizar clusters con t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html


dataset = 'news_reducido.csv'

# Leer los datos en formato csv
data = pd.read_csv(dataset)

# Nos quedamos con el texto (puedes quedarte con más información si quieres)
X = data['text'].astype(str).to_numpy()

# Tokenizamos el texto
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data)
print(X_train_counts.shape)

# Ahora, vamos a convertir las palabras en vectores de frecuencias
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)




# Ahora, procesamos las etiquetas, para cada clase, le asignamos un valor numérico entre 0 y el número de clases
enc = OrdinalEncoder()
y = enc.fit_transform(np.reshape(data['category'], (-1, 1))).reshape(-1)

# Hacemos la partición train-test con Validacion cruzada estratificada
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf.get_n_splits(X, y)



# Definir aquí los pipelines necesarios para cada problema (clustering, clasificación, etc.)
text_binary = Pipeline([
    ('vect', CountVectorizer(binary=True)),
    ('clf', KMeans(n_clusters=4)),
])

text_frecuency = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', KMeans(n_clusters=4)),
])

text_kmeans = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', KMeans(n_clusters=4)),
])

text_sgd = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])


# Ahora, para cada fold:
accuracies1 = np.zeros(5)
accuracies2 = np.zeros(5)
accuracies3 = np.zeros(5)
accuracies4 = np.zeros(5)
for i, (tra, tst) in enumerate(skf.split(X,y)):
        
    fit_clustering = True
    fit_classification = True
    
    # Clustering
    if fit_clustering:
        # Entrenamiento
        text_binary.fit(X[tra])
        text_frecuency.fit(X[tra])
        text_kmeans.fit(X[tra])
        text_sgd.fit(X[tra])
        
        # Test
        labels1 = text_binary.predict(X[tst])
        labels2 = text_frecuency.predict(X[tst])
        labels3 = text_kmeans.predict(X[tst])
        labels4 = text_sgd.predict(X[tst])
        
        # Calculo de metricas
        acc1 = np.mean(labels1 == y[tst])
        acc2 = np.mean(labels2 == y[tst])
        acc3 = np.mean(labels3 == y[tst])
        acc4 = np.mean(labels4 == y[tst])
        print(acc1)
        print(acc2)
        print(acc3)
        print(acc4)

    # Clasificacion
    if fit_classification:
        # Entrenamiento
        text_sgd.fit(X[tra], y[tra])
        text_binary.fit(X[tra], y[tra])
        text_frecuency.fit(X[tra], y[tra])
        text_kmeans.fit(X[tra], y[tra])

        # Test (obtener predicciones)
        predictedsgd = text_sgd.predict(X[tst])
        predictedbinary = text_binary.predict(X[tst])
        predictedfrecuency = text_frecuency.predict(X[tst])
        predictedtfidf = text_kmeans.predict(X[tst])
        
        # Calculo de metricas de calidad (ahora, solo accuracy)
        acc_sgd = np.mean(predictedsgd == y[tst])
        acc_binary = np.mean(predictedbinary == y[tst])
        acc_frecuency = np.mean(predictedfrecuency == y[tst])
        acc_tfidf = np.mean(predictedtfidf == y[tst])

        print(f'Binary: {acc_binary}')
        accuracies1[i] = acc_binary
        print(f'Frecuency: {acc_frecuency}')
        accuracies2[i] = acc_frecuency
        print(f'KMeans: {acc_tfidf}')
        accuracies3[i] = acc_tfidf
        print(f'TF-IDF: {acc_sgd}')
        accuracies4[i] = acc_tfidf
        
# Tras el K-Fold, hay que mostrar la precision media obtenida ( o cualquier otra metrica de interes, pero promediada)
avg_acc1 = np.average(accuracies1)
print(f'Precision media binary = {avg_acc1}')
avg_acc2 = np.average(accuracies2)
print(f'Precision media frecuency = {avg_acc2}')
avg_acc3 = np.average(accuracies3)
print(f'Precision media kmeans = {avg_acc3}')
avg_acc4 = np.average(accuracies4)
print(f'Precision media tfidf = {avg_acc4}')


