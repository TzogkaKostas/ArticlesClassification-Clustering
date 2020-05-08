import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from statistics import stdev
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer, cosine_distance
# from sklearn.metrics.pairwise import cosine_similarity


def multi_classify(clusterer, X_test):
    clusters = []
    for query in X_test.toarray():
        cluster = clusterer.classify(query)
        clusters.append(cluster)
        
    return clusters



# read data
df_train = pd.read_csv("train_set.csv", sep='\t')
df_train['full_data'] = df_train['title'].fillna('') + " " + df_train['content'].fillna('')

df_test = pd.read_csv("test_set.csv", sep='\t')
df_test['full_data'] = df_test['title'].fillna('') + " " + df_test['content'].fillna('')


# vectorization by CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_train_count = count_vectorizer.fit_transform(df_train['full_data'])
X_test_count = count_vectorizer.transform(df_test['full_data'])

# vectorization by TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['full_data'])
X_test_tfidf = tfidf_vectorizer.transform(df_test['full_data'])

# labels
le = preprocessing.LabelEncoder()
le.fit(df_train['category'])
y_train = le.transform(df_train['category'])
y_test = le.transform(df_test['category'])

################################### CountVectorizer ####################################
clusterer = KMeansClusterer(5, distance=cosine_distance)
clusters = clusterer.cluster(X_train_count.toarray(), True)
print(clusters)
# print(clusterer.classify(X_test_count.toarray()))