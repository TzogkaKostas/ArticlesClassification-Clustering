import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from statistics import stdev
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import PCA
from matplotlib.pyplot import figure


def multi_classify(clusterer, X_test):
	clusters = []
	for query in X_test.toarray():
		cluster = clusterer.classify(query)
		clusters.append(cluster)

	return clusters

def get_indices(array, value):
	return [i for i, x in enumerate(array) if x == value]

def plot_data(clusters, k, points, categories, method_name):
	colours = ['red', 'blue', 'green', 'black', 'pink']
	markers = ['$B$', '$E$', '$P$', '$S$', '$T$']
	figure(figsize=(10, 10))
	for i in range(k):
		indices = get_indices(clusters, i)
		cluster_i = points[get_indices(clusters, i)]
		for row, point in enumerate(cluster_i):
			plt.scatter(point[0], point[1], c=colours[i],
				marker=markers[categories[indices[row]]])

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title(method_name)
	plt.legend()
	plt.show()
	plt.savefig(method_name + '.png')



# read data
df_train = pd.read_csv("train_set.csv", sep='\t')


# vectorization by CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_train_count = count_vectorizer.fit_transform(df_train['content'])

# vectorization by TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['content'])

# labels
le = preprocessing.LabelEncoder()
le.fit(df_train['category'])
y_train = le.transform(df_train['category'])


######### CountVectorizer #########
# clustering
clusterer = KMeansClusterer(5, distance=cosine_distance)
clusters = clusterer.cluster(X_train_count.toarray(), True)

# dimension reductionality
transformer = PCA(random_state=2020)
pca_data = transformer.fit_transform(X_train_count.toarray())

# plot data
plot_data(clusters, 5, pca_data, y_train, 'pca_tfidf')


######### TfidfVectorizer #########
# clustering
clusterer = KMeansClusterer(5, distance=cosine_distance)
clusters = clusterer.cluster(X_train_tfidf.toarray(), True)

# dimension reductionality
transformer = PCA(random_state=2020)
pca_data = transformer.fit_transform(X_train_tfidf.toarray())

# plot data
plt.xlim([-6, 20])
plt.ylim([-20, 20])
plot_data(clusters, 5, pca_data, y_train, 'pca_count')



# print(clusterer.classify(X_test_count.toarray()))
