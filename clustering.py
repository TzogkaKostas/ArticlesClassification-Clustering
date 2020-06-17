import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from statistics import stdev
import matplotlib.pyplot as plt
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from gensim.models import Word2Vec
from gensim.models import KeyedVectors


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
	for i in range(k):
		indices = get_indices(clusters, i)
		cluster_i = points[get_indices(clusters, i)]
		for row, point in enumerate(cluster_i):
			plt.scatter(point[0], point[1], c=colours[i],
				marker=markers[categories[indices[row]]])

	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title(method_name)

	red_patch = mpatches.Patch(color='red', label='cluster 0')
	blue_patch = mpatches.Patch(color='blue', label='cluster 1')
	green_patch = mpatches.Patch(color='green', label='cluster 2')
	black_patch = mpatches.Patch(color='black', label='cluster 3')
	pink_patch = mpatches.Patch(color='pink', label='cluster 4')

	b_mark = mlines.Line2D([], [], color='gray', marker='$B$',
		markersize=8, label='Business', linestyle='None')
	e_mark = mlines.Line2D([], [], color='gray', marker='$E$',
		markersize=8, label='Entertainment', linestyle='None')
	p_mark = mlines.Line2D([], [], color='gray', marker='$P$',
		markersize=8, label='politics', linestyle='None')
	s_mark = mlines.Line2D([], [], color='gray', marker='$S$',
		markersize=8, label='Sport', linestyle='None')
	t_mark = mlines.Line2D([], [], color='gray', marker='$T$',
		markersize=8, label='Tech', linestyle='None')
	
	plt.legend(handles=[red_patch, blue_patch, green_patch, black_patch,
			pink_patch, b_mark, e_mark, p_mark, s_mark, t_mark])
	plt.show()
	plt.savefig(method_name + '.png')

def get_embeddings(model, words, dimension):
	embeddings = []
	for word in words:
		try:
			embeddings.append(model[word])
		except:
			embeddings.append(np.zeros(dimension))

	return embeddings

def get_vectors_from_w2v(model, docs, dimension):
	vectors = []
	for doc in docs:
		embeddings = np.array(get_embeddings(model, doc, dimension))
		doc_vector = np.empty([0])
		for i in range(dimension):
			doc_vector = np.append(doc_vector, np.mean(embeddings[:, i]))

		vectors.append(doc_vector)
	return vectors

def plot_PCA_data(X_train, clusters, y_train, method_name):
	# PCA dimension reductionality 
	transformer = PCA(random_state=2020)
	pca_data = transformer.fit_transform(X_train)

	# plot PCA data
	plt.figure(num=1, figsize=(10, 10))
	plot_data(clusters, 5, pca_data, y_train, method_name)	

def plot_SVD_data(X_train, clusters, y_train, method_name):
	# SVD dimension reductionality 
	svd = TruncatedSVD(random_state=42)
	svd_data = svd.fit_transform(X_train)

	# plot PCA data
	plt.figure(num=1, figsize=(10, 10))
	plot_data(clusters, 5, svd_data, y_train, method_name)

def plot_ICA_data(X_train, clusters, y_train, method_name):
	# SVD dimension reductionality 
	ica = FastICA(n_components=2, random_state=42)
	ica_data = ica.fit_transform(X_train)

	# plot ICA data
	plot_data(clusters, 5, ica_data, y_train, method_name)

if __name__ == "__main__":
	# read data
	df_train = pd.read_csv("train_set.csv", sep='\t')
	df_train['content'] = df_train['content'].str.replace('\.|\,', '', regex=True)

	# labels
	le = preprocessing.LabelEncoder()
	le.fit(df_train['category'])
	y_train = le.transform(df_train['category'])


	######### CountVectorizer #########
	# vectorization by CountVectorizer
	count_vectorizer = CountVectorizer(stop_words='english')
	X_train_count = count_vectorizer.fit_transform(df_train['content'])

	# clustering
	clusterer = KMeansClusterer(5, distance=cosine_distance)
	clusters = clusterer.cluster(X_train_count.toarray(), True)

	# plot PCA data
	# plot_PCA_data(X_train_count.toarray(), clusters, y_train, 'PCA_count')

	# plot SVD data
	# plot_SVD_data(X_train_count, clusters, y_train, 'SVD_count')

	# plot ICA data
	print("aa")
	plt.figure(num=11, figsize=(10, 10))
	plt.xlim([-0.12, 0.05])
	plt.ylim([-0.05, 0.08])
	plot_ICA_data(X_train_count.toarray(), clusters, y_train, 'ICA_count')

	######### TfidfVectorizer #########
	# vectorization by TfidfVectorizer
	tfidf_vectorizer = TfidfVectorizer(stop_words='english')
	X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['content'])

	# clustering
	clusterer = KMeansClusterer(5, distance=cosine_distance)
	clusters = clusterer.cluster(X_train_tfidf.toarray(), True)

	# plot PCA data
	# plot_PCA_data(X_train_tfidf.toarray(), clusters, y_train, 'PCA_tfidf')

	# plot SVD data
	# plot_SVD_data(X_train_tfidf, clusters, y_train, 'SVD_tfidf')

	# plot ICA data
	print("bb")
	plt.figure(num=22, figsize=(10, 10))
	plt.xlim([-0.08, 0.03])
	plt.ylim([-0.05, 0.08])
	plot_ICA_data(X_train_count.toarray(), clusters, y_train, 'ICA_tfidf')


	######### document-embeddings #########
	model = KeyedVectors.load_word2vec_format('../GoogleNews-vectors-negative300.bin',
			binary=True)  

	# vectorization 
	w2v_vectors = get_vectors_from_w2v(model, df_train['content'], model.vector_size)

	# clustering
	clusterer = KMeansClusterer(5, distance=cosine_distance)
	clusters = clusterer.cluster(w2v_vectors, True)

	# plot PCA data
	# plot_PCA_data(w2v_vectors, clusters, y_train, 'PCA_w2v')

	# plot SVD data
	# plot_SVD_data(w2v_vectors, clusters, y_train, 'SVD_w2v')

	# plot ICA data
	plt.figure(num=33, figsize=(10, 10))
	# plt.xlim([-0.08, 0.06])
	# plt.ylim([-0.05, 0.08])
	plot_ICA_data(w2v_vectors, clusters, y_train, 'ICA_w2v')