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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from gensim.models import Word2Vec


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

def get_vectors_from_w2v(model, docs):
	vectors = []
	for doc in docs:
		vec = []
		for word in doc.split():
			mean_value = np.mean(model[word])
			vec.append(mean_value)
		vectors.append(vec)

	return vectors



# read data
df_train = pd.read_csv("train_set.csv", sep='\t')
df_train['content'] = df_train['content'].str.replace('\.|\,', '', regex=True)

# labels
le = preprocessing.LabelEncoder()
le.fit(df_train['category'])
y_train = le.transform(df_train['category'])

######### document-embeddings #########
tokenized_tweet = df_train['content'].apply(lambda x: x.split()) # tokenizing
model_w2v = Word2Vec(tokenized_tweet,
	size=200, # desired no. of features/independent variables
	window=5, # context window size
	min_count=2,
	sg = 1, # 1 for skip-gram model
	hs = 0,
	negative = 10, # for negative sampling
	workers= 2, # no.of cores
	seed = 34)

model_w2v.train(tokenized_tweet, total_examples= df_train['content'].shape[0], epochs=20)

# vectorization 
vectors = get_vectors_from_w2v(model_w2v, df_train['content'])

# clustering
clusterer = KMeansClusterer(5, distance=cosine_distance)
clusters = clusterer.cluster(vectors, True)

# dimension reductionality
transformer = PCA(random_state=2020)
pca_data = transformer.fit_transform(vectors)

# plot data
plt.figure(num=3, figsize=(10, 10))
plot_data(clusters, 5, pca_data, y_train, 'w2v')



exit()


######### CountVectorizer #########
# vectorization by CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_train_count = count_vectorizer.fit_transform(df_train['content'])

# clustering
clusterer = KMeansClusterer(5, distance=cosine_distance)
clusters = clusterer.cluster(X_train_count.toarray(), True)

# dimension reductionality
transformer = PCA(random_state=2020)
pca_data = transformer.fit_transform(X_train_count.toarray())

# plot data
plt.figure(num=1, figsize=(10, 10))
plt.xlim([-6, 20])
plt.ylim([-20, 20])
plot_data(clusters, 5, pca_data, y_train, 'pca_count')


######### TfidfVectorizer #########
# vectorization by TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['content'])

# clustering
clusterer = KMeansClusterer(5, distance=cosine_distance)
clusters = clusterer.cluster(X_train_tfidf.toarray(), True)

# dimension reductionality
transformer = PCA(random_state=2020)
pca_data = transformer.fit_transform(X_train_tfidf.toarray())

# plot data
plt.figure(num=2, figsize=(10, 10))
plot_data(clusters, 5, pca_data, y_train, 'pca_tfidf')


# print(clusterer.classify(X_test_count.toarray()))