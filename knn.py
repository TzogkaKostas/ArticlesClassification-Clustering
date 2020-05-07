import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import groupby
from operator import itemgetter
from collections import Counter
from sklearn.metrics import accuracy_score, roc_curve, auc

class KNN_classifier(BaseEstimator, ClassifierMixin):
	def __init__(self, k=5):
		self.k = k
	
	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train
		return self

	def predict(self, X_test):
		predictions = []

		similarities = cosine_similarity(X_test, self.X_train)
		for (i, row) in enumerate(similarities):
			idx = np.argsort(-row)
			predictions.append(self.majority_voting(idx[:self.k]))

		return predictions
	
	def majority_voting(self, neighbors):
		counts = Counter(iter(y_train[neighbors])).most_common()
		maxcount, mode_items = next(groupby(counts, key=itemgetter(1)), (0, []))
		return list(map(itemgetter(0), mode_items))[0]




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

clf = KNN_classifier(5)
clf.fit(X_train_count, y_train)


res = clf.predict(X_test_count)

# print ('Accuracy (mean, stdev): ', accuracy_score(y_test, res))
print(cross_val_score(clf, X_train_count, y_train, cv=10, scoring='precision_weighted'))

