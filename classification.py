import pandas as pd
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc
from statistics import stdev
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from itertools import groupby
from operator import itemgetter
from collections import Counter
import scikitplot as skplt
from sklearn.model_selection import cross_val_predict
from itertools import cycle
from sklearn.preprocessing import label_binarize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize 
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet as wn


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


def evaluation(clf, X_train, y_train, y_test, predictions, flag=True):
	# precision_micro recall_micro f1_micro evaluation
	precisions = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision_weighted')
	print ('Precision (mean, stdev): ', np.mean(precisions), stdev(precisions), precisions)
	recalls = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall_weighted')
	print ('Recalls (mean, stdev): ', np.mean(recalls), stdev(recalls), precisions)
	f1s = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1_weighted')
	print ('F1 (mean, stdev): ', np.mean(f1s), stdev(f1s), precisions)

	# accuracy evaluation
	accuracies = cross_val_score(clf, X_train_count, y_train, cv=10, scoring='accuracy')
	print ('Accuracy (mean, stdev): ', np.mean(accuracies), stdev(accuracies), accuracies)

	print ('Accuracy on test data: ', accuracy_score(y_test, predictions))

	# ROC evaluation
	if flag == True:
		roc_evaluation(clf, X_train, y_train, [0, 1, 2, 3, 4])

figure_num = 1
def roc_evaluation(clf, X_train, y_train, labels):
	# Binarize the output
	y_bin = label_binarize(y_train, classes=labels)
	n_classes = y_bin.shape[1]

	y_score = cross_val_predict(clf, X_train, y_train, cv=10 ,method='predict_proba')

	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(n_classes):
		fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])
	colors = cycle(['blue', 'red', 'green'])
	for i, color in zip(range(n_classes), colors):
		plt.plot(fpr[i], tpr[i], color=color,
				label='ROC curve of class {0} (area = {1:0.2f})'
				''.format(i, roc_auc[i]))

	plt.figure(num=figure_num, figsize=(10, 10))
	figure_num += 1
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([-0.05, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic for multi-class data')
	plt.legend(loc="lower right")
	plt.show()
	plt.savefig('roc_plot.png')

def wordnet_pos_code(tag):
	if tag in ['JJ', 'JJR', 'JJS']:
		return wn.ADJ
	elif tag in ['RB', 'RBR', 'RBS']:
		return wn.ADV
	elif tag in ['NN', 'NNS', 'NNP', 'NNPS']:
		return wn.NOUN
	elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
		return wn.VERB
	return wn.NOUN 

def lemmas(data):
	lemmatizer = WordNetLemmatizer() 
	return [lemmatizer.lemmatize(token, pos=wordnet_pos_code(tag)) for token, tag in data]

def lemmatization(data):
	try:
		data = lemmas(data)
	except LookupError:
		nltk.download('punkt')
		nltk.download('wordnet')
		nltk.download('averaged_perceptron_tagger')
		data = lemmas(data)
	return ' '.join(data)

def data_preprocessing(docs):
	new_docs = []
	for doc in docs:
		doc = pos_tag(word_tokenize(doc))
		new_docs.append(lemmatization(doc))

	return new_docs


# read data
df_train = pd.read_csv("train_set.csv", sep='\t')
df_train['content'] = df_train['content'].str.replace('\.|\,', '', regex=True)
df_test = pd.read_csv("test_set.csv", sep='\t')
df_test['content'] = df_test['content'].str.replace('\.|\,', '', regex=True)


# vectorization by CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
X_train_count = count_vectorizer.fit_transform(df_train['content'])
X_test_count = count_vectorizer.transform(df_test['content'])

# vectorization by TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['content'])
X_test_tfidf = tfidf_vectorizer.transform(df_test['content'])

# labels
le = preprocessing.LabelEncoder()
le.fit(df_train['category'])
y_train = le.transform(df_train['category'])
y_test = le.transform(df_test['category'])


#################################### SVM ####################################
######### CountVectorizer #########
# fit
clf = svm.SVC(probability=True)
# clf = GridSearchCV(clf, {'kernel':('linear', 'rbf'), 'C':[1, 10]})

clf.fit(X_train_count, y_train)

# predict
predictions_count = clf.predict(X_test_count)

# print evaluations scores
print("SUPPORT VECTOR MACHINES (Bow):")
evaluation(clf, X_train_count, y_train, y_test, predictions_count, False)

######### beat #########
# vectorization by CountVectorizer
count_vectorizer = CountVectorizer(stop_words='english')
preprocessed_train_data = data_preprocessing(df_train['content'].tolist())
preprocessed_test_data = data_preprocessing(df_test['content'].tolist())
X_train_count = count_vectorizer.fit_transform(preprocessed_train_data)
X_test_count = count_vectorizer.transform(preprocessed_test_data)

# fit
clf = svm.SVC(probability=True)

clf.fit(X_train_count, y_train)

# predict
predictions_count = clf.predict(X_test_count)

# print evaluations scores
print("SUPPORT VECTOR MACHINES (beat):")
evaluation(clf, X_train_count, y_train, y_test, predictions_count, False)

exit()
######### TfidfVectorizer #########
# fit
clf = svm.SVC(probability=True)
clf.fit(X_train_tfidf, y_train)

# predict
predictions_count = clf.predict(X_test_tfidf)

# print evaluations scores
print("SUPPORT VECTOR MACHINES (TfIdf):")
evaluation(clf, X_train_tfidf, y_train, y_test, predictions_count)
print("\n")


#################################### Random Forests ####################################
######### CountVectorizer #########
# fit
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_train_count, y_train)

# predict
predictions_count = clf.predict(X_test_count)

# print evaluations scores
print("RANDOM FORESTS (Counts):")
evaluation(clf, X_train_count, y_train, y_test, predictions_count)

######### TfidfVectorizer #########
# fit
clf = RandomForestClassifier(max_depth=4, random_state=0)
clf.fit(X_train_tfidf, y_train)

# predict
predictions_count = clf.predict(X_test_tfidf)

# print evaluations scores
print("RANDOM FORESTS (TfIdf):")
evaluation(clf, X_train_tfidf, y_train, y_test, predictions_count)
print("\n")


#################################### Naive Bayes ####################################
######### CountVectorizer #########
# fit
clf = GaussianNB()
clf.fit(X_train_count.toarray(), y_train)

# predict
predictions_count = clf.predict(X_test_count.toarray())

# print evaluations scores
print("Naive Bayes (Counts):")
evaluation(clf, X_train_count.toarray(), y_train, y_test, predictions_count)

######### TfidfVectorizer #########
# fit
clf = GaussianNB()
clf.fit(X_train_tfidf.toarray(), y_train)

# predict
predictions_count = clf.predict(X_test_tfidf.toarray())

# print evaluations scores
print("Naive Bayes (TfIdf):")
evaluation(clf, X_train_tfidf.toarray(), y_train, y_test, predictions_count)
print("\n")

################################### KNN ####################################
######### CountVectorizer #########
# fit
clf = KNN_classifier(5)
clf.fit(X_train_count, y_train)

# predict
predictions_count = clf.predict(X_test_count)

# print evaluations scores
print("k-nearest neighbors (Bow):")
evaluation(clf, X_train_count, y_train, y_test, predictions_count, False)

######### TfidfVectorizer #########
# fit
clf = KNN_classifier(5)
clf.fit(X_train_tfidf, y_train)

# predict
predictions_count = clf.predict(X_test_tfidf)

# print evaluations scores
print("k-nearest neighbors (TfIdf):")
evaluation(clf, X_train_tfidf, y_train, y_test, predictions_count, False)
print("\n")