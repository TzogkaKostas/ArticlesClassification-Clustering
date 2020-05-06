import pandas as pd
from sklearn import svm, preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from statistics import stdev
import matplotlib.pyplot as plt


def evaluation(clf, X_train, y_train, y_test, predictions):
    # precision_micro recall_micro f1_micro evaluation
    precisions = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision_micro')
    print ('Precision (mean, stdev): ', np.mean(precisions), stdev(precisions))
    recalls = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall_micro')
    print ('Recalls (mean, stdev): ', np.mean(recalls), stdev(recalls))
    f1s = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1_micro')
    print ('F1 (mean, stdev): ', np.mean(f1s), stdev(f1s))

    # accuracy evaluation
    print ('Accuracy (mean, stdev): ', accuracy_score(y_test, predictions))

    # ROC evaluation
    # false_positive_rate, recall, thresholds = roc_curve(y_test, predictions)
    # roc_auc = auc(false_positive_rate, recall)
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.ylabel('Recall')
    # plt.xlabel('Fall-out')
    # plt.show()

# read data
df_train = pd.read_csv("train_set.csv", sep='\t')
df_train['full_data'] = df_train['title'].fillna('') + " " + df_train['content'].fillna('')

df_test = pd.read_csv("test_set.csv", sep='\t')
df_test['full_data'] = df_test['title'].fillna('') + " " + df_test['content'].fillna('')


# vectorization by CountVectorizer
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(df_train['full_data'])
X_test_count = count_vectorizer.transform(df_test['full_data'])

# vectorization by TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['full_data'])
X_test_tfidf = tfidf_vectorizer.transform(df_test['full_data'])

# labels
le = preprocessing.LabelEncoder()
le.fit(df_train['category'])
y_train = le.transform(df_train['category'])
y_test = le.transform(df_test['category'])


# #################################### SVM ####################################
# ######### CountVectorizer #########
# # fit
# clf = svm.SVC()
# clf.fit(X_train_count, y_train)

# # predict
# predictions_count = clf.predict(X_test_count)

# # print evaluations scores
# print("SUPPORT VECTOR MACHINES (Bow):")
# evaluation(clf, X_train_count, y_train, y_test, predictions_count)

# ######### TfidfVectorizer #########
# # fit
# clf = svm.SVC()
# clf.fit(X_train_tfidf, y_train)

# # predict
# predictions_count = clf.predict(X_test_tfidf)

# # print evaluations scores
# print("SUPPORT VECTOR MACHINES (TfIdf):")
# evaluation(clf, X_train_tfidf, y_train, y_test, predictions_count)
# print("\n")


# #################################### Random Forests ####################################
# ######### CountVectorizer #########
# # fit
# clf = RandomForestClassifier(max_depth=4, random_state=0)
# clf.fit(X_train_count, y_train)

# # predict
# predictions_count = clf.predict(X_test_count)

# # print evaluations scores
# print("RANDOM FORESTS (Counts):")
# evaluation(clf, X_train_count, y_train, y_test, predictions_count)

# ######### TfidfVectorizer #########
# # fit
# clf = RandomForestClassifier(max_depth=4, random_state=0)
# clf.fit(X_train_tfidf, y_train)

# # predict
# predictions_count = clf.predict(X_test_tfidf)

# # print evaluations scores
# print("RANDOM FORESTS (TfIdf):")
# evaluation(clf, X_train_tfidf, y_train, y_test, predictions_count)
# print("\n")

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
