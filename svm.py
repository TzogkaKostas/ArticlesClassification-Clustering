import pandas as pd
from sklearn import svm, preprocessing, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

df_train = pd.read_csv("train_set.csv", sep='\t')
df_train['full_data'] = df_train['title'].fillna('') + " " + df_train['content'].fillna('')

df_test = pd.read_csv("test_set.csv", sep='\t')
df_test['full_data'] = df_test['title'].fillna('') + " " + df_test['content'].fillna('')


# train data CountVectorizer
count_vectorizer = CountVectorizer()
X_train = count_vectorizer.fit_transform(df_train['full_data'])

# train labels
le = preprocessing.LabelEncoder()
le.fit(df_train['category'])
y_train = le.transform(df_train['category'])

# SVM fit
clf = svm.SVC()
clf.fit(X_train, y_train)

# precision_micro recall_micro f1_micro evaluation
precisions = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision_micro')
print ('Precision', np.mean(precisions), precisions)
recalls = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall_micro')
print ('Recalls', np.mean(recalls), recalls)
f1s = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1_micro')
print ('F1', np.mean(f1s), f1s)

# predict
X_test = count_vectorizer.transform(df_test['full_data'])
y_test = le.transform(df_test['category'])
predictions = clf.predict(X_test)

# accuracy evaluation
print ('Accuracy:', accuracy_score(y_test, predictions))

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




# train data by TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train = tfidf_vectorizer.fit_transform(df_train['full_data'])

# SVM fit
clf = svm.SVC()
clf.fit(X_train, y_train)

# precision_micro recall_micro f1_micro evaluation
precisions = cross_val_score(clf, X_train, y_train, cv=10, scoring='precision_micro')
print ('Precision', np.mean(precisions), precisions)
recalls = cross_val_score(clf, X_train, y_train, cv=10, scoring='recall_micro')
print ('Recalls', np.mean(recalls), recalls)
f1s = cross_val_score(clf, X_train, y_train, cv=10, scoring='f1_micro')
print ('F1', np.mean(f1s), f1s)

# predict
X_test = tfidf_vectorizer.transform(df_test['full_data'])
predictions = clf.predict(X_test)

# accuracy evaluation
print ('Accuracy:', accuracy_score(y_test, predictions))

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
