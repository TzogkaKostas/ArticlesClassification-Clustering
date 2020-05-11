import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd
 
def read_title_from_file(file_name):
	with open(file_name, 'rb') as f:
		return f.readline()

def read_content_from_file(file_name):
	with open(file_name, 'rb') as f:
		next(f)
		return f.read()

def create_dataframe():
	rootDir = '../fulltext/data'
	df = pd.DataFrame(columns=['id', 'title', 'content', 'category'])
	id = 1
	for dirName, subdirList, fileList in os.walk(rootDir):
		category = os.path.basename(dirName)
		for fname in fileList:
			file_name = dirName + "/" + fname
			title = read_title_from_file(file_name).decode('utf-8').rstrip('\n')
			content = read_content_from_file(file_name).decode('ISO-8859-1').replace('\n', ' ')
			df = df.append([{'id':id,  'title':title, 'content':content, 'category':category}],
				ignore_index=True)
			id += 1
	return df


df = create_dataframe()

X_train, X_test, y_train, y_test = train_test_split(df, df['category'], test_size=0.2,
	random_state=42, stratify=df['category'])

with open("data.csv", 'w+') as file:
	file.write(df.to_csv(index=False, sep='\t'))

with open("train_set.csv", 'w+') as file:
	file.write(X_train.to_csv(index=False, sep='\t'))

with open("test_set.csv", 'w+') as file:
	file.write(X_test.to_csv(index=False, sep='\t'))
