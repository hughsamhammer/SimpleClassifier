import numpy as np
import pandas as pd
from scipy.sparse import hstack
import os
from datetime import datetime
from copy import deepcopy
import xgboost
from functions.stopwords import get_stop_words
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score,precision_score,f1_score,balanced_accuracy_score
import csv
from numpy.random import Generator, PCG64
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


### KONTAKT GrUNDE
os.chdir(r"""Z:\\Projekte\\Melitta\\KontaktgrundVorhersage\\""")
data_file = r"""KontaktgrundTrainingsDaten03112021smartkitt.csv"""
num_labels=50
#os.chdir(r"""Z:\\Projekte\\Melitta\\EmailTemplates\\python_check""")
#data_file = r"""EmailTemplatePredictionSwaggerTrainingData03112021pythoncheck.csv"""
df = pd.read_csv(data_file ,sep=';', header=0 ,encoding='utf-8')
print("shape: ", df.shape)
print("columns: ",df.columns)
#df['text'] = df['Title'] + " " + df['Description']
df['text'] = df['body']
df.reset_index(inplace=True, drop=True)
print(df.label.value_counts(dropna=False))
df.loc[df.label >= num_labels-1,'label'] = num_labels-1
print("max: ",max(df.label))
print(df.label.value_counts(dropna=False))
print(df.label.value_counts(normalize=True,dropna=False))
#df['text'] = df.body
print("shape2:", df.shape)
df.dropna(subset=['text'] , inplace=True)
print("shape2:", df.shape)
df.text = df.text.str.lower()
df['text'] = df['text'].str.replace(r"[\;\,\.]", " ", regex=True)
df['text'] = df['text'].str.replace(r"[\n]", " ", regex=True)
df['text'] = df['text'].str.replace(r"[\r]", " ", regex=True)
df['text'] = df['text'].str.replace(r"\s{2,20}", " ", regex=True)
print(df.text.at[2])
print(df.text.at[22])
print(df.text.at[222])
print(df.text.at[2222])

#comb_cats_by_freq_desc = pd.value_counts(df.EMailTemplateName).index
#mapping = dict(zip(comb_cats_by_freq_desc, np.arange(len(comb_cats_by_freq_desc))))
#mapping = {x: min(max_group_number, y) for (x, y) in self.mapping.items()}
#df['label'] = df.EMailTemplateName.map(mapping)
data_train, data_test = train_test_split(df, train_size = 0.75)
vectorizer = CountVectorizer(max_df = 0.95 , min_df = 5, analyzer='word' ,max_features=7000,ngram_range=(1,1),encoding='utf-8',lowercase=True)
trVect = vectorizer.fit(data_train.text)
dtm_train = trVect.transform(data_train.text)
dtm_test = trVect.transform(data_test.text)
print("dtm_train shape: ", dtm_train.shape)
print("dtm_test  shape: ", dtm_test.shape)
train_matrix = xgboost.DMatrix(dtm_train, label=data_train.label)
xgboost_params= {'objective': 'multi:softprob','max_depth': 6,'num_class': num_labels, 'eta': 0.05}
trModel = xgboost.train(xgboost_params, train_matrix, num_boost_round = 100,verbose_eval=True)
test_matrix = xgboost.DMatrix(dtm_test, label=data_test.label)
predictions = trModel.predict(test_matrix)
#acc1 = accuracy_score(data_test.label, predictions)
#print("accuracy: ", acc1)
p1 = predictions.argsort()[:, -1]
p2 = predictions.argsort()[:, -2]
p3 = predictions.argsort()[:, -3]
p4 = predictions.argsort()[:, -4]
p5 = predictions.argsort()[:, -5]
p6 = predictions.argsort()[:, -6]
acc1 = accuracy_score(data_test.label, p1)
acc2 = accuracy_score(data_test.label, p2)
acc3 = accuracy_score(data_test.label, p3)
acc4 = accuracy_score(data_test.label, p4)
acc5 = accuracy_score(data_test.label, p5)
print("accuracy: ", acc1)
print("accuracy: ", acc2)
print("accuracy: ", acc3)
print("accuracy: ", acc4)
print("accuracy: ", acc5)
print("accuracy: ", acc1+acc2+acc3)
print("accuracy: ", acc1+acc2+acc3+acc4+acc5)
table = df['label'].value_counts(dropna=False)
coverage = sum(table[table.index != num_labels-1]) / sum(table)
print("coverage: ", coverage)
