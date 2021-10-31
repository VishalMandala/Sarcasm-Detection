#Code for K nearest neighbours

import sys
import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

import time
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = np.array(pd.read_csv(r’C:2.csv’))
print(dataset.head())

features = [’Label’, ’Text’]
data_df = pd.DataFrame(dataset, columns=features)
print(data_df)

X = data_df.Text
y = data_df.Label
y=y.astype(’int’)
print(X.shape)
print(y.shape)

pipeline = Pipeline([(’features’, FeatureUnion([(’ngram_tf_idf’, Pipeline([
    (’counts_ngram’, CountVectorizer(ngram_range=(1,2),analyzer=’char’))])),])),
    (’classifier’, KNeighborsClassifier(n_neighbors=5))])

pipeline.fit(X,y)

from sklearn.model_selection import cross_val_predict 
from sklearn import metrics

prediction = cross_val_predict(pipeline, X, y, cv =10)

print(’accuracy’, metrics.accuracy_score(y, prediction))