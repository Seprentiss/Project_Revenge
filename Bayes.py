# load the iris dataset
from sklearn.datasets import load_iris

import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import tree
from collections import defaultdict
label_encoder = LabelEncoder()
list1 = []

dataset = pd.read_csv("Book3.csv")
dataset = dataset[dataset["ODK"] == "D"]

label_encoder.fit(dataset["OFF FORM"])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

dataset["OFF FORM"] = label_encoder.fit_transform(dataset["OFF FORM"])
dataset=dataset[["DIST","YARD LN","OFF FORM","Margin","PASS/RUN","PLAY #"]].dropna()
X=dataset[["DIST","YARD LN","OFF FORM","Margin","PLAY #"]]
y=dataset["PASS/RUN"]

# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# training the model on training set
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = gnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

cv = RepeatedKFold(n_splits=10, n_repeats=10)
# create model
model = gnb
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Categorical Naive Bayes model accuracy(in %):", np.mean(scores))

from sklearn.naive_bayes import CategoricalNB, ComplementNB,BernoulliNB

catnb = BernoulliNB()
catnb.fit(X_train, y_train)

# making predictions on the testing set
y_pred = catnb.predict(X_test)

# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics

cv = RepeatedKFold(n_splits=10, n_repeats=10)
# create model
model = catnb
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print("Categorical Naive Bayes model accuracy(in %):", np.mean(scores))