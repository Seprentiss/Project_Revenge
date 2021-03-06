import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
label_encoder = LabelEncoder()
list = []

dataset = pd.read_csv("Book3.csv")
dataset = dataset[dataset["ODK"] == "D"]
label_encoder.fit(dataset["OFF FORM"])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
dataset=dataset[["DN","DIST","YARD LN","Margin","OFF FORM"]].dropna()
dataset["OFF FORM"] = label_encoder.fit_transform(dataset["OFF FORM"])
X=dataset[["DN","DIST","YARD LN","Margin"]]
y=dataset["OFF FORM"]

size = .30
for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    list.append(sum((cross_val_score(classifier, X, y, cv=10))/10))
print(sum(list) / len(list))

feature_imp = pd.Series(classifier.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_imp)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier,
                   feature_names=X.columns,
                   class_names=str(y.unique()),
                   filled=True)
fig.savefig("Guess_Formation_tree.png")
input = [1,10,-18, 0]
prediction = classifier.predict([input])
for fomation, val in le_name_mapping.items():
    if prediction[0] == val:
        print(fomation)

