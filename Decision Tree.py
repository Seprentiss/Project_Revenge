import pandas as pd
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
size = .20
for i in range(1000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size)
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    list1.append(metrics.accuracy_score(y_test, y_pred))
print(sum(list1) / len(list1))

from scipy.stats import sem
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
# evaluate a model with a given number of repeats
def evaluate_model(X, y, repeats):
    # prepare the cross-validation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=repeats)
    # create model
    model = classifier
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# configurations to test
repeats = range(1, 16)
results = list()
for r in repeats:
    # evaluate using a given number of repeats
    scores = evaluate_model(X, y, r)
    # summarize
    print('>%d mean=%.4f se=%.3f' % (r, np.mean(scores), sem(scores)))
    # store
    results.append(scores)
# plot the results
pyplot.boxplot(results, labels=[str(r) for r in repeats], showmeans=True)
pyplot.show()

evaluate_model(X,y,3)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(classifier,
                   feature_names=X.columns,
                   class_names=str(y.unique()),
                   filled=True)
fig.savefig("decistion_tree.png")
input = [11,le_name_mapping.get('OPEN LEFT'),46, -14]
prediction = classifier.predict([input])
if(prediction[0] == 0):
    print("Run")
else:
    print("Pass")

