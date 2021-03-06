import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree, metrics
import matplotlib.pyplot as plt
import numpy as np
label_encoder = LabelEncoder()
dataset = pd.read_csv("Book3.csv")

dataset = dataset[dataset["ODK"] == "D"]
label_encoder.fit(dataset["OFF FORM"])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

dataset["OFF FORM"] = label_encoder.fit_transform(dataset["OFF FORM"])
dataset=dataset[["DIST","YARD LN","OFF FORM","Margin","PLAY #","PASS/RUN"]].dropna()
X=dataset[["DIST","YARD LN","OFF FORM","Margin","PLAY #"]]
y=dataset["PASS/RUN"]

acc=[]
pre=[]
rec=[]
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    acc.append(metrics.accuracy_score(y_test, y_pred))
    pre.append(metrics.precision_score(y_test, y_pred))
    rec.append(metrics.recall_score(y_test, y_pred))

print("Accuracy:",str(sum(acc)/len(acc)*100) + chr(37))
print("Precision:", str(sum(pre)/len(pre) *100) + chr(37))
print("Recall:", str(sum(rec)/len(rec) *100) + chr(37))

feature_imp = pd.Series(clf.feature_importances_,index=X.columns).sort_values(ascending=False)
print(feature_imp)

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
    model = clf
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
_ = tree.plot_tree(clf.estimators_[0],
                   feature_names=X.columns,
                   class_names=str(y.unique()),
                   filled=True)
fig.savefig("Random_Forest_tree.png")