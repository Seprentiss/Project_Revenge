import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score


data = pd.read_csv("Book3.csv")
label_encoder = LabelEncoder()


dataset = pd.read_csv("Book3.csv")
dataset = dataset[dataset["ODK"] == "D"]

heavy_pass=['TRIPS RIGHT', 'TIGER', 'TIGER RIGHT', 'EMPTY', '3X1 TRIPS TITE', '5 WIDE', 'TROJAN RIGHT', 'TRIPS SPLIT', 'LEDGE SPREAD', 'LEDGE', 'LOAD LEFT', 'BUNCH SPREAD', 'LOAD']
heavy_run= ['SLOTS', 'THUNDER', 'LIGHTNING', 'OPEN RIGHT', 'WISHBONE', 'ROAR', 'OPEN LEFT', 'HEAVY', 'SINGLE WING', 'LION', 'OVER RIGHT', 'LIGHTNING OVER', '2X2 SLOTS TE', 'Pro', '2X2 TWINS TITE SPLIT', '2X1 PRO W WING', 'SLOTS RIGHT', 'HEAVY RIGHT', 'TWINS LEFT WING RIGHT', 'PRO RIGHT WING LEFT', 'DOUBLES RIGHT', 'TWINS LEFT SPLIT', 'TWINS RIGHT SPLIT', 'ACE', '4 WIDE RIGHT', 'LIGHTNING TIGHT', 'TWINS SPLIT', 'BUNCH', 'TIGER LEFT', 'COMET', 'LOAD RIGHT', 'BEAST', '4 WIDE STACK', 'TOUCHDOWN SPLIT', 'POWER I']
balanced=['DOUBLES', '4 WIDE', 'DOUBLE WING LEFT', 'TRIPS LEFT','3X1 TRIPS SPLIT', '2X2 4 WIDE', 'TROJAN LEFT', 'TRIPS', 'LEDGE TIGHT', 'TWINS']


label_encoder.fit(dataset["OFF FORM"])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
dataset["OFF FORM"] = label_encoder.fit_transform(dataset["OFF FORM"])
dataset=dataset[["DIST","YARD LN","OFF FORM","Margin","PASS/RUN"]].dropna()
X=dataset[["DIST","YARD LN","OFF FORM","Margin"]]
y=dataset["PASS/RUN"]

acc=[]
pre=[]
rec=[]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,)
logreg = LogisticRegression()
    # fit the model with data
logreg.fit(X_train, y_train)
y_pred=logreg.predict(X_test)
acc.append(metrics.accuracy_score(y_test, y_pred))
pre.append(metrics.precision_score(y_test, y_pred))
rec.append(metrics.recall_score(y_test, y_pred))


print("Accuracy:",str(sum(acc)/len(acc)*100) + chr(37))
print("Precision:", str(sum(pre)/len(pre) *100) + chr(37))
print("Recall:", str(sum(rec)/len(rec) *100) + chr(37))



# compare the number of repeats for repeated k-fold cross-validation
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
    # evaluate model
    scores = cross_val_score(logreg, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
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


