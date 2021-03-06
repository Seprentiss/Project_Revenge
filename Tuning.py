# importing libraries
from sklearn import decomposition, datasets
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

def Snippet_146_Ex_2():
    print('**Optimizing hyper-parameters of a Decision Tree model using Grid Search in Python**\n')
    label_encoder = LabelEncoder()

    # Loading wine dataset
    dataset = pd.read_csv("Book3.csv")
    dataset = dataset[dataset["ODK"] == "D"]

    label_encoder.fit(dataset["OFF FORM"])
    le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    dataset["OFF FORM"] = label_encoder.fit_transform(dataset["OFF FORM"])
    dataset = dataset[["DN", "DIST", "OFF FORM", "Margin", "PASS/RUN"]].dropna()
    X = dataset[["DN", "DIST", "OFF FORM", "Margin"]]
    y = dataset["PASS/RUN"]

    # Creating an standardscaler object
    std_slc = StandardScaler()

    # Creating a pca object
    pca = decomposition.PCA()

    # Creating a DecisionTreeClassifier
    dec_tree = tree.DecisionTreeClassifier()

    # Creating a pipeline of three steps. First, standardizing the data.
    # Second, tranforming the data with PCA.
    # Third, training a Decision Tree Classifier on the data.
    pipe = Pipeline(steps=[('std_slc', std_slc),
                           ('pca', pca),
                           ('dec_tree', dec_tree)])

    # Creating lists of parameter for Decision Tree Classifier
    criterion = ['gini', 'entropy']
    max_depth = [2, 4, 6, 8, 10, 12, 14, 16 , 18, 20]

    # Creating a dictionary of all the parameter options
    # Note that we can access the parameters of steps of a pipeline by using '__’
    parameters = dict( dec_tree__criterion=criterion,
                      dec_tree__max_depth=max_depth)

    # Conducting Parameter Optmization With Pipeline
    # Creating a grid search object
    clf_GS = GridSearchCV(pipe, parameters)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
    # Fitting the grid search
    clf_GS.fit(X, y)

    # Viewing The Best Parameters
    print('Best Criterion:', clf_GS.best_estimator_.get_params()['dec_tree__criterion'])
    print('Best max_depth:', clf_GS.best_estimator_.get_params()['dec_tree__max_depth'])
    print('Best Number Of Components:', clf_GS.best_estimator_.get_params()['pca__n_components'])
    print(clf_GS.best_estimator_.get_params()['dec_tree'])

#Snippet_146_Ex_2()

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 20)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 20, num = 20)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4, 6]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, n_jobs = -1)
# Fit the random search model
label_encoder = LabelEncoder()
list=[]
dataset = pd.read_csv("Book3.csv")
dataset = dataset[dataset["ODK"] == "D"]

label_encoder.fit(dataset["OFF FORM"])
le_name_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

dataset["OFF FORM"] = label_encoder.fit_transform(dataset["OFF FORM"])
dataset=dataset[["DIST","YARD LN","OFF FORM","Margin","PASS/RUN"]].dropna()
X=dataset[["DIST","YARD LN","OFF FORM","Margin"]]
y=dataset["PASS/RUN"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)

rf_random.fit(X_train,y_train)
print(rf_random.best_params_)
