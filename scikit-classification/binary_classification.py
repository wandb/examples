import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random.mtrand import permutation
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import wandb
wandb.init(project="sklearn")

# load data
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
features = iris.feature_names

y[y != 0] = 1

# shuffle data
random_idx = permutation(np.arange(len(y)))
X = X[random_idx]
y = y[random_idx]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# create model
model = RandomForestClassifier()
model.fit(X, y)
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)

# visualize model
wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test, y_pred, y_probas, labels, True, 'RandomForest', features)
wandb.log({'roc': wandb.plots.ROC(y_test, y_probas, labels)})
wandb.log({'pr': wandb.plots.precision_recall(y_test, y_probas, labels)})

'''
wandb.sklearn.plot_learning_curve(model, X_test, y_test)
wandb.sklearn.plot_class_balance(y_train, y_test)
wandb.sklearn.plot_calibration_curve(RandomForestClassifier(), X, y, "Random Forest")
wandb.sklearn.plot_decision_boundaries(model, X, y)
wandb.sklearn.plot_feature_importances(model)
wandb.sklearn.plot_summary_metrics(model, X=X, y=y, X_test=X_test, y_test=y_test)
'''
