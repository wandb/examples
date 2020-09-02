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
wandb.init()

# load data
iris = load_iris()
X = iris.data
y = iris.target
y[y != 0] = 1

# shuffle data
random_idx = permutation(np.arange(len(y)))
X = X[random_idx]
y = y[random_idx]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# create model
model = RandomForestClassifier()
wandb.sklearn.plot_learning_curve(model, X_test, y_test)
wandb.sklearn.plot_class_balance(y_train, y_test)
wandb.sklearn.plot_calibration_curve(X, y, RandomForestClassifier(), name="Random Forest")
wandb.sklearn.plot_decision_boundaries(model, X, y)

'''
# Visualize model performance
# wandb.sklearn.log(rf, X=X, y=y, X_test=X_test, y_test=y_test, labels=feature_names)
'''
