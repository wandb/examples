import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from numpy.random.mtrand import permutation
from sklearn.datasets import load_iris
import wandb

# initialize wandb run
wandb.init()

# load data
iris = load_iris()
X = iris.data
y = iris.target
labels = iris.target_names
features = iris.feature_names

y[y != 0] = 1

# shuffle data
X, y = shuffle(X, y)
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
