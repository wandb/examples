from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
import wandb
wandb.init()

# Load data
wine_quality = pd.read_csv("wine.csv")
y = wine_quality["quality"]
X = wine_quality.drop(["quality"], axis = 1)
feature_names=wine_quality.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model, get predictions
clf = LinearSVC(max_iter=10000)
clf.fit(X, y)
y_pred = clf.predict(X_test)
y_probas = clf.predict_proba(X_test)

# Visualize model performance
wandb.sklearn.plot_class_balance(y_train, y_test)
wandb.sklearn.plot_calibration_curve(X, y, LinearSVC(max_iter=10000), name="SVC")
wandb.sklearn.plot_decision_boundaries(model, X, y)
# wandb.sklearn.log(rf, X=X, y=y, X_test=X_test, y_test=y_test, labels=feature_names)
