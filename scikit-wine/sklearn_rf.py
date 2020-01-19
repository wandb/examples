from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
rf = RandomForestClassifier()
rf.fit(X, y)
y_pred = rf.predict(X_test)
y_probas = rf.predict_proba(X_test)

# Visualize model performance
wandb.sklearn.plot_learning_curve(rf, X_test, y_test)
wandb.sklearn.plot_class_balance(y_train, y_test)
wandb.sklearn.plot_calibration_curve(X, y, RandomForestClassifier(), name="Random Forest")
# wandb.sklearn.log(rf, X=X, y=y, X_test=X_test, y_test=y_test, labels=feature_names)
