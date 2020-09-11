import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import wandb

# initialize wandb run
wandb.init()

# Load data
wine_quality = pd.read_csv("wine.csv")
y = wine_quality["quality"]
X = wine_quality.drop(["quality"], axis=1)
feature_names = wine_quality.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
labels = ['one', 'two', 'three', 'four', 'five',
          'six', 'seven', 'eight', 'nine', 'ten']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Get predictions
y_pred = model.predict(X_test)
y_probas = model.predict_proba(X_test)
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print(X_train.info())

# Visualize model performance
wandb.sklearn.plot_classifier(
    model, X_train, X_test, y_train, y_test, y_pred, y_probas, labels,
    is_binary=False, model_name='RandomForest', feature_names=feature_names)