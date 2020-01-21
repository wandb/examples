from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd
import wandb
wandb.init()

# Load data
boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model, get predictions
reg = Ridge()
reg.fit(X, y)
y_pred = reg.predict(X_test)

# Visualize model performance
wandb.sklearn.plot_outlier_candidates(reg, X, y)
wandb.sklearn.plot_residuals(reg, X, y)
wandb.sklearn.plot_alpha_selection(reg, X, y)
# wandb.sklearn.log(reg, X=X, y=y, X_test=X_test, y_test=y_test)
