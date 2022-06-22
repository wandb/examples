from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pandas as pd
import wandb

# initialize wandb run
wandb.init(project="regression-housing-demo")

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model, get predictions
reg = Ridge()
reg.fit(X.values, y)
y_pred = reg.predict(X_test.values)

# Visualize model performance
wandb.sklearn.plot_outlier_candidates(reg, X.values, y)
wandb.sklearn.plot_residuals(reg, X.values, y)
