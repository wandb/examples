from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pandas as pd
import wandb

# initialize wandb run
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
