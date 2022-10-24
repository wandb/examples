# Sweeps with XGBoost
# Similar to this post: https://towardsdatascience.com/how-to-use-w-b-sweeps-with-lightgbm-for-hyperparameter-tuning-b67c3cac435c but uses xgboost

import wandb
from xgboost import XGBRegressor
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
Housing_Dataset = fetch_california_housing()
X = Housing_Dataset["data"]
y = Housing_Dataset["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123
)


def train_model():
    wandb.init()

    model = XGBRegressor(
        max_depth=wandb.config.max_depth,
        learning_rate=wandb.config.learning_rate,
        n_estimators=wandb.config.n_estimators,
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_preds = model.predict(X_test)

    # Evaluate predictions
    mae_score = mean_absolute_error(y_test, y_preds)
    mse_score = mean_squared_error(y_test, y_preds)

    # Log model performance metrics to W&B
    wandb.log({"mae": mae_score, "mse": mse_score})


sweep_configs = {
    "method": "bayes",
    "metric": {"name": "mse", "goal": "minimize"},
    "parameters": {
        "max_depth": {"values": [1, 2, 5, 10]},
        "learning_rate": {"distribution": "uniform", "min": 0, "max": 0.1},
        "n_estimators": {"values": [100, 500, 1000]},
    },
}

sweep_id = wandb.sweep(sweep_configs, project="traditional_ml_sweeps_example")
wandb.agent(sweep_id=sweep_id, function=train_model)
