# W&B 💖 🌳s

## 📍 XGBoost
Use our callback to compare results between different versions of your XGBoost model.
```python
import wandb

# 1. Start a new run
wandb.init(project="visualize-models", name="xgboost")

# 2. Add the callback
bst = xgboost.train(param, xg_train, num_round, watchlist, callbacks=[wandb.xgboost.wandb_callback()])

# Get predictions
pred = bst.predict(xg_test)
```

- **[Try in a colab →](http://wandb.me/xgb-colab)**
- [Docs](https://docs.wandb.com/library/integrations/xgboost)


## 📍 LightGBM
Use our callback to visualize your LightGBM’s performance in just one line of code.
```python
import wandb
import lightgbm as lgb

# 1. Start a W&B run
wandb.init(project="visualize-models", name="lightgbm")

# 2. Add the wandb callback
bst = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                valid_names=('validation'),
                callbacks=[wandb.lightgbm.callback()])

# Get prediction
pred = bst.predict(lgb_test)
```

- **[Try in a colab →](http://wandb.me/lightgbm-colab)**
- [Docs](https://docs.wandb.com/library/integrations/lightgbm)
