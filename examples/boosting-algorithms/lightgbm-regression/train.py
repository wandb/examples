# import wandb and the lightgbm callback
import wandb
from wandb.lightgbm import log_summary, wandb_callback
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error

# initialize a new wandb project
wandb.init()

# load or create your dataset
import requests
train = requests.get('https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.train')
test = requests.get('https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.test')
open('regression.train', 'wb').write(train.content)
open('regression.test', 'wb').write(test.content)
df_train = pd.read_csv('regression.train', header=None, sep='\t')
df_test = pd.read_csv('regression.test', header=None, sep='\t')

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': ['rmse', 'l2', 'l1', 'huber'],
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1
}
wandb.config.update(params)

# train
# add lightgbm callback
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                valid_names=('validation'),
                callbacks=[wandb_callback(),
                           lgb.early_stopping(stopping_rounds=5)])

# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# log feature importance and model checkpoint 
log_summary(gbm, save_model_checkpoint=True)
