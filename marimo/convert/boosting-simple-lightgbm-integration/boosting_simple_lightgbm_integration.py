# /// script
# dependencies = ["lightgbm", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Simple_LightGBM_Integration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{simple-lightgbm} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="800">
    <!--- @wandbcode{simple-lightgbm} -->

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B + LightGBM
    Gradient boosting decision trees are the state of the art when it comes to building predictive models for structured data.

    [LigthGBM](https://github.com/microsoft/LightGBM), a gradient boosting framework by Microsoft, has dethroned xgboost and become the go to GBDT algorithm (along with catboost). It outperforms xgboost in training speeds, memory usage and the size of datasets it can handle. LightGBM does so by using histogram-based algorithms to bucket continuous features into discrete bins during training.

    You can find the **[W&B + LightGBM documentation here](https://docs.wandb.ai/guides/integrations/boosting)**

    ## What this notebook covers
    * Easy integration of Weights and Biases with LightGBM.
    * `wandb_callback()` callback for metrics logging
    * `log_summary()` function to log a feature importance plot and enable model saving to W&B

    We want to make it incredible easy for people to look under the hood of their models, so we built a callback that helps you visualize your LightGBM’s performance in just one line of code.

    **Note**: Sections starting with _Step_ is all you need to integrate W&B.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Install, Import, and Log in
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Usual Suspects
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: lightgbm>=4.0.0 !pip install -Uq 'lightgbm>=4.0.0'
    return


@app.cell
def _():
    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error

    return lgb, mean_squared_error, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 0: Install W&B
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qU wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Import W&B and Login
    """)
    return


@app.cell
def _():
    import wandb
    from wandb.integration.lightgbm import wandb_callback, log_summary

    return log_summary, wandb, wandb_callback


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download and Prepare Dataset
    """)
    return


@app.cell
def _(subprocess):
    #! wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.train -qq
    subprocess.call(['wget', 'https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.train', '-qq'])
    #! wget https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.test -qq
    subprocess.call(['wget', 'https://raw.githubusercontent.com/microsoft/LightGBM/master/examples/regression/regression.test', '-qq'])
    return


@app.cell
def _(lgb, pd):
    # load or create your dataset
    df_train = pd.read_csv('regression.train', header=None, sep='\t')
    df_test = pd.read_csv('regression.test', header=None, sep='\t')

    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis=1)
    X_test = df_test.drop(0, axis=1)

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    return X_test, lgb_eval, lgb_train, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: Initialize your wandb run.

    Using `wandb.init()` initialize your W&B run. You can also pass a dictionary of configs. [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/init)

    You can't deny the importance of configs in your ML/DL workflow. W&B makes sure that you have access to the right config to reproduce your model.

    [Learn more about configs in this colab notebook $\rightarrow$](http://wandb.me/config-colab)
    """)
    return


@app.cell
def _(wandb):
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
        'verbosity': 0,
        'early_stopping_rounds': 5,
    }

    wandb.init(project='my-lightgbm-project', config=params)
    return (params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Once you have trained your model come back and click on the **Project page**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: Train with `wandb_callback`
    """)
    return


@app.cell
def _(lgb, lgb_eval, lgb_train, params, wandb_callback):
    # train 
    # add lightgbm callback
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=30,
                    valid_sets=lgb_eval,
                    valid_names=('validation'),
                    callbacks=[wandb_callback()],
                    )
    return (gbm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 4: Log Feature Importance and Upload Model with `log_summary`
    `log_summary` will upload calculate and upload the feature importance import and (optionally) upload your trained model to W&B Artifacts so you can use it later
    """)
    return


@app.cell
def _(gbm, log_summary):
    log_summary(gbm, save_model_checkpoint=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluate
    """)
    return


@app.cell
def _(X_test, gbm, mean_squared_error, wandb, y_test):
    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    wandb.log({'rmse_prediction': mean_squared_error(y_test, y_pred) ** 0.5})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you are finished logging for a particular W&B run its a good idea to call `wandb.finish()` to tidy up the wandb process (only necessary when using notebooks/colabs)
    """)
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualize Results

    Click on the **project page** link above to see your results automatically visualized.

    <img src="https://imgur.com/S6lwSig.png" alt="Viz" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sweep 101

    Use Weights & Biases Sweeps to automate hyperparameter optimization and explore the space of possible models.

    ## [Check out Hyperparameter Optimization with XGBoost  using W&B Sweep $\rightarrow$](http://wandb.me/xgb-colab)

    Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

    1. **Define the sweep:** We do this by creating a dictionary or a [YAML file](https://docs.wandb.com/library/sweeps/configuration) that specifies the parameters to search through, the search strategy, the optimization metric et all.

    2. **Initialize the sweep:**
    `sweep_id = wandb.sweep(sweep_config)`

    3. **Run the sweep agent:**
    `wandb.agent(sweep_id, function=train)`

    And voila! That's all there is to running a hyperparameter sweep!

    <img src="https://imgur.com/SVtMfa2.png" alt="Sweep Result" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example Gallery

    See examples of projects tracked and visualized with W&B in our [Gallery →](https://app.wandb.ai/gallery)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Basic Setup
    1. **Projects**: Log multiple runs to a project to compare them. `wandb.init(project="project-name")`
    2. **Groups**: For multiple processes or cross validation folds, log each process as a runs and group them together. `wandb.init(group='experiment-1')`
    3. **Tags**: Add tags to track your current baseline or production model.
    4. **Notes**: Type notes in the table to track the changes between runs.
    5. **Reports**: Take quick notes on progress to share with colleagues and make dashboards and snapshots of your ML projects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Advanced Setup
    1. [Environment variables](https://docs.wandb.com/library/environment-variables): Set API keys in environment variables so you can run training on a managed cluster.
    2. [Offline mode](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): Use `dryrun` mode to train offline and sync results later.
    3. [On-prem](https://docs.wandb.com/self-hosted): Install W&B in a private cloud or air-gapped servers in your own infrastructure. We have local installations for everyone from academics to enterprise teams.
    4. [Sweeps](https://docs.wandb.com/sweeps): Set up hyperparameter search quickly with our lightweight tool for tuning.
    """)
    return


if __name__ == "__main__":
    app.run()
