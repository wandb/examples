import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/scikit/wandb_decision_tree.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Author: [@SauravMaheshkar](https://twitter.com/MaheshkarSaurav)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Packages 📦 and Basic Setup
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install Packages
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # ## Install Sklearn
    # !pip install -U scikit-learn
    # ## Install the latest version of wandb client 🔥🔥
    # !pip install -q --upgrade wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Project Configuration using **`wandb.config`**
    """)
    return


@app.cell
def _():
    import wandb

    ## Importing Libraries
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    return (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        load_iris,
        train_test_split,
        wandb,
    )


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _(wandb):
    # Initialize the run
    run = wandb.init(project='simple-scikit')

    # Feel free to change these and experiment !!
    config = wandb.config
    config.max_depth = 5
    config.min_samples_split = 2
    config.clf_criterion = "gini"
    config.reg_criterion = "mse"
    config.splitter = "best"
    config.dataset = "iris"
    config.test_size = 0.2
    config.random_state = 42
    config.labels =['setosa', 'versicolor', 'virginica']

    # Update the config
    wandb.config.update(config)
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 💿 Dataset
    ---
    """)
    return


@app.cell
def _(load_iris):
    ## Loading the Dataset
    iris = load_iris(return_X_y = True, as_frame= True)
    dataset = iris[0]
    target = iris[1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ✍️ Model Architecture
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification
    """)
    return


@app.cell
def _(DecisionTreeClassifier, config, load_iris, train_test_split, wandb):
    _X, _y = load_iris(return_X_y=True)
    _x_train, _x_test, _y_train, _y_test = train_test_split(_X, _y, test_size=config.test_size, random_state=config.random_state)
    clf = DecisionTreeClassifier(max_depth=config.max_depth, min_samples_split=config.min_samples_split, criterion=config.clf_criterion, splitter=config.splitter)
    clf = clf.fit(_x_train, _y_train)
    y_pred = clf.predict(_x_test)
    # Visualize Confustion Matrix
    wandb.sklearn.plot_confusion_matrix(_y_test, y_pred, config.labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regression
    """)
    return


@app.cell
def _(DecisionTreeRegressor, config, load_iris, train_test_split, wandb):
    _X, _y = load_iris(return_X_y=True)
    _x_train, _x_test, _y_train, _y_test = train_test_split(_X, _y, test_size=config.test_size, random_state=config.random_state)
    reg = DecisionTreeRegressor(max_depth=config.max_depth, min_samples_split=config.min_samples_split, criterion=config.reg_criterion, splitter=config.splitter)
    reg = reg.fit(_x_train, _y_train)
    # All regression plots
    wandb.sklearn.plot_regressor(reg, _x_train, _x_test, _y_train, _y_test, model_name='DecisionTreeRegressor')
    return


@app.cell
def _(wandb):
    # Finish the W&B Process
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
