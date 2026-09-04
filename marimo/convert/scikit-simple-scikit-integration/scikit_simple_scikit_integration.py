# /// script
# dependencies = ["wandb"]
# ///

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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/scikit/Simple_Scikit_Integration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{simple-sklearn} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{simple-sklearn} -->

    # 🏋️‍♀️ W&B + 🧪 Scikit-learn
    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    ## What this notebook covers:
    * Easy integration of Weights and Biases with Scikit.
    * W&B Scikit plots for model interpretation and diagnostics for regression, classification, and clustering.

    **Note**: Sections starting with _Step_ are all you need to integrate W&B to existing code.

    ## The interactive W&B Dashboard will look like this:

    ![](https://i.imgur.com/F1ZgR4A.png)
    """)
    return


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cluster import KMeans
    from sklearn import datasets, cluster

    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    return (
        KMeans,
        RandomForestClassifier,
        Ridge,
        datasets,
        np,
        pd,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 0: Install W&B
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qU
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

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Regression

    **Let's check out a quick example**
    """)
    return


@app.cell
def _(Ridge, datasets, pd, train_test_split, wandb):
    # Load data
    housing = datasets.fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    _y = housing.target
    X, _y = (X[::2], _y[::2])  # subsample for faster demo
    wandb.errors.term._show_warnings = False
    # ignore warnings about charts being built from subset of data
    X_train, X_test, y_train, y_test = train_test_split(X, _y, test_size=0.3)
    reg = Ridge()
    reg.fit(X_train, y_train)
    # Train model, get predictions
    y_pred = reg.predict(X_test)
    return X_test, X_train, reg, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Initialize W&B run
    """)
    return


@app.cell
def _(wandb):
    _run = wandb.init(project='my-scikit-integration', name='regression')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Visualize model performance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Residual Plot

    Measures and plots the predicted target values (y-axis) vs the difference between actual and predicted target values (x-axis), as well as the distribution of the residual error.

    Generally, the residuals of a well-fit model should be randomly distributed because good models will account for most phenomena in a data set, except for random error.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#residuals-plot)
    """)
    return


@app.cell
def _(X_train, reg, wandb, y_train):
    wandb.sklearn.plot_residuals(reg, X_train, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Outlier Candidate

    Measures a datapoint's influence on regression model via Cook's distance. Instances with heavily skewed influences could potentially be outliers. Useful for outlier detection.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#outlier-candidates-plot)
    """)
    return


@app.cell
def _(X_train, reg, wandb, y_train):
    wandb.sklearn.plot_outlier_candidates(reg, X_train, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All-in-one: Regression plot

    Using this all in one API one can:
    * Log summary of metrics
    * Log learning curve
    * Log outlier candidates
    * Log residual plot
    """)
    return


@app.cell
def _(X_test, X_train, reg, wandb, y_test, y_train):
    wandb.sklearn.plot_regressor(reg, X_train, X_test, y_train, y_test, model_name='Ridge')

    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Classification

    **Let's check out a quick example.**
    """)
    return


@app.cell
def _(RandomForestClassifier, datasets, np, train_test_split):
    # Load data
    wbcd = wisconsin_breast_cancer_data = datasets.load_breast_cancer()
    feature_names = wbcd.feature_names
    labels = wbcd.target_names
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(wbcd.data, wbcd.target, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train_1, y_train_1)
    y_pred_1 = model.predict(X_test_1)
    # Train model, get predictions
    y_probas = model.predict_proba(X_test_1)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return (
        X_test_1,
        X_train_1,
        labels,
        model,
        y_pred_1,
        y_probas,
        y_test_1,
        y_train_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Initialize W&B run
    """)
    return


@app.cell
def _(wandb):
    _run = wandb.init(project='my-scikit-integration', name='classification')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Visualize model performance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Class Proportions

    Plots the distribution of target classes in training and test sets. Useful for detecting imbalanced classes and ensuring that one class doesn't have a disproportionate influence on the model.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#class-proportions)
    """)
    return


@app.cell
def _(labels, wandb, y_test_1, y_train_1):
    wandb.sklearn.plot_class_proportions(y_train_1, y_test_1, labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Learning Curve

    Trains model on datasets of varying lengths and generates a plot of cross validated scores vs dataset size, for both training and test sets.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#learning-curve)
    """)
    return


@app.cell
def _(X_train_1, model, wandb, y_train_1):
    wandb.sklearn.plot_learning_curve(model, X_train_1, y_train_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ROC

    ROC curves plot true positive rate (y-axis) vs false positive rate (x-axis). The ideal score is a `TPR = 1` and `FPR = 0`, which is the point on the top left. Typically we calculate the area under the ROC curve (AUC-ROC), and the greater the AUC-ROC the better.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#roc)
    """)
    return


@app.cell
def _(labels, wandb, y_probas, y_test_1):
    wandb.sklearn.plot_roc(y_test_1, y_probas, labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Precision Recall Curve

    Computes the tradeoff between precision and recall for different thresholds. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#precision-recall-curve)
    """)
    return


@app.cell
def _(labels, wandb, y_probas, y_test_1):
    wandb.sklearn.plot_precision_recall(y_test_1, y_probas, labels)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Feature Importances

    Evaluates and plots the importance of each feature for the classification task. Only works with classifiers that have a `feature_importances_` attribute, like trees.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#feature-importances)
    """)
    return


@app.cell
def _(model, wandb):
    wandb.sklearn.plot_feature_importances(model);
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All-in-one: Classifier Plot

    Using this all in one API one can:
    * Log feature importance
    * Log learning curve
    * Log confusion matrix
    * Log summary metrics
    * Log class proportions
    * Log calibration curve
    * Log roc curve
    * Log precision recall curve
    """)
    return


@app.cell
def _(
    X_test_1,
    X_train_1,
    labels,
    model,
    wandb,
    y_pred_1,
    y_probas,
    y_test_1,
    y_train_1,
):
    wandb.sklearn.plot_classifier(model, X_train_1, X_test_1, y_train_1, y_test_1, y_pred_1, y_probas, labels, is_binary=True, model_name='RandomForest')
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Clustering
    """)
    return


@app.cell
def _(KMeans, datasets, np):
    iris = datasets.load_iris()
    X_1, _y = (iris.data, iris.target)
    names = iris.target_names

    def get_label_ids(classes):
        return np.array([names[aclass] for aclass in classes])
    labels_1 = get_label_ids(_y)
    kmeans = KMeans(n_clusters=4, random_state=1)
    cluster_labels = kmeans.fit_predict(X_1)
    return X_1, cluster_labels, kmeans, labels_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Initialize W&B run
    """)
    return


@app.cell
def _(wandb):
    _run = wandb.init(project='my-scikit-integration', name='clustering')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Visualize model performance
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elbow Plot

    Measures and plots the percentage of variance explained as a function of the number of clusters, along with training times. Useful in picking the optimal number of clusters.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#elbow-plot)
    """)
    return


@app.cell
def _(X_1, kmeans, wandb):
    wandb.sklearn.plot_elbow_curve(kmeans, X_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Silhouette Plot

    Measures & plots how close each point in one cluster is to points in the neighboring clusters. The thickness of the clusters corresponds to the cluster size. The vertical line represents the average silhouette score of all the points.

    [Check out the official documentation here $\rightarrow$](https://docs.wandb.com/library/integrations/scikit#silhouette-plot)
    """)
    return


@app.cell
def _(X_1, kmeans, labels_1, wandb):
    wandb.sklearn.plot_silhouette(kmeans, X_1, labels_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All in one: Clusterer Plot

    Using this all-in-one API you can:
    * Log elbow curve
    * Log silhouette plot
    """)
    return


@app.cell
def _(X_1, cluster_labels, kmeans, labels_1, wandb):
    wandb.sklearn.plot_clusterer(kmeans, X_1, cluster_labels, labels_1, 'KMeans')
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Sweep 101

    Use Weights & Biases Sweeps to automate hyperparameter optimization and explore the space of possible models.

    ## [Check out Hyperparameter Optimization in PyTorch using W&B Sweeps $\rightarrow$](http://wandb.me/sweeps-colab)

    Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

    1. **Define the sweep:** We do this by creating a dictionary or a [YAML file](https://docs.wandb.com/library/sweeps/configuration) that specifies the parameters to search through, the search strategy, the optimization metric et all.

    2. **Initialize the sweep:**
    `sweep_id = wandb.sweep(sweep_config)`

    3. **Run the sweep agent:**
    `wandb.agent(sweep_id, function=train)`

    And voila! That's all there is to running a hyperparameter sweep! In the notebook below, we'll walk through these 3 steps in more detail.

    <img src="https://imgur.com/sdQXdDz.png" alt="Sweep Result" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example Gallery

    See examples of projects tracked and visualized with W&B in our gallery, [Fully Connected →](https://wandb.me/fc)
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
    """)
    return


if __name__ == "__main__":
    app.run()
