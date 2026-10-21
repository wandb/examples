# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.2",
#     "numpy>=1.26,<3",
#     "pandas>=2.1,<3",
#     "scikit-learn>=1.2,<2",
#     "scipy>=1.11,<2",
#     "wandb>=0.19.10,<1",
#     "xgboost>=2.0,<4",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///
"""Track a vehicle-loan classification pipeline with XGBoost and Weights & Biases."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Vehicle loan classification with XGBoost and Weights & Biases",
)

with app.setup:
    import inspect
    import tempfile
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import xgboost as xgb
    from scipy.stats import ks_2samp
    from sklearn import metrics, model_selection
    import wandb
    from wandb.integration.xgboost import WandbCallback


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-config-logo--dark {
      display: none;
    }

    body.dark .wandb-config-logo--light {
      display: none;
    }

    body.dark .wandb-config-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-config-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-config-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />

    # Vehicle Loan Default Prediction with XGBoost and Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/credit-scorecards-with-xgboost-and-w-b/credit_scorecards_with_xgboost_and_w_b.py/server)

    This lesson tracks a classification pipeline from raw data to trained model.
    We use Weights & Biases Artifacts to record dataset lineage, Tables to inspect sample rows,
    the XGBoost callback to log training, and Sweeps to compare hyperparameters.

    The example uses a subset of the
    [Vehicle Loan Default Prediction dataset](https://www.kaggle.com/sneharshinde/ltfs-av-data)
    and preprocessing code shared by Scotiabank in the
    [RAPIDS credit-scorecard example](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard).
    The teaching objective is to track the data, settings, and results of a
    classification experiment. The code does not construct or validate a
    points-based lending scorecard.

    > On macOS, XGBoost also needs OpenMP. If importing XGBoost reports a missing
    `libomp.dylib`, install it with `brew install libomp` and restart.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `vehicle_loan_default`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="vehicle_loan_default", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    entity = _entity
    project = _project
    mo.md("✓ Connected to Weights & Biases.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Get the dataset

    The next cell identifies a dataset Artifact already stored in Weights & Biases,
    sets the ID and target column names, and creates a temporary working directory.
    The preprocessing run will download this Artifact and record it as an input.
    You need access to the source Artifact to run that step.
    """)
    return


@app.cell
def _():
    source_artifact_path = "morgan/credit_scorecard/vehicle_loan_defaults:latest"
    id_columns = ["UniqueID"]
    target_column = "loan_default"
    data_dir = Path(tempfile.mkdtemp(prefix="wandb-credit-scorecard-"))
    return data_dir, id_columns, source_artifact_path, target_column


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The source example treats age, time since disbursement, loan-to-value ratio
    (`LTV`), and credit bureau score (`PERFORM_CNS_SCORE`) as numeric features.
    Other predictors, including numeric IDs for states and manufacturers, are
    categories. Their numeric codes identify groups rather than an ordered scale.

    The function below preserves the four numeric features and creates an
    indicator column for each category with `pd.get_dummies()`. It retains
    the row ID and target in the saved dataset, but excludes both from the
    predictor list used for training. Encoding once before the split gives
    both partitions the same columns. Although this uses no target values, it
    does expose the category vocabulary of the validation data. In a production
    evaluation, fit the encoder on training data and handle unseen categories
    when transforming validation or test data.
    """)
    return


@app.function
def one_hot_encode_data(dataset, id_columns, target_column):
    numeric_columns = ["AgeInMonths", "DaysSinceDisbursement", "LTV", "PERFORM_CNS_SCORE"]
    excluded_columns = id_columns + [target_column]
    categorical_columns = sorted(set(dataset.columns) - set(excluded_columns + numeric_columns))
    processed_data = pd.get_dummies(
        dataset,
        columns=categorical_columns,
        prefix_sep="__",
        dtype=int,
    )
    predictor_columns = processed_data.columns.drop(excluded_columns).tolist()
    return processed_data, predictor_columns


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Preprocess and Log a Dataset Artifact

    The next cell starts `preprocessing_run`. Calling
    `preprocessing_run.use_artifact()` records the raw dataset as this run’s input.
    Calling `download()` retrieves its files. We read the CSV and apply the
    encoding function (`one_hot_encode_data`) to produce numeric predictors.

    We save the processed CSV, attach it to a new `wandb.Artifact()`, and call
    `preprocessing_run.log_artifact()`. Consuming and producing Artifacts in
    the same run records the transformation in Weights & Biases's lineage graph.
    The Artifact metadata also stores the encoding function's source so the
    saved dataset can be traced to its transformation.
    `wait()` ensures the upload finishes before the next stage fetches it.
    """)
    return


@app.cell
def _(
    data_dir,
    entity,
    id_columns,
    project,
    source_artifact_path,
    target_column,
):
    with wandb.init(entity=entity, project=project, job_type="preprocess-data", dir=str(data_dir)) as preprocessing_run:
        source_artifact = preprocessing_run.use_artifact(source_artifact_path, type="dataset")
        raw_dataset_dir = Path(source_artifact.download(root=str(data_dir / "raw")))
        raw_csv_path = next(raw_dataset_dir.rglob("vehicle_loans_subset.csv"))
        raw_data = pd.read_csv(raw_csv_path)
        processed_data, predictor_columns = one_hot_encode_data(raw_data, id_columns, target_column)

        processed_csv_path = data_dir / "proc_ds.csv"
        processed_data.to_csv(processed_csv_path, index=False)
        processed_artifact = wandb.Artifact(
            "vehicle_defaults_processed",
            type="processed_dataset",
            description="One-hot encoded vehicle-loan data",
            metadata={"preprocessing_fn": inspect.getsource(one_hot_encode_data)},
        )
        processed_artifact.add_file(str(processed_csv_path))
        logged_processed_artifact = preprocessing_run.log_artifact(processed_artifact)
        logged_processed_artifact.wait()
        processed_artifact_path = logged_processed_artifact.qualified_name
    return predictor_columns, processed_artifact_path


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Split the Data and Record Lineage

    The next cell consumes the processed Artifact in `split_run`, then makes a
    reproducible, stratified train/validation split. Stratification preserves
    the target-class proportions approximately in both partitions.

    The split settings are recorded in the run configuration and Artifact
    metadata. Logging the two CSVs as one Artifact connects the raw data,
    preprocessing, and split stages. We keep the uploaded version's full path
    for subsequent runs so they consume this exact split.
    """)
    return


@app.cell
def _(data_dir, entity, processed_artifact_path, project, target_column):
    split_config = {"test_size": 0.25, "random_state": 42}

    with wandb.init(entity=entity, project=project, job_type="train-val-split", config=split_config, dir=str(data_dir)) as split_run:
        split_input_artifact = split_run.use_artifact(processed_artifact_path, type="processed_dataset")
        split_input_dir = Path(split_input_artifact.download(root=str(data_dir / "processed")))
        split_data = pd.read_csv(next(split_input_dir.rglob("proc_ds.csv")))
        train_data, validation_data = model_selection.train_test_split(
            split_data,
            **split_config,
            stratify=split_data[target_column],
        )
        train_csv_path = data_dir / "train.csv"
        validation_csv_path = data_dir / "val.csv"
        train_data.to_csv(train_csv_path, index=False)
        validation_data.to_csv(validation_csv_path, index=False)

        split_artifact = wandb.Artifact(
            "vehicle_defaults_split",
            type="train-val-dataset",
            description="Training and validation partitions",
            metadata=split_config,
        )
        split_artifact.add_file(str(train_csv_path))
        split_artifact.add_file(str(validation_csv_path))
        logged_split_artifact = split_run.log_artifact(split_artifact)
        logged_split_artifact.wait()
        split_artifact_path = logged_split_artifact.qualified_name
        print(f"Training rows: {len(train_data):,}; validation rows: {len(validation_data):,}")
    return split_artifact_path, train_data, validation_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Inspect Data with Weights & Biases Tables

    The next cell initializes a run (`table_run`) and records the split Artifact as an input.
    `wandb.Table(dataframe=...)` turns a reproducible sample of up to 1,000
    training rows into an interactive table. `table_run.log()` saves it under
    `processed_dataset` for sorting, filtering, and inspection in the Weights & Biases App UI.
    """)
    return


@app.cell
def _(data_dir, entity, project, split_artifact_path, train_data):
    with wandb.init(entity=entity, project=project, job_type="log-dataset", dir=str(data_dir)) as table_run:
        table_run.use_artifact(split_artifact_path, type="train-val-dataset")
        dataset_table = wandb.Table(
            dataframe=train_data.sample(min(1000, len(train_data)), random_state=42)
        )
        table_run.log({"processed_dataset": dataset_table})
        print(f"View the dataset table: {table_run.url}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare Predictors and Model Settings

    The next cell separates predictors from the binary target, computes the
    training default rate, and defines the XGBoost parameters. We use histogram trees.
    `base_score` initializes predictions at the training default rate before
    any trees are added.

    Early stopping monitors the last evaluation metric, validation log loss,
    and stops after five rounds without improvement. `n_estimators` caps
    training at 25 trees, so the demonstration remains short.
    The settings are recorded in the training run and reused as sweep defaults.
    """)
    return


@app.cell
def _(predictor_columns, target_column, train_data, validation_data):
    X_train = train_data[predictor_columns]
    X_validation = validation_data[predictor_columns]
    y_train = train_data[target_column].astype(int)
    y_validation = validation_data[target_column].astype(int)
    base_rate = float(y_train.mean())
    early_stopping_rounds = 5
    model_parameters = {
        "objective": "binary:logistic",
        "base_score": base_rate,
        "gamma": 1,
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 100,
        "n_estimators": 25,
        "n_jobs": 4,
        "random_state": 42,
        "reg_alpha": 0,
        "reg_lambda": 0,
        "eval_metric": ["auc", "logloss"],
        "tree_method": "hist",
    }
    print(f"Training default rate: {base_rate:.2%}; predictors: {len(predictor_columns)}")
    return (
        X_train,
        X_validation,
        early_stopping_rounds,
        model_parameters,
        y_train,
        y_validation,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define Evaluation Metrics

    The following helper calculates AUC, log loss, and KS statistics for the
    baseline and sweep models from their predicted default probabilities.

    - AUC measures how well the model ranks positive examples above negative examples. Higher is better.
    - Log loss penalizes incorrect predicted probabilities, especially confident errors. Lower is better.
    - KS measures the largest gap between the two classes’ cumulative score distributions. A larger gap indicates stronger separation.

    The helper uses the full training and validation sets, with a threshold
    of 0.5 for validation accuracy. Predictions use XGBoost's best iteration
    selected by early stopping.

    These are descriptive model metrics. They do not establish calibration,
    fairness, or suitability for lending decisions.
    """)
    return


@app.function
def evaluate_classifier(classifier, X_train, y_train, X_validation, y_validation):
    train_probabilities = classifier.predict_proba(X_train)[:, 1]
    validation_probabilities = classifier.predict_proba(X_validation)[:, 1]
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(y_train, train_probabilities)
    ks_stat, ks_pval = ks_2samp(
        validation_probabilities[y_validation == 1],
        validation_probabilities[y_validation == 0],
    )
    return {
        "best_iteration": classifier.get_booster().best_iteration,
        "train_ks_stat": float(np.max(true_positive_rate - false_positive_rate)),
        "train_auc": float(metrics.roc_auc_score(y_train, train_probabilities)),
        "train_log_loss": float(metrics.log_loss(y_train, train_probabilities)),
        "val_ks_2samp": float(ks_stat),
        "val_ks_pval": float(ks_pval),
        "val_auc": float(metrics.roc_auc_score(y_validation, validation_probabilities)),
        "val_acc_0.5": float(metrics.accuracy_score(y_validation, validation_probabilities >= 0.5)),
        "val_log_loss": float(metrics.log_loss(y_validation, validation_probabilities)),
    }


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train and Log the Baseline

    The next cell starts `training_run`, records the split Artifact as an input,
    and logs hyperparameters in its configuration. Passing
    `WandbCallback(log_model=True)` to `XGBClassifier` logs evaluation metrics
    and feature importance during training, and uploads the trained booster as
    a model Artifact.

    After fitting, we add evaluation values to `training_run.summary` and log
    validation AUC plus a ROC curve. The curve receives probabilities for both
    classes. For large datasets, only the chart is subsampled. The numeric
    evaluation metrics still use all validation rows.

    Inspect `validation_0-auc` and `validation_0-logloss` in the run's charts,
    then compare the summary's `train_auc` and `val_auc` for signs of overfitting.
    """)
    return


@app.cell
def _(
    X_train,
    X_validation,
    data_dir,
    early_stopping_rounds,
    entity,
    model_parameters,
    project,
    split_artifact_path,
    y_train,
    y_validation,
):
    with wandb.init(
        entity=entity,
        project=project,
        job_type="train-model",
        dir=str(data_dir),
        config={**model_parameters, "early_stopping_rounds": early_stopping_rounds},
    ) as training_run:
        training_run.use_artifact(split_artifact_path, type="train-val-dataset")
        classifier = xgb.XGBClassifier(
            **model_parameters,
            callbacks=[WandbCallback(log_model=True)],
            early_stopping_rounds=early_stopping_rounds,
        )
        classifier.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])
        baseline_metrics = evaluate_classifier(classifier, X_train, y_train, X_validation, y_validation)
        training_run.summary.update(baseline_metrics)

        validation_scores = classifier.predict_proba(X_validation)
        roc_stride = max(1, int(np.ceil(len(y_validation) / 10_000)))
        training_run.log({
            "val_auc": baseline_metrics["val_auc"],
            "ROC_Curve": wandb.plot.roc_curve(
                y_validation.to_numpy()[::roc_stride],
                validation_scores[::roc_stride],
                labels=["no_default", "loan_default"],
                classes_to_plot=[1],
            ),
        })

    print(f"Validation AUC: {baseline_metrics['val_auc']:.4f}")
    return (baseline_metrics,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define a Hyperparameter Sweep

    The next cell specifies random search over learning rate, gamma, minimum
    child weight, and early-stopping rounds. The objective is to maximize
    `val_auc`, which each trial explicitly logs.

    The patience values are smaller than the 25-tree limit so every choice
    can affect when a trial stops. Learning rate controls each tree's
    contribution while the gamma and minimum child weight constrain tree growth.

    Sweeps automate comparison on this fixed validation split. Repeated tuning
    can overfit the validation set. It is recommended that you keep an independent test set to assess the selected model's performance.
    """)
    return


@app.cell
def _():
    sweep_config = {
        "method": "random",
        "metric": {"name": "val_auc", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"min": 0.001, "max": 1.0},
            "gamma": {"min": 0.001, "max": 1.0},
            "min_child_weight": {"min": 1, "max": 150},
            "early_stopping_rounds": {"values": [3, 5, 10]},
        },
    }
    return (sweep_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define Training Function

    The next cell defines the function the sweep agent calls for each trial.

    Each trial consumes the same split Artifact, creates a fresh classifier
    and callback, trains, and logs `val_auc`. `job_type="sweep"` distinguishes
    these runs from the baseline. We leave `log_model=False` here to avoid
    saving every trial's booster.
    """)
    return


@app.cell
def _(
    X_train,
    X_validation,
    data_dir,
    early_stopping_rounds,
    entity,
    model_parameters,
    project,
    split_artifact_path,
    y_train,
    y_validation,
):
    def sweep_train():
        with wandb.init(
            entity=entity,
            project=project,
            job_type="sweep",
            dir=str(data_dir),
            config={**model_parameters, "early_stopping_rounds": early_stopping_rounds},
        ) as sweep_run:
            sweep_run.use_artifact(split_artifact_path, type="train-val-dataset")
            trial_parameters = {key: sweep_run.config[key] for key in model_parameters}
            trial_classifier = xgb.XGBClassifier(
                **trial_parameters,
                callbacks=[WandbCallback(log_model=False)],
                early_stopping_rounds=sweep_run.config["early_stopping_rounds"],
            )
            trial_classifier.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])
            trial_metrics = evaluate_classifier(trial_classifier, X_train, y_train, X_validation, y_validation)
            sweep_run.summary.update(trial_metrics)
            sweep_run.log({"val_auc": trial_metrics["val_auc"]})

    return (sweep_train,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Create the Sweep and Start an Agent

    The next cell creates a sweep (`wandb.sweep()`), stores
    the sweep ID, then calls `wandb.agent()`. We pass `sweep_train` as the training function and
    limit this agent to five training trials (`count=5`). Lower `count` for a quicker experiment.

    Each run closes through its context manager before the next trial starts.
    The dashboard compares the selected parameters against validation AUC.
    """)
    return


@app.cell
def _(baseline_metrics, entity, project, sweep_config, sweep_train):
    print(f"Baseline validation AUC: {baseline_metrics['val_auc']:.4f}")
    sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)
    wandb.agent(sweep_id, function=sweep_train, entity=entity, project=project, count=5)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify and Explore the Experiment Records

    Open the baseline run to inspect its metrics,
    feature importance, ROC curve, and saved model. The Artifacts graph connects
    the source dataset, `vehicle_defaults_processed`, `vehicle_defaults_split`,
    and the training run. Open the `log-dataset` run's `processed_dataset`
    Table to inspect the encoded inputs.

    Open the sweep dashboard to compare hyperparameters and
    validation AUC. Inspect parameter trends rather than relying on a single
    best trial.

    See the [Weights & Biases XGBoost integration](https://docs.wandb.ai/models/integrations/xgboost)
    and [Sweeps configuration guide](https://docs.wandb.ai/models/sweeps/define-sweep-configuration)
    for more options.
    """)
    return


if __name__ == "__main__":
    app.run()
