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
# ///
"""Track a vehicle-loan classification pipeline with XGBoost and W&B."""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Vehicle loan classification with XGBoost and W&B",
)

with app.setup:
    import importlib.util
    import inspect
    import sys
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
    # Vehicle Loan Default Prediction with XGBoost and Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/credit-scorecards-with-xgboost-and-w-b/credit_scorecards_with_xgboost_and_w_b.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    This lesson tracks a classification pipeline from raw data to trained model.
    We use Weights & Biases Artifacts to record dataset lineage, Tables to inspect sample rows,
    the XGBoost callback to log training, and Sweeps to compare hyperparameters.

    The example uses a subset of the
    [Vehicle Loan Default Prediction dataset](https://www.kaggle.com/sneharshinde/ltfs-av-data)
    and preprocessing code shared by Scotiabank in the
    [RAPIDS credit-scorecard example](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard).
    This notebook demonstrates experiment tracking for a classifier; it does not
    construct or validate a points-based lending scorecard.

    ## 1. Set Up and Authenticate

    Open the notebook using its declared dependencies:

    ```bash
    uvx marimo edit --sandbox credit_scorecards_with_xgboost_and_w_b.py
    ```

    > On macOS, XGBoost also needs OpenMP. If importing XGBoost reports a missing
    `libomp.dylib`, install it with `brew install libomp` and restart.

    The next cell calls `wandb.login()` and sets the project. Use your team name
    for `entity`, or `None` for your default entity. Configure `WANDB_API_KEY`
    in your environment or molab Secrets; get a key from
    [Weights & Biases](https://wandb.ai/authorize). Terminal-backed sessions can also prompt
    for a key.

    Running the code cells downloads the source Artifact, uploads derived
    datasets and a model, then runs a small hyperparameter sweep.
    """)
    return


@app.cell
def _():
    entity = "wandb"
    project = "vehicle_loan_default"
    if not wandb.login():
        raise RuntimeError("Configure Weights & Biases credentials before running the lesson.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Choose the Dataset

    The next cell identifies the source Artifact, ID columns, and target column,
    and creates a temporary working directory. The target is `loan_default`;
    ID columns are excluded from the model's predictors by preprocessing.

    The helper below loads `one_hot_encode_data` from the Artifact's
    `data_utils.py`. The preprocessing run records that function's source so
    the resulting dataset can be traced to its transformation.
    """)
    return


@app.cell
def _():
    source_artifact_path = "morgan/credit_scorecard/vehicle_loan_defaults:latest"
    id_columns = ["UniqueID"]
    target_column = "loan_default"
    data_dir = Path(tempfile.mkdtemp(prefix="wandb-credit-scorecard-"))
    return data_dir, id_columns, source_artifact_path, target_column


@app.function
def load_preprocessing_function(module_dir):
    module_path = next(Path(module_dir).rglob("data_utils.py"), None)
    if module_path is None:
        raise FileNotFoundError("The source Artifact did not contain data_utils.py.")
    spec = importlib.util.spec_from_file_location("credit_scorecard_data_utils", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.one_hot_encode_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Preprocess and Log a Dataset Artifact

    The next cell starts `preprocessing_run`, retrieves the raw dataset with
    `use_artifact()`, and downloads its files. We read the CSV and use the
    supplied one-hot encoding function to produce numeric predictors.

    We save the processed CSV, attach it to a new `wandb.Artifact()`, and call
    `preprocessing_run.log_artifact()`. Consuming and producing Artifacts in
    the same run records the transformation in Weights & Biases's lineage graph.
    `wait()` ensures the upload finishes before the next stage fetches it.

    For data stored in an existing S3 or GCS bucket, you can instead construct
    an Artifact with `artifact.add_reference("s3://bucket/path")`. That records
    an external reference rather than uploading another copy.
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
        one_hot_encode_data = load_preprocessing_function(raw_dataset_dir)
        raw_csv_path = next(raw_dataset_dir.rglob("vehicle_loans_subset.csv"))
        raw_data = pd.read_csv(raw_csv_path)
        processed_data, predictor_columns = one_hot_encode_data(raw_data, id_columns, target_column)
        predictor_columns = list(predictor_columns)

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
    ## 4. Split the Data and Record Lineage

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
    ## 5. Inspect Data with Weights & Biases Tables

    The next cell starts `table_run` and records the split Artifact as an input.
    `wandb.Table(dataframe=...)` turns a reproducible sample of up to 1,000
    training rows into an interactive table. `table_run.log()` saves it under
    `processed_dataset` for sorting, filtering, and inspection in Weights & Biases.
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
    ## 6. Prepare Predictors and Model Settings

    The next cell separates predictors from the binary target, computes the
    training default rate, and defines the XGBoost parameters. We use histogram
    trees on CPU; a supported CUDA environment can add `device="cuda"`.

    The early-stopping setting monitors the last evaluation metric, log loss,
    on validation data. With only 25 trees, this is a short demonstration run.
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
    early_stopping_rounds = 40
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
    ## 7. Define Evaluation Metrics

    The following helper calculates AUC, log loss, and KS statistics for the
    baseline and sweep models. It uses the full training and validation sets.
    Validation accuracy uses a threshold of 0.5.

    These are descriptive model metrics. They do not establish calibration,
    fairness, or suitability for lending decisions. The Weights & Biases calls that record
    these values appear in the training cells below.
    """)
    return


@app.function
def evaluate_classifier(classifier, X_train, y_train, X_validation, y_validation):
    train_probabilities = classifier.predict_proba(X_train)[:, 1]
    validation_probabilities = classifier.predict_proba(X_validation)[:, 1]
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_train, train_probabilities)
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
    ## 8. Train and Log the Baseline

    The next cell starts `training_run`, records the split Artifact as an input,
    and logs hyperparameters in its configuration. Passing
    `WandbCallback(log_model=True)` to `XGBClassifier` logs evaluation metrics
    and feature importance during training, and uploads the trained booster as
    a model Artifact.

    After fitting, we add evaluation values to `training_run.summary` and log
    validation AUC plus a ROC curve. The curve receives probabilities for both
    classes. For large datasets, we subsample only the visualization; the numeric
    evaluation metrics use all validation rows.
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
        resolved_entity = training_run.entity

    print(f"Validation AUC: {baseline_metrics['val_auc']:.4f}")
    return baseline_metrics, resolved_entity


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 9. Define a Hyperparameter Sweep

    The next cell specifies random search over learning rate, gamma, minimum
    child weight, and early-stopping rounds. The objective is to maximize
    `val_auc`, which each trial explicitly logs.

    Sweeps automate comparison on this fixed validation split. Repeated tuning
    can overfit the validation set; an independent test set is needed to assess
    the selected model's performance.
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
            "early_stopping_rounds": {"values": [10, 20, 30, 40]},
        },
    }
    return (sweep_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 10. Define One Sweep Trial

    The next cell defines the zero-argument function the Weights & Biases agent calls.
    `wandb.init()` starts `sweep_run` with baseline defaults; Weights & Biases replaces
    the searched values with the settings selected for that trial.

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
    early_stopping_rounds,
    model_parameters,
    split_artifact_path,
    y_train,
    y_validation,
):
    def sweep_train():
        with wandb.init(
            job_type="sweep",
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
    ## 11. Create the Sweep and Start an Agent

    The next cell registers the configuration with `wandb.sweep()` and stores
    its ID. The following cell calls `wandb.agent()` with `sweep_train` and a
    limit of five trials. Lower `count` for a quicker experiment.

    Each run closes through its context manager before the next trial starts.
    The dashboard compares the selected parameters against validation AUC.
    """)
    return


@app.cell
def _(baseline_metrics, project, resolved_entity, sweep_config):
    print(f"Baseline validation AUC: {baseline_metrics['val_auc']:.4f}")
    sweep_id = wandb.sweep(sweep=sweep_config, entity=resolved_entity, project=project)
    sweep_url = f"https://wandb.ai/{resolved_entity}/{project}/sweeps/{sweep_id}"
    print(f"View the sweep: {sweep_url}")
    return (sweep_id,)


@app.cell
def _(project, resolved_entity, sweep_id, sweep_train):
    wandb.agent(sweep_id, function=sweep_train, entity=resolved_entity, project=project, count=5)
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## 12. Explore the Experiment Records

    Open the baseline run to inspect its metrics,
    feature importance, ROC curve, and saved model. The Artifacts graph connects
    the source dataset, processed dataset, split, and training run.

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
