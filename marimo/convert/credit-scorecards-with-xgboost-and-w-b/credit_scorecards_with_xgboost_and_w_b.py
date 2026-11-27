# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "dill>=0.3.8,<1",
#     "marimo>=0.24.2",
#     "numpy>=1.26,<3",
#     "pandas>=2.1,<3",
#     "scikit-learn>=1.2,<2",
#     "scipy>=1.11,<2",
#     "wandb>=0.19.10,<1",
#     "xgboost>=2.0,<4",
# ]
# ///
"""Build an interpretable vehicle-loan scorecard with XGBoost and W&B."""

import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="Vehicle loan scorecards with XGBoost and W&B",
)

with app.setup:
    import importlib.util
    import sys
    import tempfile
    from pathlib import Path

    from dill import detect
    from dill.source import getsource
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy.stats import ks_2samp
    from sklearn import metrics, model_selection

    import wandb

    pd.set_option("display.max_columns", None)
    data_root = Path(tempfile.gettempdir()) / "wandb-credit-scorecard"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/credit-scorecards-with-xgboost-and-w-b/credit_scorecards_with_xgboost_and_w_b.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" /><br>

    <img src="https://wandb.me/mini-diagram" width="600" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Vehicle Loan Default Prediction with XGBoost
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this notebook we'll train an XGBoost model to classify whether submitted loan applications will default or not. Using boosting algorithms such as XGBoost increases the performance of a loan assessment, whilst retaining interpretability for internal Risk Management functions as well as external regulators.

    This notebook is based on a talk from Nvidia GTC21 by Paul Edwards at ScotiaBank who [presented](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31327/) how XGBoost can be used to construct more performant credit scorecards that remain interpretable. They also kindly [shared sample code](https://github.com/rapidsai-community/showcase/tree/main/event_notebooks/GTC_2021/credit_scorecard) which we will use throughout this notebook, credit to [Stephen Denton](stephen.denton@scotiabank.com) from Scotiabank for sharing this code publicly.

    **[Click here](https://wandb.ai/morgan/credit_scorecard) to view and interact with a live W&B Dashboard built with this notebook.**
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## In this notebook

    In this notebook we'll cover how Weights & Biases enables regulated entities to
    - **Track and version** their data ETL pipelines (locally or in cloud services such as S3 and GCS)
    - **Track experiment results** and store trained models
    - **Visually inspect** multiple evaluation metrics
    - **Optimize performance** with hyperparameter sweeps
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Track Experiments and Results**

    We will track all of the training hyperparameters and output metrics in order to generate an Experiments Dashboard like the one below:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ![W&B experiment dashboard with validation metrics and ROC curves](https://raw.githubusercontent.com/wandb/examples/main/marimo/convert/credit-scorecards-with-xgboost-and-w-b/assets/experiment-dashboard.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Run a Hyperparameter Sweep to Find the Best HyperParameters**

    Weights and Biases also enables you to do hyperparameter sweeps with [W&B Sweeps](https://docs.wandb.ai/models/sweeps/). See the docs for a full guide to advanced hyperparameter search options.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ![W&B sweep parallel-coordinates chart](https://raw.githubusercontent.com/wandb/examples/main/marimo/convert/credit-scorecards-with-xgboost-and-w-b/assets/sweep-parallel-coordinates.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup

    The setup cell imports the packages declared in this notebook's script
    metadata. Runtime data is written under the operating system's temporary
    directory, not into the repository.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Data
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### AWS S3, Google Cloud Storage and W&B Artifacts
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ![Amazon Web Services and Google Cloud logos](https://raw.githubusercontent.com/wandb/examples/main/marimo/convert/credit-scorecards-with-xgboost-and-w-b/assets/cloud-storage-logos.jpeg)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Weights and Biases **Artifacts** enable you to log end-to-end training pipelines to ensure your experiments are always reproducible.

    Data privacy is critical to Weights & Biases, so Artifacts can refer to files in private cloud storage such as Amazon S3 or Google Cloud Storage. Self-Managed W&B deployments are also available.

    By default, W&B stores artifact files in a private Google Cloud Storage bucket located in the United States. All files are encrypted at rest and in transit. For sensitive files, we recommend a private W&B installation or the use of reference artifacts.

    ### Artifacts Reference Example
    **Create an artifact with the S3/GCS metadata**

    The artifact only consists of metadata about the S3/GCS object such as its ETag, size, and version ID (if object versioning is enabled on the bucket).

    ```
    run = wandb.init()
    artifact = wandb.Artifact('mnist', type='dataset')
    artifact.add_reference('s3://my-bucket/datasets/mnist')
    run.log_artifact(artifact)
    ```

    **Download the artifact locally when needed**

    W&B will use the metadata recorded when the artifact was logged to retrieve the files from the underlying bucket.

    ```
    artifact = run.use_artifact('mnist:latest', type='dataset')
    artifact_dir = artifact.download()
    ```

    See [Create an artifact](https://docs.wandb.ai/models/artifacts/construct-an-artifact) for more on using external references and configuring credentials.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to
    use `WANDB_API_KEY` from the molab Secrets panel or credentials already
    stored in this runtime. A fresh molab session does not inherit credentials
    from your computer.

    The entity is the team name in `wandb.ai/<entity>/<project>`. Leave it
    blank to use your default entity. Changing an unsubmitted field does not
    authenticate or create any W&B objects.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="W&B API key (optional)",
                placeholder="Paste a key or use configured credentials",
                full_width=True,
            ),
            entity=mo.ui.text(
                label="W&B entity or team (optional)",
                placeholder="Leave blank to use your default entity",
                full_width=True,
            ),
            project=mo.ui.text(
                value="vehicle_loan_default",
                label="W&B project",
                full_width=True,
            ),
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Submit the form above before running the artifact pipeline."),
            kind="info",
        ),
    )

    _submitted = wandb_login_form.value
    _api_key = _submitted["api_key"].strip()
    _project = _submitted["project"].strip()
    mo.stop(
        not _project,
        mo.callout(mo.md("Enter a W&B project name and submit again."), kind="danger"),
    )
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False
    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key or "
                "molab Secrets configuration and submit again."
            ),
            kind="danger",
        ),
    )

    _entity = _submitted["entity"].strip() or wandb.Api().default_entity
    mo.stop(
        not _entity,
        mo.callout(
            mo.md(
                "W&B did not return a default entity. Enter a team entity and "
                "submit the authentication form again."
            ),
            kind="danger",
        ),
    )
    wandb_connection = {"entity": _entity, "project": _project}
    mo.callout(
        mo.md(f"Connected to W&B as `{_entity}` for project `{_project}`."),
        kind="success",
    )
    return (wandb_connection,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Vehicle Loan Dataset

    We will be using a simplified version of the [Vehicle Loan Default Prediction dataset](https://www.kaggle.com/sneharshinde/ltfs-av-data) from L&T which has been stored in W&B Artifacts.
    """)
    return


@app.cell
def _():
    # specify a folder to save the data, a new folder will be created if it doesn't exist
    data_dir = data_root
    id_vars = ['UniqueID']
    targ_var = 'loan_default'
    return data_dir, id_vars, targ_var


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Create function to pickle functions
    """)
    return


@app.function
def function_to_string(fn):
    return getsource(detect.code(fn))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Download Data from W&B Artifacts

    We will download our dataset from W&B Artifacts. First we need to create a W&B run object, which we will use to download the data. Once the data is downloaded it will be one-hot encoded. This processed data will then be logged to the same W&B as a new Artifact. By logging to the W&B that downloaded the data, we tie this new Artifact to the raw dataset Artifact
    """)
    return


@app.cell
def _(wandb_connection):
    preprocess_button = mo.ui.run_button(
        label="Download, preprocess, and log data"
    )
    mo.vstack(
        [
            mo.md(
                "This creates a preprocessing run and a processed-dataset "
                f"Artifact in `{wandb_connection['entity']}/"
                f"{wandb_connection['project']}`."
            ),
            preprocess_button,
        ]
    )
    return (preprocess_button,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Download the subset of the vehicle loan default data from W&B, this contains `train.csv` and `val.csv` files as well as some utils files.
    """)
    return


@app.function
def load_data_utils(module_dir):
    data_utils_path = next(Path(module_dir).rglob("data_utils.py"), None)
    if data_utils_path is None:
        raise FileNotFoundError("The dataset Artifact did not contain data_utils.py")

    spec = importlib.util.spec_from_file_location(
        "credit_scorecard_data_utils", data_utils_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {data_utils_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.describe_data_g_targ, module.one_hot_encode_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### One-Hot Encode the Data
    """)
    return


@app.function
def preprocess_and_log_data(connection, data_dir, id_vars, targ_var):
    data_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=connection["project"],
        entity=connection["entity"],
        job_type="preprocess-data",
        reinit="create_new",
        dir=str(data_dir),
    ) as run:
        ARTIFACT_PATH = "morgan/credit_scorecard/vehicle_loan_defaults:latest"
        dataset_art = run.use_artifact(ARTIFACT_PATH, type="dataset")
        dataset_dir = Path(dataset_art.download(root=str(data_dir / "raw")))
        describe_data_g_targ, one_hot_encode_data = load_data_utils(dataset_dir)

        # Load data into Dataframe
        dataset_path = next(dataset_dir.rglob("vehicle_loans_subset.csv"))
        dataset = pd.read_csv(dataset_path)

        # One Hot Encode Data
        dataset, p_vars = one_hot_encode_data(dataset, id_vars, targ_var)

        # Save Preprocessed data
        processed_data_path = data_dir / "proc_ds.csv"
        dataset.to_csv(processed_data_path, index=False)

        # Create a new artifact for the processed data, including the function that created it, to Artifacts
        processed_ds_art = wandb.Artifact(
            name="vehicle_defaults_processed",
            type="processed_dataset",
            description="One-hot encoded dataset",
            metadata={
                "preprocessing_fn": function_to_string(one_hot_encode_data)
            },
        )

        # Attach our processed data to the Artifact
        processed_ds_art.add_file(str(processed_data_path))

        # Log this Artifact to the current wandb run
        logged_artifact = run.log_artifact(processed_ds_art)
        logged_artifact.wait()
        result = {
            "artifact_path": logged_artifact.qualified_name,
            "data_utils_dir": str(dataset_dir),
            "p_vars": list(p_vars),
            "run_url": run.url,
        }
    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Log Processed Data to W&B Artifacts
    """)
    return


@app.cell
def _(data_dir, id_vars, preprocess_button, targ_var, wandb_connection):
    mo.stop(
        not preprocess_button.value,
        mo.callout(
            mo.md(
                "Click **Download, preprocess, and log data** to run this stage."
            ),
            kind="info",
        ),
    )
    preprocess_result = preprocess_and_log_data(
        wandb_connection, data_dir, id_vars, targ_var
    )
    return (preprocess_result,)


@app.cell(hide_code=True)
def _(preprocess_result):
    mo.callout(
        mo.md(
            f"Logged `{preprocess_result['artifact_path']}`. "
            f"[Open the preprocessing run]({preprocess_result['run_url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Get Train/Validation Split

    Here we show an alternative pattern for how to create a wandb run object. In the cell below, the code to split the dataset is wrapped with a call to `wandb.init() as run`.

    Here we will:

    - Start a wandb run
    - Download our one-hot-encoded dataset from Artifacts
    - Do the Train/Val split and log the params used in the split
    - Log the new `trndat` and `valdat` datasets to Artifacts
    - Finish the wandb run automatically
    """)
    return


@app.cell(hide_code=True)
def _(preprocess_result):
    split_button = mo.ui.run_button(label="Create and log train/validation split")
    mo.vstack(
        [
            mo.md(
                f"Use `{preprocess_result['artifact_path']}` as the input to "
                "the next lineage step."
            ),
            split_button,
        ]
    )
    return (split_button,)


@app.function
def split_and_log_data(connection, data_dir, preprocess_result, targ_var):
    split_dir = data_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=connection["project"],
        entity=connection["entity"],
        job_type="train-val-split",
        reinit="create_new",
        dir=str(data_dir),
    ) as run:
        dataset_art = run.use_artifact(
            preprocess_result["artifact_path"], type="processed_dataset"
        )
        dataset_dir = Path(dataset_art.download(root=str(split_dir / "processed")))
        processed_data_path = next(dataset_dir.rglob("proc_ds.csv"))
        dataset = pd.read_csv(processed_data_path)

        # Set Split Params
        test_size = 0.25
        random_state = 42

        # Log the splilt params
        run.config.update({"test_size": test_size, "random_state": random_state})

        # Do the Train/Val Split
        trndat, valdat = model_selection.train_test_split(
            dataset,
            test_size=test_size,
            random_state=random_state,
            stratify=dataset[targ_var],
        )
        print(f"Train dataset size: {trndat[targ_var].value_counts()} \n")
        print(f"Validation dataset size: {valdat[targ_var].value_counts()}")

        # Save split datasets
        train_path = split_dir / "train.csv"
        val_path = split_dir / "val.csv"
        trndat.to_csv(train_path, index=False)
        valdat.to_csv(val_path, index=False)

        # Create a new artifact for the processed data, including the function that created it, to Artifacts
        split_ds_art = wandb.Artifact(
            name="vehicle_defaults_split",
            type="train-val-dataset",
            description="Processed dataset split into train and validation",
            metadata={"test_size": test_size, "random_state": random_state},
        )

        # Attach our processed data to the Artifact
        split_ds_art.add_file(str(train_path))
        split_ds_art.add_file(str(val_path))

        # Log the Artifact
        logged_artifact = run.log_artifact(split_ds_art)
        logged_artifact.wait()
        describe_data_g_targ, _ = load_data_utils(
            preprocess_result["data_utils_dir"]
        )
        trndict = describe_data_g_targ(trndat, targ_var)
        result = {
            "artifact_path": logged_artifact.qualified_name,
            "run_url": run.url,
            "train": trndat,
            "validation": valdat,
            "training_description": trndict,
        }
    return result


@app.cell
def _(data_dir, preprocess_result, split_button, targ_var, wandb_connection):
    mo.stop(
        not split_button.value,
        mo.callout(
            mo.md("Click **Create and log train/validation split** to continue."),
            kind="info",
        ),
    )
    split_result = split_and_log_data(
        wandb_connection, data_dir, preprocess_result, targ_var
    )
    return (split_result,)


@app.cell(hide_code=True)
def _(split_result):
    mo.callout(
        mo.md(
            f"Logged `{split_result['artifact_path']}`. "
            f"[Open the split run]({split_result['run_url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Inspect Training Dataset
    Get an overview of the training dataset
    """)
    return


@app.cell
def _(split_result):
    trndat = split_result["train"]
    valdat = split_result["validation"]
    trndict = split_result["training_description"]
    trndat.head()
    return trndat, trndict, valdat


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log Dataset with W&B Tables

    With W&B Tables you can log, query, and analyze tabular data that contains rich media such as images, video, audio and more. With it you can understand your datasets, visualize model predictions, and share insights; learn more in the [W&B Tables guide](https://docs.wandb.ai/models/tables).
    """)
    return


@app.cell(hide_code=True)
def _():
    log_table_button = mo.ui.run_button(label="Log a sample W&B Table")
    log_table_button
    return (log_table_button,)


@app.function
def log_dataset_table(connection, data_dir, trndat):
    # Create a wandb run, with an optional "log-dataset" job type to keep things tidy
    with wandb.init(
        project=connection["project"],
        entity=connection["entity"],
        job_type="log-dataset",
        reinit="create_new",
        dir=str(data_dir),
    ) as run:  # config is optional here
        # Create a W&B Table and log 1000 random rows of the dataset to explore
        table = wandb.Table(
            dataframe=trndat.sample(min(1000, len(trndat)), random_state=42)
        )
        # Log the Table to your W&B workspace
        run.log({"processed_dataset": table})
        run_url = run.url
    # Close the wandb run
    return {"rows": len(table.data), "run_url": run_url}


@app.cell
def _(data_dir, log_table_button, trndat, wandb_connection):
    mo.stop(
        not log_table_button.value,
        mo.callout(
            mo.md("Click **Log a sample W&B Table** to create the table run."),
            kind="info",
        ),
    )
    table_result = log_dataset_table(wandb_connection, data_dir, trndat)
    return (table_result,)


@app.cell(hide_code=True)
def _(table_result):
    mo.callout(
        mo.md(
            f"Logged {table_result['rows']} rows. "
            f"[Open the table run]({table_result['run_url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Modelling
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Fit the XGBoost Model

    We will now fit an XGBoost model to classify whether a vehicle loan application
    will result in a default.

    Local macOS environments need the OpenMP runtime used by the XGBoost wheel.
    Install it once with `brew install libomp`, then restart this notebook. Hosted
    Linux environments generally provide the required runtime with the wheel.

    #### Training on GPU
    XGBoost 2.x uses the `device` parameter for accelerator selection. To train
    with CUDA, add the following alongside `tree_method='hist'`:

    ```
    'device': 'cuda'
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 1) Initialise a W&B Run

    Training, metric logging, and model-Artifact logging begin only when you
    click the button below.
    """)
    return


@app.cell(hide_code=True)
def _(trndict):
    train_model_button = mo.ui.run_button(label="Train and log the XGBoost model")
    mo.vstack(
        [
            mo.md(
                f"The training set base rate is `{trndict['base_rate']:.6f}`."
            ),
            train_model_button,
        ]
    )
    return (train_model_button,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 2) Setup and Log the Model Parameters
    """)
    return


@app.cell
def _(trndict):
    base_rate = round(trndict['base_rate'], 6)
    early_stopping_rounds = 40
    return base_rate, early_stopping_rounds


@app.cell
def _(base_rate):
    bst_params = {
            'objective': 'binary:logistic'
            , 'base_score': base_rate
            , 'gamma': 1               ## def: 0
            , 'learning_rate': 0.1     ## def: 0.1
            , 'max_depth': 3
            , 'min_child_weight': 100  ## def: 1
            , 'n_estimators': 25
            , 'nthread': 24
            , 'random_state': 42
            , 'reg_alpha': 0
            , 'reg_lambda': 0          ## def: 1
            , 'eval_metric': ['auc', 'logloss']
            , 'tree_method': 'hist'  # add `device='cuda'` to train on CUDA
        }
    return (bst_params,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The training transaction logs the XGBoost parameters to the W&B run config.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 3) Let's select the data for train/validation
    """)
    return


@app.cell
def _(valdat):
    valdat
    return


@app.cell
def _(preprocess_result, targ_var, trndat, valdat):
    ## Extract target column as a series
    y_trn = trndat.loc[:,targ_var].astype(int)
    y_val = valdat.loc[:,targ_var].astype(int)
    p_vars = preprocess_result["p_vars"]
    return p_vars, y_trn, y_val


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 4) Fit the model, log results to W&B and save model to W&B Artifacts

    The current W&B XGBoost integration uses `WandbCallback`. It logs model
    configuration, evaluation metrics, feature importance, and—because
    `log_model=True` below—the trained booster as a W&B Artifact. See the
    [XGBoost integration guide](https://docs.wandb.ai/models/integrations/xgboost).
    """)
    return


@app.function
def train_and_log_model(
    connection,
    data_dir,
    bst_params,
    early_stopping_rounds,
    p_vars,
    trndat,
    valdat,
    y_trn,
    y_val,
):
    import xgboost as xgb
    from wandb.integration.xgboost import WandbCallback

    with wandb.init(
        project=connection["project"],
        entity=connection["entity"],
        job_type="train-model",
        dir=str(data_dir),
        config={
            **bst_params,
            "early_stopping_rounds": early_stopping_rounds,
        },
    ) as run:
        # Initialize the XGBoostClassifier with the WandbCallback
        xgbmodel = xgb.XGBClassifier(
            **bst_params,
            callbacks=[WandbCallback(log_model=True)],
            early_stopping_rounds=early_stopping_rounds,
        )

        # Train the model
        xgbmodel.fit(
            trndat[p_vars],
            y_trn,
            eval_set=[(valdat[p_vars], y_val)],
        )

        bstr = xgbmodel.get_booster()

        # Get train and validation predictions
        trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:, 1]
        valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:, 1]

        # Log additional Train metrics
        false_positive_rate, true_positive_rate, _thresholds = metrics.roc_curve(
            y_trn, trnYpreds
        )
        run.summary["best_iteration"] = bstr.best_iteration
        run.summary["train_ks_stat"] = max(
            true_positive_rate - false_positive_rate
        )
        run.summary["train_auc"] = metrics.auc(
            false_positive_rate, true_positive_rate
        )
        _safe_train_predictions = np.clip(trnYpreds, 1e-7, 1 - 1e-7)
        run.summary["train_log_loss"] = -(
            y_trn * np.log(_safe_train_predictions)
            + (1 - y_trn) * np.log(1 - _safe_train_predictions)
        ).sum() / len(y_trn)

        # Log additional Validation metrics
        ks_stat, ks_pval = ks_2samp(
            valYpreds[y_val == 1], valYpreds[y_val == 0]
        )
        run.summary["val_ks_2samp"] = ks_stat
        run.summary["val_ks_pval"] = ks_pval
        run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
        run.summary["val_acc_0.5"] = metrics.accuracy_score(
            y_val, np.where(valYpreds >= 0.5, 1, 0)
        )
        _safe_validation_predictions = np.clip(valYpreds, 1e-7, 1 - 1e-7)
        run.summary["val_log_loss"] = -(
            y_val * np.log(_safe_validation_predictions)
            + (1 - y_val) * np.log(1 - _safe_validation_predictions)
        ).sum() / len(y_val)

        # Log the ROC curve to W&B
        valYpreds_2d = np.array(
            [1 - valYpreds, valYpreds]
        )  # W&B expects a 2d array
        y_val_arr = y_val.values
        stride = max(1, int(np.ceil(len(y_val_arr) / 10_000)))
        valYpreds_2d = valYpreds_2d[:, ::stride]
        y_val_arr = y_val_arr[::stride]
        run.log(
            {
                "ROC_Curve": wandb.plot.roc_curve(
                    y_val_arr,
                    valYpreds_2d.T,
                    labels=["no_default", "loan_default"],
                    classes_to_plot=[1],
                )
            }
        )
        result = {
            "model": xgbmodel,
            "run_url": run.url,
            "validation_auc": float(run.summary["val_auc"]),
        }
    return result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 5) Log Additional Train and Evaluation Metrics to W&B
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### 6) Log the ROC Curve To W&B
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Finish the W&B Run
    """)
    return


@app.cell
def _(
    bst_params,
    data_dir,
    early_stopping_rounds,
    p_vars,
    train_model_button,
    trndat,
    valdat,
    wandb_connection,
    y_trn,
    y_val,
):
    mo.stop(
        not train_model_button.value,
        mo.callout(
            mo.md("Click **Train and log the XGBoost model** to run training."),
            kind="info",
        ),
    )

    # Validate XGBoost before starting a W&B run so a missing native runtime does
    # not leave a failed remote run behind.
    try:
        import xgboost as _xgb  # noqa: F401
    except (ImportError, OSError, ValueError) as _error:
        _xgboost_error = str(_error)
    else:
        _xgboost_error = None

    _macos_help = (
        " On macOS, run `brew install libomp`, then restart this notebook."
        if "libomp.dylib" in (_xgboost_error or "")
        else ""
    )
    mo.stop(
        _xgboost_error is not None,
        mo.callout(
            mo.md(
                "**XGBoost could not load its native runtime.**"
                f"{_macos_help}\n\n"
                "Training has not started, so this attempt did not create a W&B run."
            ),
            kind="danger",
        ),
    )

    model_result = train_and_log_model(
        wandb_connection,
        data_dir,
        bst_params,
        early_stopping_rounds,
        p_vars,
        trndat,
        valdat,
        y_trn,
        y_val,
    )
    return (model_result,)


@app.cell(hide_code=True)
def _(model_result):
    mo.callout(
        mo.md(
            f"Validation AUC: `{model_result['validation_auc']:.4f}`. "
            f"[Open the completed training run]({model_result['run_url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we've trained a single model, let's optimize its performance by running a hyperparameter sweep.

    ## HyperParameter Sweep

    Weights and Biases also enables you to do hyperparameter sweeps with [W&B Sweeps](https://docs.wandb.ai/models/sweeps/). See the [configuration guide](https://docs.wandb.ai/models/sweeps/define-sweep-configuration) for advanced options.

    **[Click Here](https://wandb.ai/morgan/credit_score_sweeps/sweeps/iuppbs45)** to check out the results of a 1000 run sweep generated using this notebook
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ![W&B sweep parallel-coordinates chart](https://raw.githubusercontent.com/wandb/examples/main/marimo/convert/credit-scorecards-with-xgboost-and-w-b/assets/sweep-parallel-coordinates.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define the Sweep Config
    First we define the hyperparameters to sweep over as well as the type of sweep to use. We'll do a random search over the learning rate, gamma, minimum child weight, and early-stopping rounds, maximizing validation AUC.
    """)
    return


@app.cell
def _():
    sweep_config = {
      "method" : "random",
      "metric": {
        "name": "val_auc",
        "goal": "maximize"
      },
      "parameters" : {
        "learning_rate" :{
          "min": 0.001,
          "max": 1.0
        },
        "gamma" :{
          "min": 0.001,
          "max": 1.0
        },
        "min_child_weight" :{
          "min": 1,
          "max": 150
        },
        "early_stopping_rounds" :{
          "values" : [10, 20, 30, 40]
        },
      }
    }
    return (sweep_config,)


@app.cell(hide_code=True)
def _(model_result, wandb_connection):
    create_credit_sweep_button = mo.ui.run_button(label="Create W&B sweep")
    mo.vstack(
        [
            mo.md(
                "The single-model run has finished. Create the sweep in "
                f"`{wandb_connection['entity']}/{wandb_connection['project']}` "
                f"after reviewing [that run]({model_result['run_url']})."
            ),
            create_credit_sweep_button,
        ]
    )
    return (create_credit_sweep_button,)


@app.cell
def _(create_credit_sweep_button, sweep_config, wandb_connection):
    mo.stop(
        not create_credit_sweep_button.value,
        mo.callout(mo.md("Click **Create W&B sweep** to continue."), kind="info"),
    )
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        entity=wandb_connection["entity"],
        project=wandb_connection["project"],
    )
    sweep_result = {
        "id": sweep_id,
        "entity": wandb_connection["entity"],
        "project": wandb_connection["project"],
        "url": (
            f"https://wandb.ai/{wandb_connection['entity']}/"
            f"{wandb_connection['project']}/sweeps/{sweep_id}"
        ),
    }
    return (sweep_result,)


@app.cell(hide_code=True)
def _(sweep_result):
    mo.callout(
        mo.md(
            f"Created sweep `{sweep_result['id']}`. "
            f"[Open the live Sweep dashboard]({sweep_result['url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define the Training Function
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Then we define the function that will train our model using these hyperparameters. Note that `job_type='sweep'` when initialising the run, so that we can easily filter out these runs from our main workspace if we need to
    """)
    return


@app.function
def make_sweep_train(
    base_rate,
    p_vars,
    trndat,
    valdat,
    y_trn,
    y_val,
):
    def train():
        import xgboost as xgb
        from wandb.integration.xgboost import WandbCallback

        with wandb.init(job_type="sweep") as run:

            bst_params = {
                'objective': 'binary:logistic'
                , 'base_score': base_rate
                , 'gamma': run.config['gamma']
                , 'learning_rate': run.config['learning_rate']
                , 'max_depth': 3
                , 'min_child_weight': run.config['min_child_weight']
                , 'n_estimators': 25
                , 'nthread': 24
                , 'random_state': 42
                , 'reg_alpha': 0
                , 'reg_lambda': 0          ## def: 1
                , 'eval_metric': ['auc', 'logloss']
                , 'tree_method': 'hist'
            }

            # Initialize the XGBoostClassifier with the WandbCallback
            xgbmodel = xgb.XGBClassifier(**bst_params,
                                         callbacks=[WandbCallback()],
                                         early_stopping_rounds=run.config['early_stopping_rounds'])

            # Train the model
            xgbmodel.fit(trndat[p_vars], y_trn,
                         eval_set=[(valdat[p_vars], y_val)])

            bstr = xgbmodel.get_booster()

            # Log booster metrics
            run.summary["best_iteration"] = bstr.best_iteration

            # Get train and validation predictions
            trnYpreds = xgbmodel.predict_proba(trndat[p_vars])[:,1]
            valYpreds = xgbmodel.predict_proba(valdat[p_vars])[:,1]

            # Log additional Train metrics
            false_positive_rate, true_positive_rate, _thresholds = metrics.roc_curve(y_trn, trnYpreds)
            run.summary['train_ks_stat'] = max(true_positive_rate - false_positive_rate)
            run.summary['train_auc'] = metrics.auc(false_positive_rate, true_positive_rate)
            _safe_train_predictions = np.clip(trnYpreds, 1e-7, 1 - 1e-7)
            run.summary['train_log_loss'] = -(y_trn * np.log(_safe_train_predictions) + (1-y_trn) * np.log(1-_safe_train_predictions)).sum() / len(y_trn)

            # Log additional Validation metrics
            ks_stat, ks_pval = ks_2samp(valYpreds[y_val==1], valYpreds[y_val==0])
            run.summary["val_ks_2samp"] = ks_stat
            run.summary["val_ks_pval"] = ks_pval
            run.summary["val_auc"] = metrics.roc_auc_score(y_val, valYpreds)
            run.summary["val_acc_0.5"] = metrics.accuracy_score(y_val, np.where(valYpreds >= 0.5, 1, 0))
            _safe_validation_predictions = np.clip(valYpreds, 1e-7, 1 - 1e-7)
            run.summary["val_log_loss"] = -(y_val * np.log(_safe_validation_predictions)
                                                 + (1-y_val) * np.log(1-_safe_validation_predictions)).sum() / len(y_val)
            run.log({"val_auc": run.summary["val_auc"]})

    return train


@app.cell
def _(base_rate, p_vars, trndat, valdat, y_trn, y_val):
    credit_sweep_train = make_sweep_train(
        base_rate, p_vars, trndat, valdat, y_trn, y_val
    )
    return (credit_sweep_train,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Run the Sweeps Agent
    """)
    return


@app.cell(hide_code=True)
def _(credit_sweep_train, sweep_result):
    _training_function = credit_sweep_train.__name__
    sweep_agent_form = (
        mo.md(
            f"Run an agent for sweep `{sweep_result['id']}` using "
            f"`{_training_function}`.\n\n{{count}}"
        )
        .batch(
            count=mo.ui.number(
                start=1,
                stop=5,
                step=1,
                value=5,
                label="Maximum sweep trials",
            )
        )
        .form(submit_button_label="Start sweep agent and training", bordered=True)
    )
    sweep_agent_form
    return (sweep_agent_form,)


@app.cell(hide_code=True)
def _(sweep_agent_form, sweep_result):
    mo.stop(
        sweep_agent_form.value is None,
        mo.callout(
            mo.md(
                "Submit the form above to start the requested W&B training runs."
            ),
            kind="info",
        ),
    )
    agent_request = {
        "count": int(sweep_agent_form.value["count"]),
        "sweep": sweep_result.copy(),
    }
    return (agent_request,)


@app.cell
def _(agent_request, credit_sweep_train):
    _sweep = agent_request["sweep"]
    # number of runs to execute
    wandb.agent(
        _sweep["id"],
        function=credit_sweep_train,
        entity=_sweep["entity"],
        project=_sweep["project"],
        count=agent_request["count"],
    )
    agent_result = {
        "count": agent_request["count"],
        "url": _sweep["url"],
    }
    return (agent_result,)


@app.cell(hide_code=True)
def _(agent_result):
    mo.callout(
        mo.md(
            f"Completed up to {agent_result['count']} trials. "
            f"[Inspect the sweep results]({agent_result['url']})."
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## W&B already in your favorite ML library

    Weights and Biases has integrations in all of your favourite ML and Deep Learning libraries such as:

    - Pytorch Lightning
    - Keras
    - Hugging Face
    - JAX
    - Fastai
    - XGBoost
    - Sci-Kit Learn
    - LightGBM

    **See [W&B integrations for details](https://docs.wandb.ai/models/integrations/)**
    """)
    return


if __name__ == "__main__":
    app.run()
