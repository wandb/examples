# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=1.26",
#     "scikit-learn>=1.5",
#     "wandb>=0.18",
#     "xgboost>=2.1,<3.3",  # Preserve the tree-versus-linear booster notebook.
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Weights & Biases Sweeps with XGBoost",
)

with app.setup:
    from urllib.request import urlopen

    import marimo as mo
    from numpy import loadtxt
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier
    import wandb

    DATASET_URL = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )


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

    # Using Weights & Biases Sweeps with XGBoost

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/using-w-b-sweeps-with-xgboost/using_w_b_sweeps_with_xgboost.py/server)

    Hyperparameter sweeps search combinations of model settings and record
    their performance. In This notebook, compare XGBoost's tree and linear
    boosters while exploring learning rate, tree depth, and row sampling.

    There are three steps:

    1. **Define the sweep:** choose the search strategy, parameters, and metric.
    2. **Initialize the sweep:** register the configuration with `wandb.sweep`.
    3. **Run an agent:** call `wandb.agent` with a training function and run count.

    This notebook runs on a CPU. On macOS, install the OpenMP runtime with
    `brew install libomp` before starting the notebook if it is not installed.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _notify_submission(_value):
        mo.status.toast(
            "Credentials submitted",
            "Run the next cell to verify them with Weights & Biases.",
        )

    wandb_api_form = (
        mo.md("{api_key}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="Forge API key",
                full_width=True,
            )
        )
        .form(
            submit_button_label="Connect to Weights & Biases",
            bordered=True,
            on_change=_notify_submission,
        )
    )

    wandb_api_form
    return (wandb_api_form,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After submitting the API key, run the following cell to verify the
    credentials and load your available Weights & Biases entities.
    """)
    return


@app.cell(hide_code=True)
def _():
    from wandb.sdk.lib.service.service_connection import (
        WandbApiFailedError as _WandbApiFailedError,
    )

    mo.stop(wandb_api_form.value is None)

    _submission = wandb_api_form.value
    _api_key = _submission["api_key"].strip()

    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
            verify=True,
            prompt=False,
        )
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    try:
        _api = wandb.Api(timeout=15)
        _viewer = _api.viewer
    except (wandb.errors.Error, _WandbApiFailedError, ValueError):
        mo.stop(
            True,
            mo.callout(
                "Weights & Biases verified the credentials but could not retrieve your account entities.",
                kind="danger",
            ),
        )

    available_entities = sorted({
        _value
        for _value in (
            getattr(_viewer, "username", None),
            getattr(_viewer, "entity", None),
            *_viewer.teams,
        )
        if _value
    })
    mo.stop(
        not available_entities,
        mo.callout(
            "Weights & Biases authenticated successfully, but no personal or team entities were found.",
            kind="danger",
        ),
    )

    mo.callout(
        mo.md("✓ Credentials verified. Specify your team entity below."),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `XGBoost-sweeps`.

    Then select **Use this Weights & Biases target**.

    > If the project doesn't exist, Weights & Biases creates it when the first run starts, provided you have write access.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _validate_wandb_target(value):
        if value is None:
            return None
        if not value["entity"]:
            return "Select a Weights & Biases entity."
        if not value["project"].strip():
            return "Enter a project name."
        return None

    def _notify_wandb_submission(_value):
        mo.status.toast(
            "Weights & Biases target selected",
            "Run the next cell to confirm the destination.",
        )

    wandb_setup_form = (
        mo.md("{entity}\n\n{project}")
        .batch(
            entity=mo.ui.dropdown(
                options=available_entities,
                value=available_entities[0],
                label="Weights & Biases entity",
                searchable=True,
                full_width=True,
            ),
            project=mo.ui.text(
                value="XGBoost-sweeps",
                label="Project",
                full_width=True,
            ),
        )
        .form(
            submit_button_label="Use this Weights & Biases target",
            validate=_validate_wandb_target,
            on_change=_notify_wandb_submission,
            bordered=True,
        )
    )

    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After submitting the target form, run the next cell to confirm the selected
    destination. Weights & Biases verifies project write access when the first run starts.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

    mo.callout(
        mo.md(
            f"✓ Using `{entity}/{project}`. Weights & Biases will confirm write access "
            "when the first run starts."
        ),
        kind="success",
    )
    return entity, project




@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define the sweep

    A sweep configuration can be a [Python dictionary or YAML file](https://docs.wandb.ai/models/sweeps/define-sweep-configuration).
    Here we use a dictionary with three main fields:

    - **Metric:** the logged metric to optimize and whether to maximize or
      minimize it. Our training function logs validation accuracy as `accuracy`.
    - **Method:** `random` samples combinations, `grid` tries every combination,
      and `bayes` uses previous results to choose promising settings.
    - **Parameters:** the values, ranges, or distributions to search.

    Both boosters use `learning_rate`. The [tree parameters](https://xgboost.readthedocs.io/en/stable/parameter.html#parameters-for-tree-booster)
    `max_depth` and `subsample` affect only `gbtree`, so the training function
    applies them only to tree models. Their sampled values are still recorded
    in every run’s config. Filter to tree runs when comparing those parameters.
    """)
    return


@app.cell
def _():
    sweep_config = {
        "method": "random",  # Try "grid" or "bayes".
        "metric": {
            "name": "accuracy",
            "goal": "maximize",
        },
        "parameters": {
            "booster": {"values": ["gbtree", "gblinear"]},
            "max_depth": {"values": [3, 6, 9, 12]},
            "learning_rate": {"values": [0.1, 0.05, 0.2]},
            "subsample": {"values": [1, 0.5, 0.3]},
        },
    }
    return (sweep_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Initialize the sweep

    `wandb.sweep` registers this configuration with Weights & Biases and returns a sweep ID.
    The sweep service supplies parameter choices to agents and tracks the
    metrics their runs report.

    Running this cell creates a new sweep. To add trials to the same sweep,
    rerun only the agent cell below with the existing `sweep_id`.
    """)
    return


@app.cell
def _(entity, project, sweep_config):
    sweep_id = wandb.sweep(sweep=sweep_config, entity=entity, project=project)
    return (sweep_id,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prepare the dataset

    Download the Pima Indians Diabetes CSV once and reuse it across trials.
    The first eight columns are features and the last column is the binary
    target. Each trial uses the same split seed and held-out fraction so its
    accuracy can be compared with other trials.

    Because the sweep uses this held-out split to select hyperparameters, it
    serves as validation data. Use a separate test set for a final evaluation.
    """)
    return


@app.cell
def _():
    with urlopen(DATASET_URL, timeout=30) as dataset_response:
        dataset = loadtxt(dataset_response, delimiter=",")
    X, y = dataset[:, :8], dataset[:, 8]
    return X, y


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define the training function

    The agent calls `train()` once per trial. Inside the function:

    - `wandb.init(config=config_defaults)` starts a run. Values supplied by the
      sweep override the defaults.
    - `run.config` provides that trial's hyperparameters.
    - `run.log({"accuracy": accuracy})` reports the metric named in the sweep
      configuration.

    The context manager finishes each run before the next trial begins.
    """)
    return


@app.cell
def _(X, y):
    def train():
        config_defaults = {
            "booster": "gbtree",
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1,
            "seed": 117,
            "test_size": 0.33,
        }

        with wandb.init(config=config_defaults) as run:
            config = run.config
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                test_size=config.test_size,
                random_state=config.seed,
            )

            model_params = {
                "booster": config.booster,
                "learning_rate": config.learning_rate,
                "random_state": config.seed,
                "n_jobs": 2,
            }
            if config.booster == "gbtree":
                model_params.update(
                    max_depth=config.max_depth,
                    subsample=config.subsample,
                )
            model = XGBClassifier(**model_params)
            model.fit(X_train, y_train)

            predictions = model.predict(X_valid)
            accuracy = accuracy_score(y_valid, predictions)
            print(f"Validation accuracy: {accuracy:.1%}")
            run.log({"accuracy": accuracy})

    return (train,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run the sweep agent

    The agent requests hyperparameters from Weights & Biases and calls `train` for each trial.
    `count` limits how many runs this agent launches. Without a limit or a
    manual stop, a random-search agent can keep requesting trials. Start with a smaller `run_count` for a quick check.

    Agents on other machines can join the same sweep using its ID, entity,
    project, and the same training function and dataset. The count applies to
    each agent, rather than the whole sweep.
    """)
    return


@app.cell
def _(entity, project, sweep_id, train):
    run_count = 25
    wandb.agent(
        sweep_id,
        function=train,
        count=run_count,
        entity=entity,
        project=project,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify the XGBoost sweep

    Open the sweep link, which ends in `/sweeps/<sweep-id>`. Add `booster`,
    `learning_rate`, `max_depth`, `subsample`, and `accuracy` to the runs table.
    Confirm that every agent run has sampled values in **Config** and a final
    accuracy value.

    Inspect parallel coordinates, scatter plots, and parameter importance. Filter
    to `booster="gbtree"` before interpreting `max_depth` or `subsample`, because
    they do not affect the linear booster. Treat importance estimates from a small
    trial count as leads. See the [Sweeps guide](https://docs.coreweave.com/products/wandb/sweeps/visualize-sweep-results).

    ## Next steps

    Narrow learning-rate choices or switch to Bayesian search, then create a new
    sweep so the search space has its own record. Keep the split fixed. After
    selecting by validation accuracy, evaluate the chosen model on a separate test
    set instead of continuing to tune against the same held-out examples.
    """)
    return


if __name__ == "__main__":
    app.run()
