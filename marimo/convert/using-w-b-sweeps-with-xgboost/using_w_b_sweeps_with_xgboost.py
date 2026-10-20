# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=1.26",
#     "scikit-learn>=1.5",
#     "wandb>=0.18",
#     "xgboost>=2.1,<3.3",  # Preserve the tree-versus-linear booster lesson.
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="W&B Sweeps with XGBoost")

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
    # Using Weights & Biases Sweeps with XGBoost

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/using-w-b-sweeps-with-xgboost/using_w_b_sweeps_with_xgboost.py/server)

    Hyperparameter sweeps search combinations of model settings and record
    their performance. In this lesson, compare XGBoost's tree and linear
    boosters while exploring learning rate, tree depth, and row sampling.

    There are three steps:

    1. **Define the sweep:** choose the search strategy, parameters, and metric.
    2. **Initialize the sweep:** register the configuration with `wandb.sweep`.
    3. **Run an agent:** call `wandb.agent` with a training function and run count.

    This notebook runs on a CPU. On macOS, install the OpenMP runtime with
    `brew install libomp` before starting the notebook if it is not installed.

    ## Set up your experiment

    Get your [Weights & Biases API key](https://wandb.ai/authorize) and set
    `WANDB_API_KEY` in your runtime's environment or **Secrets** panel. Locally,
    you can also run `wandb login` in a terminal. A hosted runtime needs its
    own credentials.

    Set `entity` to your team slug, or leave it as `None` to use your default
    entity. The slug appears in `wandb.ai/<entity>/<project>`.
    This lesson needs an online connection to the Weights & Biases sweep service.
    """)
    return


@app.cell
def _():
    entity = None
    project = "XGBoost-sweeps"
    wandb.login()
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
    in every run's config; filter to tree runs when comparing those parameters.
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
    `count` limits how many runs this agent launches; without it, a random sweep
    can continue indefinitely. Start with a smaller `run_count` for a quick check.

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
    ## Visualize your results

    Click on the link returned by Weights & Biases in the previous cell. The link will look similar to the following:

    ```bash
    🧹 View sweep at https://wandb.ai/wandb/XGBoost-sweeps/sweeps/wna9genj
    ```

    Navigate to the **Sweep** panel. By default, Weights & Biases plots [parallel coordinates](https://docs.coreweave.com/products/wandb/app/features/panels/parallel-coordinates#parallel-coordinates), [scatter plot, and a parameter importance plot](https://docs.coreweave.com/products/wandb/sweeps/visualize-sweep-results#scatter-plot-and-parameter-importance).

    ## Verify and next steps

    Check that the agent's runs finished and that each run has an `accuracy`
    value and the sampled parameters in **Config**. Inspect the parallel
    coordinates and parameter importance panels, filtering to tree runs when
    evaluating `max_depth` and `subsample`.

    Try narrowing the learning-rate values or changing `method` to `bayes`.
    Run the sweep-creation cell again, then the agent cell, to compare the new
    sweep with the first one. See the [Sweeps documentation](https://docs.wandb.ai/models/sweeps)
    for more configuration options.
    """)
    return


if __name__ == "__main__":
    app.run()
