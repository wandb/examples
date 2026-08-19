# /// script
# dependencies = ["ray", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/raytune/tune-wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using Weights & Biases with Tune

    [Weights & Biases](https://www.wandb.ai/) (Wandb) is a tool for experiment
    tracking, model optimizaton, and dataset versioning. It is very popular
    in the machine learning and data science community for its superb visualization
    tools.

    Ray Tune currently offers two lightweight integrations for Weights & Biases.
    One is the {ref}`WandbLoggerCallback <tune-wandb-logger>`, which automatically logs
    metrics reported to Tune to the Wandb API.

    The other one is the {ref}`@wandb_mixin <tune-wandb-mixin>` decorator, which can be
    used with the function API. It automatically
    initializes the Wandb API with Tune's training information. You can just use the
    Wandb API like you would normally do, e.g. using `wandb.log()` to log your training
    process.

    ## Running A Weights & Biases Example

    In the following example we're going to use both of the above methods, namely the `WandbLoggerCallback` and
    the `wandb_mixin` decorator to log metrics.
    Let's start with a few crucial imports:
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: ray[tune] wandb !pip install -Uq ray[tune] wandb
    return


@app.cell
def _():
    import numpy as np
    import wandb

    from ray import air, tune
    from ray.air import session
    from ray.tune import Trainable
    from ray.air.callbacks.wandb import WandbLoggerCallback
    from ray.tune.integration.wandb import (
        WandbTrainableMixin,
        wandb_mixin,
    )

    return (
        Trainable,
        WandbLoggerCallback,
        WandbTrainableMixin,
        air,
        np,
        session,
        tune,
        wandb,
        wandb_mixin,
    )


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, let's define an easy `objective` function (a Tune `Trainable`) that reports a random loss to Tune.
    The objective function itself is not important for this example, since we want to focus on the Weights & Biases
    integration primarily.
    """)
    return


@app.cell
def _(np, session):
    def objective(config, checkpoint_dir=None):
        for i in range(30):
            loss = config["mean"] + config["sd"] * np.random.randn()
            session.report({"loss": loss})

    return (objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given that you provide an `api_key_file` pointing to your Weights & Biases API key, you cna define a
    simple grid-search Tune run using the `WandbLoggerCallback` as follows:
    """)
    return


@app.cell
def _(WandbLoggerCallback, air, objective, tune):
    def tune_function(api_key_file):
        """Example for using a WandbLoggerCallback with the function API"""
        tuner = tune.Tuner(
            objective,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
            ),
            run_config=air.RunConfig(
                callbacks=[
                    WandbLoggerCallback(api_key_file=api_key_file, project="Wandb_example")
                ],
            ),
            param_space={
                "mean": tune.grid_search([1, 2, 3, 4, 5]),
                "sd": tune.uniform(0.2, 0.8),
            },
        )
        results = tuner.fit()

        return results.get_best_result().config

    return (tune_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To use the `wandb_mixin` decorator, you can simply decorate the objective function from earlier.
    Note that we also use `wandb.log(...)` to log the `loss` to Weights & Biases as a dictionary.
    Otherwise, the decorated version of our objective is identical to its original.
    """)
    return


@app.cell
def _(np, session, wandb, wandb_mixin):
    @wandb_mixin
    def decorated_objective(config, checkpoint_dir=None):
        for i in range(30):
            loss = config["mean"] + config["sd"] * np.random.randn()
            session.report({"loss": loss})
            wandb.log(dict(loss=loss))

    return (decorated_objective,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With the `decorated_objective` defined, running a Tune experiment is as simple as providing this objective and
    passing the `api_key_file` to the `wandb` key of your Tune `config`:
    """)
    return


@app.cell
def _(objective, tune):
    def tune_decorated(api_key_file):
        """Example for using the @wandb_mixin decorator with the function API"""
        tuner = tune.Tuner(
            objective,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
            ),
            param_space={
                "mean": tune.grid_search([1, 2, 3, 4, 5]),
                "sd": tune.uniform(0.2, 0.8),
                "wandb": {"api_key_file": api_key_file, "project": "Wandb_example"},
            },
        )
        results = tuner.fit()

        return results.get_best_result().config

    return (tune_decorated,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, you can also define a class-based Tune `Trainable` by using the `WandbTrainableMixin` to define your objective:
    """)
    return


@app.cell
def _(Trainable, WandbTrainableMixin, np, wandb):
    class WandbTrainable(WandbTrainableMixin, Trainable):
        def step(self):
            for i in range(30):
                loss = self.config["mean"] + self.config["sd"] * np.random.randn()
                wandb.log({"loss": loss})
            return {"loss": loss, "done": True}

    return (WandbTrainable,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Running Tune with this `WandbTrainable` works exactly the same as with the function API.
    The below `tune_trainable` function differs from `tune_decorated` above only in the first argument we pass to
    `Tuner()`:
    """)
    return


@app.cell
def _(WandbTrainable, tune):
    def tune_trainable(api_key_file):
        """Example for using a WandTrainableMixin with the class API"""
        tuner = tune.Tuner(
            WandbTrainable,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
            ),
            param_space={
                "mean": tune.grid_search([1, 2, 3, 4, 5]),
                "sd": tune.uniform(0.2, 0.8),
                "wandb": {"api_key_file": api_key_file, "project": "Wandb_example"},
            },
        )
        results = tuner.fit()

        return results.get_best_result().config

    return (tune_trainable,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since you may not have an API key for Wandb, we can _mock_ the Wandb logger and test all three of our training
    functions as follows.
    If you do have an API key file, make sure to set `mock_api` to `False` and pass in the right `api_key_file` below.
    """)
    return


@app.cell
def _(
    WandbLoggerCallback,
    WandbTrainable,
    decorated_objective,
    tune_decorated,
    tune_function,
    tune_trainable,
):
    import tempfile
    from unittest.mock import MagicMock
    mock_api = True
    api_key_file = '~/.wandb_api_key'
    if mock_api:
        WandbLoggerCallback._logger_process_cls = MagicMock
        decorated_objective.__mixins__ = tuple()
        WandbTrainable._wandb = MagicMock()
        wandb_1 = MagicMock()
        temp_file = tempfile.NamedTemporaryFile()
        temp_file.write(b'1234')
        temp_file.flush()  # noqa: F811
        api_key_file = temp_file.name
    tune_function(api_key_file)
    tune_decorated(api_key_file)
    tune_trainable(api_key_file)
    if mock_api:
        temp_file.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This completes our Tune and Wandb walk-through.
    In the following sections you can find more details on the API of the Tune-Wandb integration.

    ## Tune Wandb API Reference

    ### WandbLoggerCallback

    (tune-wandb-logger)=

    ```{eval-rst}
    .. autoclass:: ray.air.callbacks.wandb.WandbLoggerCallback
       :noindex:
    ```

    ### Wandb-Mixin

    (tune-wandb-mixin)=

    ```{eval-rst}
    .. autofunction:: ray.tune.integration.wandb.wandb_mixin
       :noindex:
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
