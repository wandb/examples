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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Customize_metric_logging_with_define_metric.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{define_metric} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![wandb-logo](http://wandb.me/logo-im-png)

    <!--- @wandbcode{define_metric} -->

    # Define your custom metrics with `define_metric`

    Use `define_metric` to set custom x-axes or capture the min and max values of your metrics.

    For more details, [see the docs](http://wandb.me/define-metric-docs).
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -Uq
    return


@app.cell
def _():
    import wandb
    import random

    return random, wandb


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom X Axis

    Here's how to set a custom step so your charts have a custom x-axis:
    ```python
    wandb.define_metric("my-metric", step_metric='my-custom-x-axis')
    ```
    """)
    return


@app.cell
def _(random, wandb):
    random.seed(1)
    wandb.init(project='define-metric-demo', notes='custom step')
    # Initalize a new run
    wandb.define_metric('custom_step')
    wandb.define_metric('validation/loss', step_metric='custom_step')
    # Define the custom x axis metric
    for _i in range(10):
        _log_dict = {'train/loss': 1 / (_i + 1), 'custom_step': _i ** 2, 'validation/loss': 1 / (_i + 1)}
    # Define which metrics to plot against that x-axis
        wandb.log(_log_dict)
    # Use this in the context of a jupyter notebook to mark a run finished
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell above and click on the link that prints out to see the dashboard. It will look something like this:
    - `train_loss` is plotted against the standard W&B internal step
    - `custom_step` is plotted too, so you can see how it increases over the W&B internal step
    - `validation_loss` is plotted against the `custom_step`, replacing the default with the x-axis as the W&B internal step

    ![](https://i.imgur.com/jGcoAIV.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Min/Max of Metrics

    Each time you call `wandb.log()` to log a metric, you're writing to run `history`. The run `summary` saves a single value for each metric. By default, `summary` captures the final step of `history`. So if you log accuracy for 100 steps, your `history` will have all 100 steps and your `summary` will have just the final value for accuracy.

    Sometimes, you want to get the _best_ value instead of the _last_ value for a metric and save that to `summary`. That's where `define_metric` comes in.

    Here, you can set `summary=` to either `max` or `min`.

    ```python
    wandb.define_metric("my-metric", summary="max")
    ```
    """)
    return


@app.cell
def _(random, wandb):
    random.seed(1)
    wandb.init(project='define-metric-demo', notes='min of loss, max of acc')
    # Start a new run
    wandb.define_metric('loss', summary='min')
    wandb.define_metric('acc', summary='max')
    # For loss, capture the min value from history in summary
    for _i in range(10):
        _log_dict = {'loss': random.uniform(0, 1 / (_i + 1)), 'acc': random.uniform(1 / (_i + 1), 1)}
    # For acc, capture the max value from history in summary
        wandb.log(_log_dict)
    # Simulate a training loop where we're logging metrics
    # Mark the run as finished, useful in the context of Jupyter notebooks
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run the cell above and click on the project page link that prints out to see the dashboard. It will look something like this:
    - `acc.max` is visible in the sidebar, saved in the run summary
    - `loss.min` is visible in the sidebar, saved in the run summary

    You can see the summary values in the Project Page Table. Here I've pinned two columns in the sidebar, which you can see on the left.
    ![](https://i.imgur.com/VaO9w25.png)
    """)
    return


if __name__ == "__main__":
    app.run()
