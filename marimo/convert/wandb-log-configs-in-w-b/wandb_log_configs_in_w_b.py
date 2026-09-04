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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    # Quickstart
    Use [Weights & Biases](https://wandb.ai)
    for machine learning experiment tracking, dataset versioning, and project collaboration.

    <div><img /></div>

    <img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    <div><img /></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get started, just `pip install` the package and log using `wandb.login()`
    If this is your first time using `wandb`, you'll need to sign up. It's easy!
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -Uq wandb
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
    # What's a `config` for?

    Set [`wandb.config`](https://docs.wandb.ai/guides/track/config)
    once at the beginning of your script to save your training configuration: hyperparameters, input settings like dataset name or model type, and include any other independent variables or metadata for your experiments.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why does that matter?

    This is useful for analyzing your experiments and reproducing your work in the future. You'll be able to group by `config` values in our web interface, comparing the settings of different runs and seeing how these affect the output.

    > Note that output metrics or dependent variables (like loss and accuracy) should be saved with `wandb.log` instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How do I set up a `config`?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Your `config` should be set just once at the beginning of your training experiment.

    But workflows differ, so we offer a number of ways to set up your config.

    Let's look at all the ways you can create and send the config dictionary to the Dashboard!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting the `config` at `init`ialization

    The best time to set the `config` values is when you call [`wandb.init`](https://docs.wandb.ai/guides/track/launch),
    by passing a dictionary as the `config` keyword argument.
    """)
    return


@app.cell
def _(wandb):
    wandb.init(project='config_example', config={'dataset': 'CelebA', 'type': 'baseline'})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Head to the [Run page](https://docs.wandb.ai/ref/app/pages/run-page)
    linked in the output of `wandb.init`
    and head to the [Overview tab](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab)
    (top of the list of panels on the left-most side of the screen).
    You'll see a "Config" section that looks like this:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://i.imgur.com/nAC9KEF.png" width="450"/>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You give us a (possibly nested) dictionary as your `config`, and we'll flatten the names using dots in our backend.

    > _Side Note_: We recommend that you avoid using dots in your config variable names, and use a dash or underscore instead. Once you've created your `config` dictionary, if your script accesses `wandb.config` keys below the root, use the dictionary access syntax, `["key"]["foo"]`, instead of the attribute access syntax, `config.key.foo`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding to the `config` by hand
    You can add more parameters to the `config` later if you want:
    """)
    return


@app.cell
def _(wandb):
    wandb.config.epochs = 4
    wandb.config["batch_size"] = 32
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, your Config section on the dashboard has been updated:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://i.imgur.com/cnvEuSR.png" width="450"/>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding to the `config` with `argparse`

    `config` is a dictionary-like object, and it can be built from lots of dictionary-like objects.

    For example, you can pass in the arguments object produced by `argparse`.
    [`argparse`](https://docs.python.org/3/library/argparse.html), short for `arg`ument `parse`r, is a standard library module in Python 3.2 and above that makes it easy to write scripts that take advantage of all the flexibility and power of command line arguments. And it's Pythonic!

    This is especially convenient for tracking results from scripts that are launched from the command line.
    """)
    return


@app.cell
def _(wandb):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_per_gpu', type=int, default=8,
                        help='input batch size for training (default: 8)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.1)')

    args = parser.parse_args(args=[])
    wandb.config.update(args)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's the updated Config panel:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://i.imgur.com/zWSpGNy.png" width=450>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Updating the `config` with the API

    What if your run has finished, but you realized you forgot to log something?

    Never fear, you can always use the
    [public API](https://docs.wandb.ai/ref/python/public-api)
    to update your `config`
    (or anything else about your run!)
    at any time. You just need to know the details of the `run` you want to update.
    """)
    return


@app.cell
def _(wandb):
    api = wandb.Api()

    # pulling the relevant info automatically from the run object
    # this can also be found on the website
    username = wandb.run.entity
    project = wandb.run.project
    run_id = wandb.run.id

    run = api.run(f"{username}/{project}/{run_id}")
    run.config["bar"] = 32
    run.update()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's what the final Config panel looks like:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://i.imgur.com/mxmIbyK.png" width=450>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using `config` for great good!
    The `config` parameters are useful for performing grouping, filtering, and aggregating on your experiments and their results.

    ### The examples below come from the project [here](https://wandb.ai/wandb/DistHyperOpt).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Filtering Runs
    Filter tab allows you to display the runs that quality one or more conditions. These conditions can be formed by applying relational operators to any of the parameters logged in the `config` file.

    [Our example project](https://wandb.ai/wandb/DistHyperOpt) compares various hyper-parameter tuning methods and has more than 80 runs. Each run has a "Job Type" logged in the `config` which corresponds

    Let's say you want to visualize only the ones that are generated by a particular tuning algorithm, like Population Based Traing (`pbt`). You can do that by applying a filter on "Job Type".

    Run the cell below to see this in action!
    """)
    return


@app.cell
def _():
    from IPython import display

    display.YouTubeVideo("aSMXwOSPtJE", rel=0, width=450)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grouping Runs
    You can group your experiments in the dashboard of your project based on a particular column from `config`. A common use case for this would be grouping sub-experiments within a larger project.

    Our runs are grouped based on "Job Type". The Group tab is located next to the Filter Tab. You can group your runs by any parameter present in the config.

    ![Imgur](https://i.imgur.com/gTLKRP7.png?1)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Parallel Coordinates Chart

    Often, the main thing we want to do with a group of Runs is make comparisons.

    The W&B Dashboard includes a Chart type for exactly this purpose:
    the Parallel Coordinates chart.

    A Parallel Coordinates chart represents each Run in the group as a line.
    This line passes through as many of the `config` values
    or logged metrics as you like,
    and is colored by its value on a single metric.
    This lets you take in, at a glance,
    which hyperparameter configurations were most and least successful.
    See the example below.

    Head to a [group of Runs in this project](https://wandb.ai/wandb/DistHyperOpt/groups/dcgan_train)
    and build a Parallel Coordinates chart like the one pictured below
    by
    1. clicking the + sign in the top-right corner, aligned with "Charts",
    2. selecting "Parallel Coordinates" from the available Charts, and
    3. adding the columns in the image, in order.

    ![Imgur](https://i.imgur.com/ugrpq9K.png)
    """)
    return


if __name__ == "__main__":
    app.run()
