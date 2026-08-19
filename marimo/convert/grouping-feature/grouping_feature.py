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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/Grouping_Feature.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Weights & Biases Grouping
    From your script, use grouping to organize individual runs into larger experiments. This is useful for distributed training and cross validation.

    In `wandb.init()`:
    - **group**: the first level of organization, usually this is your unique experiment name
    - **job_type**: the second level of grouping, this is often `train`, `eval`, `optimizer`, `rollout` etc.

    **Links**
    - [Documentation](https://docs.wandb.ai/library/grouping)
    - [Example project](https://wandb.ai/carey/group-demo?workspace=user-carey)
    - [Example dedicated group page](https://wandb.ai/carey/group-demo/groups/exp_5?workspace=user-carey)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qqq
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example, I'm setting the experiment index up front, and then incrementing it every time I re-run the cell below.
    """)
    return


@app.cell
def _():
    # Set experiment index (for demo purposes)
    experiment_index = 1
    return (experiment_index,)


@app.cell
def _(experiment_index):
    # Simulate launching multiple different jobs that log to the same experiment
    import wandb
    import math
    import random
    for i in range(5):
        job_type = 'rollout'
        if i == 2:
            job_type = 'eval'
        if i == 3:
            job_type = 'eval2'
        if i == 4:
            job_type = 'optimizer'
        wandb.init(project='group-demo', group='exp_' + str(experiment_index), job_type=job_type)
        for j in range(100):
            acc = 0.1 * (math.log(1 + j + 0.1) + random.random())
            val_acc = 0.1 * (math.log(1 + j + 2) + random.random() + random.random())  # Set group and job_type to see auto-grouping in the UI
            if j % 10 == 0:
                wandb.log({'acc': acc, 'val_acc': val_acc})
        wandb.finish()
    # I'm incrementing this so you can re-run this cell and get another experiment
    # grouped in the W&B UI
    experiment_index_1 = experiment_index + 1  # Using this to mark a run complete in a notebook context
    return


if __name__ == "__main__":
    app.run()
