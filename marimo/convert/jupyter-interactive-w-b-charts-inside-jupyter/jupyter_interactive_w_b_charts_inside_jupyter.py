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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/jupyter/Interactive_W&B_Charts_Inside_Jupyter.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use W&B without leaving Jupyter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Jupyter is the preferred development environment for many ML practitioners
    because it supports rapid experimentation and
    highly visual workflows (including creating charts).
    Plus tools like Google Colab, Kaggle Kernels, and Paperspace Gradient
    make it easy to share and collaborate on notebooks.

    Quick experiments, visualization, and collaboration
    are core values of W&B,
    so we've made it easy to use W&B inside Jupyter.

    In a nutshell, the steps are:

    1. Use one of two methods to get hold of a `Run`, `Sweep`, or `Report` object, depending on whether you're logging to a new experiment or analyzing an old one.
    2. `.display` the object to get a live dashboard beneath a cell.
    3. Interact with the dashboard: log new results, create charts, or review metadata.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's a (static) preview of one such dashboard:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://i.imgur.com/LhtnH1B.png" alt= "weights and biases jupyter integration" width="500" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Import, install, and log in
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qU
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
    ## Method 1: `display` and log to a live W&B `Run`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result of the last line of each cell in a Jupyter notebook is "displayed" automatically.

    Our W&B pages hook into this system:
    they are rendered as an interactive window.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we need to kick the run off with
    [`wandb.init`](https://docs.wandb.ai/guides/track/launch).
    """)
    return


@app.cell
def _(wandb):
    run = wandb.init(project="jupyter-projo",
                     config={"batch_size": 128,
                             "learning_rate": 0.01,
                             "dataset": "CIFAR-100"})
    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then we create an interactive dashboard of the size we want for the run and display it.
    """)
    return


@app.cell
def _(run):
    run.display(height=720)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Anything logged as part of this experiment (until you call `wandb.finish`)
    will be added to that chart.

    Run the cell below to watch the metrics stream in live!
    """)
    return


@app.cell
def _(wandb):
    import time

    for ii in range(30):
      wandb.log({"acc": 1 - 2 ** -ii, "loss": 2 ** -ii})
      time.sleep(0.5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Anything else you can do from the
    [Run Page](https://docs.wandb.ai/ref/app/pages/run-page)
    can be done here** --
    [edit a chart](https://docs.wandb.ai/ref/app/pages/run-page#charts-tab),
    create a shareable link to it,
    and send it to collaborator;
    review your [system metrics](https://docs.wandb.ai/ref/app/pages/run-page#system-tabs)
    or the
    [logs from the standard out](https://docs.wandb.ai/ref/app/pages/run-page#logs-tab)
    or the
    [datasets and models you've logged](https://docs.wandb.ai/ref/app/pages/run-page#artifacts-tab);
    check the
    [configuration metadata](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `wandb` also prints a URL. That URL points to [the webpage](https://docs.wandb.ai/ref/app/pages/run-page)
    where your run's results are stored -- nothing to worry about if your notebook crashes or your kernel dies, it's all there!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Finishing the run
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When you are done with your experiment,
    call `wandb.finish` to let us know there's nothing more to log.

    We'll print out a handy summary and history of your run,
    plus links to the webpages where all your run's information is stored.

    > **Hot Tip!** If you turn on [code saving](https://docs.wandb.ai/ref/app/features/panels/code) in your W&B [settings](https://wandb.ai/settings),
    we'll also save a copy of the notebook and its "session history": all the cells you ran, in order, in the state that you ran them in, with their outputs. Handy!
    """)
    return


@app.cell
def _(wandb):
    if wandb.run is not None:
      wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Method 2: `display` and analyze a finished W&B `Run`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Interaction with W&B dashboards for training runs
    isn't limited to watching information come in live
    from the comfort of a notebook interface.

    All of the information you log to or create within W&B
    is available in perpetuity and programmatically via the W&B
    [Public API](https://docs.wandb.ai/guides/track/public-api-guide).
    """)
    return


@app.cell
def _(wandb):
    api = wandb.Api()
    return (api,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this example, we'll take a look at the training run for a chess piece detector
    created using [YOLOv5](https://ultralytics.com/yolov5),
    which includes a [W&B integration](https://docs.wandb.ai/guides/integrations/yolov5).

    You can train your own with [this colab](http://wandb.me/yolo-chess).
    """)
    return


@app.cell
def _(api):
    team, _project, _run_id = ('wandb', 'yolo-chess', '33fp7u8d')
    run_1 = api.run(f'{team}/{_project}/{_run_id}')
    run_1.display(height=1080)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # But it's not just about `Run`s
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Anything you can do in a W&B workspace can be done from inside Jupyter
    if you have the URL for the workspace.

    That means that, without leaving Jupyter, you can use W&B to:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Interactively analyze data in [Tables](https://docs.wandb.ai/guides/data-vis)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And it doesn't have to be your own work -- it can be a `coworker`'s page as well.
    """)
    return


@app.cell
def _(api):
    coworker, _project, _run_id = ('stacey', 'model_iterz', '10x1nnh2')
    run_2 = api.run(f'{coworker}/{_project}/{_run_id}')
    run_2.display(height=720)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analyze the results of hyperparameter [Sweeps](https://docs.wandb.ai/guides/sweeps)
    """)
    return


@app.cell
def _(api):
    _entity, _project, sweep_id = ('charlesfrye', 'mnist-sweeps', 'n60n6wv1')
    sweep = api.sweep(f'{_entity}/{_project}/{sweep_id}')
    sweep.display(height=1080)  # you may need to zoom out to see the whole window!
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Share results in [Reports](https://docs.wandb.ai/guides/reports)
    """)
    return


@app.cell
def _():
    _entity, _project = ('charlesfrye', 'mnist-sweeps')
    report_name = 'Third-Pass-Trying-Different-Shapes--VmlldzoxNjY1NDk'
    # magic command not supported in marimo; please file an issue to add support
    # %wandb {entity}/{project}/reports/{report_name} -h 1024
    url = f'https://wandb.ai/{_entity}/{_project}/reports/{report_name}'
    return


if __name__ == "__main__":
    app.run()
