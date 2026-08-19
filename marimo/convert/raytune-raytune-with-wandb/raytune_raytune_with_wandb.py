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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/raytune/RayTune_with_wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{raytune} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{raytune} -->

    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

    <div><img /></div>

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    <div><img /></div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌞 Ray/Tune and 🏋️‍♀️ Weights & Biases

    Both Weights and Biases and Ray/Tune are built for scale and handle millions of models every month for teams doing some of the most cutting-edge deep learning research.

    [W&B](https://wandb.com) is a toolkit with everything you need to track, reproduce, and gain insights from your models easily; [Ray/Tune](https://docs.ray.io/en/latest/tune/) provides a simple interface for scaling and running distributed experiments.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🤝 They're a natural match! 🤝

    Here's just a few reasons why our community likes Ray/Tune –

    * **Simple distributed execution**: Ray/Tune makes it easy to scale all the way from a single node on a laptop, through to multiple GPUs, and up to multiple nodes on multiple machines.
    * **State-of-the-art algorithms**: Ray/Tune has tested implementations of a huge number of potent scheduling algorithms including
    [Population-Based Training](https://docs.ray.io/en/latest/tune/tutorials/tune-advanced-tutorial.html),
    [ASHA](https://docs.ray.io/en/master/tune/tutorials/tune-tutorial.html#early-stopping-with-asha),
    and
    [HyperBand](https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#hyperband-tune-schedulers-hyperbandscheduler)
    * **Method agnostic**: Ray/Tune works across deep learning frameworks (including PyTorch, Keras, Tensorflow, and PyTorchLightning) and with other ML methods like gradient-boosted trees (XGBoost, LightGBM)
    * **Fault-tolerance**: Ray/Tune is built on top of Ray, providing tolerance for failed runs out of the box.

    This Colab demonstrates how this integration works for a simple grid search over two hyperparameters. If you've got any questions about the details,
    check out
    [our documentation](https://docs.wandb.com/library/integrations/ray-tune)
    or the
    [documentation for Ray/Tune](https://docs.ray.io/en/master/tune/api_docs/integration.html#weights-and-biases-tune-integration-wandb).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    W&B integrates with `ray.tune` by offering two lightweight standalone integrations:

    1. For simple cases, `WandbLoggerCallback` automatically logs metrics reported to Tune to W&B, along with the configuration of the experiment, using Tune's [`logger` interface](https://docs.ray.io/en/latest/tune/api_docs/logging.html).
    2. The `@wandb_mixin` decorator gives you greater control over logging by letting you call `wandb.log` inside the decorated function, allowing you to [log custom metrics, plots, and other outputs, like media](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb).

    These methods can be used together or independently.

    The example below demonstrates how they can be used together.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧹 Running a hyperparameter sweep with W&B and Ray/Tune
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📥 Install, `import`, and set seeds
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's start by installing the libraries and importing everything we need.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: ray[tune] wandb !pip install -Uq ray[tune] wandb
    return


@app.cell
def _():
    import random
    import numpy as np
    from filelock import FileLock
    import tempfile
    from ray import train, tune
    from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb
    from ray.tune.schedulers import AsyncHyperBandScheduler
    from ray.train import Checkpoint
    import torch
    import torch.optim as optim
    import wandb

    return (
        AsyncHyperBandScheduler,
        Checkpoint,
        WandbLoggerCallback,
        np,
        optim,
        random,
        setup_wandb,
        tempfile,
        torch,
        train,
        tune,
        wandb,
    )


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll make use of Ray's handy [`mnist_pytorch` example code](https://github.com/ray-project/ray/blob/master/python/ray/tune/examples/mnist_pytorch.py).
    """)
    return


@app.cell
def _():
    from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test_func, train_func

    return ConvNet, get_data_loaders, test_func, train_func


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to make this experiment reproducible, we'll set the seeds for random number generators of various libraries used in this experiment.
    """)
    return


@app.cell
def _(np, random, torch):
    torch.backends.cudnn.deterministic = True
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed_all(2022)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤝 Integrating W&B with Ray/Tune
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we define our training process, decorated with `@wandb_mixin` so we can call `wandb.log` to log our custom metric
    (here, just the error rate; you might also [log media](https://docs.wandb.com/library/log#media), e.g. images from the validation set, captioned by the model predictions).

    When we execute our hyperparameter sweep below,
    this function will be called with a `config`uration dictionary
    that contains values for any hyperparameters.
    For simplicity, we only have two hyperparameters here:
    the learning rate and momentum value for accelerated SGD.
    """)
    return


@app.cell
def _(
    Checkpoint,
    ConvNet,
    get_data_loaders,
    optim,
    os,
    setup_wandb,
    tempfile,
    test_func,
    torch,
    train,
    train_func,
):
    def train_mnist(config):
         # Setup wandb
        wandb = setup_wandb(config)
        should_checkpoint = config.get("should_checkpoint", False)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train_loader, test_loader = get_data_loaders()
        model = ConvNet().to(device)

        optimizer = optim.SGD(
            model.parameters(), lr=config["lr"], momentum=config["momentum"]
        )
        while True:
            train_func(model, optimizer, train_loader, device)
            acc = test_func(model, test_loader, device)
            metrics = {"mean_accuracy": acc}

            # Report metrics (and possibly a checkpoint)
            if should_checkpoint:
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save(model.state_dict(), os.path.join(tempdir, "model.pt"))
                    train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
            else:
                train.report(metrics)
            # enables logging custom metrics using wandb.log()
            error_rate = 100 * (1 - acc)
            wandb.log({"error_rate": error_rate})

    return (train_mnist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🚀 Launching a Sweep with W&B and Ray/Tune
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We're now almost ready to call `tune.run` to launch our hyperparameter sweep!
    We just need to do three things:
    1. set up a `wandb.Run`,
    2. give the `WandbLoggerCallback` to `tune.run` so we can capture the output of `tune.report`, and
    3. set up our hyperparameter sweep.

    A `wandb.Run` is normally created by calling `wandb.init`.
    `tune` will handle that for you, you just need to pass
    the arguments as a dictionary
    (see [our documentation](https://docs.wandb.com/library/init) for details on `wandb.init`).
    At the bare minimum, you need to pass in a `project` name --
    sort of like a `git` repo name, but for your ML projects.

    In addition to holding arguments for `wandb.init`,
    that dictionary also has a few special keys, described in
    [the documentation for the `WandbLoggerCallback`](https://docs.ray.io/en/master/tune/tutorials/tune-wandb.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We handle steps 2 and 3 when we invoke `tune.run`.

    Step 2 is handled by passing in the `WandbLoggerCallback` class in a list
    to the `loggers` argument of `tune.run`.

    The setup of the hyperparameter sweep is handled by the
    `config` argument of `tune.run`.
    For the purposes of the integration,
    the most important part is that this is where we pass in the `wandb_init`
    dictionary.

    This is also where we configure the "meat" of the hyperparameter sweep:
    what are the hyperparameters we're sweeping over,
    and how do we choose their values.

    Here, we do a simple grid search, but
    [Ray/Tune provides lots of sophisticated options](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html).
    """)
    return


@app.cell
def _(AsyncHyperBandScheduler, WandbLoggerCallback, train, train_mnist, tune):
    # for early stopping
    sched = AsyncHyperBandScheduler()

    resources_per_trial = {"gpu": 1}  # set this for GPUs
    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=50,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={
                "mean_accuracy": 0.98,
                "training_iteration": 5,
            },
            callbacks=[WandbLoggerCallback(project="raytune-colab")]
        ),
        param_space={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        },
    )
    results = tuner.fit()
    return (results,)


@app.cell
def _(results):
    print("Best config is:", results.get_best_result().config)
    return


if __name__ == "__main__":
    app.run()
