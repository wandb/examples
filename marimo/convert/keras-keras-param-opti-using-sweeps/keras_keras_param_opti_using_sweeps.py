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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Keras_param_opti_using_sweeps.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{sweeps-keras} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{sweeps-keras} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧹 Introduction to Hyperparameter Sweeps using W&B and Keras

    Searching through high dimensional hyperparameter spaces to find the most performant model can get unwieldy very fast. Hyperparameter sweeps provide an organized and efficient way to conduct a battle royale of models and pick the most accurate model. They enable this by automatically searching through combinations of hyperparameter values (e.g. learning rate, batch size, number of hidden layers, optimizer type) to find the most optimal values.

    In this tutorial we'll see how you can run sophisticated hyperparameter sweeps in 3 easy steps using Weights and Biases.

    ![](https://i.imgur.com/WVKkMWw.png)

    ## Sweeps: An Overview

    Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

    1. **Define the sweep:** we do this by creating a dictionary or a [YAML file](https://docs.wandb.com/library/sweeps/configuration) that specifies the parameters to search through, the search strategy, the optimization metric et all.

    2. **Initialize the sweep:** with one line of code we initialize the sweep and pass in the dictionary of sweep configurations:
    `sweep_id = wandb.sweep(sweep_config)`

    3. **Run the sweep agent:** also accomplished with one line of code, we call `wandb.agent()` and pass the `sweep_id` to run, along with a function that defines your model architecture and trains it:
    `wandb.agent(sweep_id, function=train)`

    And voila! That's all there is to running a hyperparameter sweep! In the notebook below, we'll walk through these 3 steps in more detail.

    We highly encourage you to fork this notebook so you can tweak the parameters,
    try out different models,
    or try a Sweep with your own dataset!

    ## Resources
    - [Sweeps docs →](https://docs.wandb.ai/sweeps)
    - [Launching from the command line →](https://www.wandb.com/articles/hyperparameter-tuning-as-easy-as-1-2-3)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🚀 Install, Import, and Log in
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 0️⃣: Install W&B
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -Uq wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1️⃣: Import W&B and Login
    """)
    return


@app.cell
def _():
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    return keras, layers, np, tf


@app.cell
def _():
    import wandb
    from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

    return WandbMetricsLogger, WandbModelCheckpoint, wandb


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Side note: If this is your first time using W&B or you are not logged in, the link that appears after running `wandb.login()` will take you to sign-up/login page. Signing up is as easy as a few clicks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 👩‍🍳 Prepare Dataset

    We will use MNIST directly from `keras.datasets`
    """)
    return


@app.cell
def _(keras, np):
    # Get the dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    num_classes = len(np.unique(y_train))
    input_shape = x_train.shape[-2:] + (1,)
    return input_shape, num_classes, x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    prepare data for training:
    - scale in [0,1]
    - transform targets to cateforical
    """)
    return


@app.cell
def _(keras, np, num_classes, x_test, x_train, y_test, y_train):
    # Scale
    x_train_1 = x_train / 255.0
    x_test_1 = x_test / 255.0
    x_train_1 = np.expand_dims(x_train_1, -1)
    # Make sure images have shape (28, 28, 1)
    x_test_1 = np.expand_dims(x_test_1, -1)
    y_train_1 = keras.utils.to_categorical(y_train, num_classes)
    # convert class vectors to binary class matrices
    y_test_1 = keras.utils.to_categorical(y_test, num_classes)
    return x_test_1, x_train_1, y_test_1, y_train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧠 Define the Model and Training Loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2️⃣ 🏗️ Build a Simple Classifier
    """)
    return


@app.cell
def _(input_shape, keras, layers, num_classes):
    def ConvNet(dropout=0.2):
        return keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model = ConvNet()

    model.summary()
    return (ConvNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3️⃣ Write a Training script
    """)
    return


@app.cell
def _(
    WandbMetricsLogger,
    WandbModelCheckpoint,
    tf,
    x_test_1,
    x_train_1,
    y_test_1,
    y_train_1,
):
    def get_optimizer(lr=0.001, optimizer='adam'):
        """Select optmizer between adam and sgd with momentum"""
        if optimizer.lower() == 'adam':
            return tf.keras.optimizers.Adam(learning_rate=lr)
        if optimizer.lower() == 'sgd':
            return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.1)

    def train(model, batch_size=64, epochs=10, lr=0.001, optimizer='adam', log_freq=10):
        tf.keras.backend.clear_session()
        model.compile(loss='categorical_crossentropy', optimizer=get_optimizer(lr, optimizer), metrics=['accuracy'])  # Compile model like you usually do.
        wandb_callbacks = [WandbMetricsLogger(log_freq=log_freq), WandbModelCheckpoint(filepath='my_model_{epoch:02d}')]
        model.fit(x_train_1, y_train_1, batch_size=batch_size, epochs=epochs, validation_data=(x_test_1, y_test_1), callbacks=wandb_callbacks)  # callback setup

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4️⃣ Define the Sweep

    Fundamentally, a Sweep combines a strategy for trying out a bunch of hyperparameter values with the code that evalutes them.
    Whether that strategy is as simple as trying every option
    or as complex as [BOHB](https://arxiv.org/abs/1807.01774),
    Weights & Biases Sweeps have you covered.
    You just need to _define your strategy_
    in the form of a [configuration](https://docs.wandb.com/sweeps/configuration).

    When you're setting up a Sweep in a notebook like this,
    that config object is a nested dictionary.
    When you run a Sweep via the command line,
    the config object is a
    [YAML file](https://docs.wandb.com/sweeps/quickstart#2-sweep-config).

    Let's walk through the definition of a Sweep config together.
    We'll do it slowly, so we get a chance to explain each component.
    In a typical Sweep pipeline,
    this step would be done in a single assignment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 👈 Pick a `method`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first thing we need to define is the `method`
    for choosing new parameter values.

    We provide the following search `methods`:
    *   **`grid` Search** – Iterate over every combination of hyperparameter values.
    Very effective, but can be computationally costly.
    *   **`random` Search** – Select each new combination at random according to provided `distribution`s. Surprisingly effective!
    *   **`bayes`ian Search** – Create a probabilistic model of metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. Works well for small numbers of continuous parameters but scales poorly.

    We'll stick with `random`.
    """)
    return


@app.cell
def _():
    sweep_config = {
        'method': 'bayes'
        }
    return (sweep_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For `bayes`ian Sweeps,
    you also need to tell us a bit about your `metric`.
    We need to know its `name`, so we can find it in the model outputs
    and we need to know whether your `goal` is to `minimize` it
    (e.g. if it's the squared error)
    or to `maximize` it
    (e.g. if it's the accuracy).
    """)
    return


@app.cell
def _(sweep_config):
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you're not running a `bayes`ian Sweep, you don't have to,
    but it's not a bad idea to include this in your `sweep_config` anyway,
    in case you change your mind later.
    It's also good reproducibility practice to keep note of things like this,
    in case you, or someone else,
    come back to your Sweep in 6 months or 6 years
    and don't know whether `val_G_batch` is supposed to be high or low.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📃 Name the hyper`parameters`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once you've picked a `method` to try out new values of the hyperparameters,
    you need to define what those `parameters` are.

    Most of the time, this step is straightforward:
    you just give the `parameter` a name
    and specify a list of legal `values`
    of the parameter.

    For example, when we choose the `optimizer` for our network,
    there's only a finite number of options.
    Here we stick with the two most popular choices, `adam` and `sgd`.
    Even for hyperparameters that have potentially infinite options,
    it usually only makes sense to try out
    a few select `values`,
    as we do here with `dropout`.
    """)
    return


@app.cell
def _(sweep_config):
    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'dropout': {
              'values': [0.1, 0.3, 0.5]
            },
        }

    sweep_config['parameters'] = parameters_dict
    return (parameters_dict,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's often the case that there are hyperparameters
    that we don't want to vary in this Sweep,
    but which we still want to set in our `sweep_config`.

    In that case, we just set the `value` directly:
    """)
    return


@app.cell
def _(parameters_dict):
    parameters_dict.update({
        'epochs': {
            'value': 1}
        })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a `grid` search, that's all you ever need.

    For a `random` search,
    all the `values` of a parameter are equally likely to be chosen on a given run.

    If that just won't do,
    you can instead specify a named `distribution`,
    plus its parameters, like the mean `mu`
    and standard deviation `sigma` of a `normal` distribution.

    See more on how to set the distributions of your random variables [here](https://docs.wandb.com/sweeps/configuration#distributions).
    """)
    return


@app.cell
def _(parameters_dict):
    import math

    parameters_dict.update({
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'uniform',
            'min': 0.001,
            'max': 0.1
          },
        'batch_size': {
            'values': [64, 128]
          }
        })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When we're finished, `sweep_config` is a nested dictionary
    that specifies exactly which `parameters` we're interested in trying
    and what `method` we're going to use to try them.
    """)
    return


@app.cell
def _(sweep_config):
    import pprint

    pprint.pprint(sweep_config)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But that's not all of the configuration options!

    For example, we also offer the option to `early_terminate` your runs with the [HyperBand](https://arxiv.org/pdf/1603.06560.pdf) scheduling algorithm. See more [here](https://docs.wandb.com/sweeps/configuration#stopping-criteria).

    You can find a list of all configuration options [here](https://docs.wandb.com/library/sweeps/configuration)
    and a big collection of examples in YAML format [here](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 5️⃣: Wrap the Training Loop

    You'll need a function, like `sweep_train` below,
    that uses `wandb.config` to set the hyperparameters
    before `train` gets called.
    """)
    return


@app.cell
def _(ConvNet, train, wandb):
    def sweep_train(config_defaults=None):
        # Initialize wandb with a sample project name
        with wandb.init(config=config_defaults):  # this gets over-written in the Sweep

            # Specify the other hyperparameters to the configuration, if any
            wandb.config.architecture_name = "ConvNet"
            wandb.config.dataset_name = "MNIST"

            # initialize model
            model = ConvNet(wandb.config.dropout)

            train(model, 
                  wandb.config.batch_size, 
                  wandb.config.epochs,
                  wandb.config.learning_rate,
                  wandb.config.optimizer)

    return (sweep_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 6️⃣: Initialize Sweep and Run Agent
    """)
    return


@app.cell
def _(sweep_config, wandb):
    sweep_id = wandb.sweep(sweep_config, project="sweeps-keras")
    return (sweep_id,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can limit the number of total runs with the `count` parameter, we will limit a 10 to make the script run fast, feel free to increase the number of runs and see what happens.
    """)
    return


@app.cell
def _(sweep_id, sweep_train, wandb):
    wandb.agent(sweep_id, function=sweep_train, count=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 👀 Visualize Results

    Click on the **Sweep URL** link above to see your live results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🤓 Advanced Setup
    1. [Environment variables](https://docs.wandb.com/library/environment-variables): Set API keys in environment variables so you can run training on a managed cluster.
    2. [Offline mode](https://docs.wandb.com/library/technical-faq#can-i-run-wandb-offline): Use `dryrun` mode to train offline and sync results later.
    3. [On-prem](https://docs.wandb.com/self-hosted): Install W&B in a private cloud or air-gapped servers in your own infrastructure. We have local installations for everyone from academics to enterprise teams.
    """)
    return


if __name__ == "__main__":
    app.run()
