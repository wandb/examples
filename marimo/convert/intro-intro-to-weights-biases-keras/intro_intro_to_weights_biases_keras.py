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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_&_Biases_keras.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{intro-colab-keras} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{intro-colab-keras} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏃‍♀️ Quickstart
    Use **[Weights & Biases](https://wandb.ai/site?utm_source=keras_intro_colab&utm_medium=code&utm_campaign=keras_intro)** for machine learning experiment tracking, model checkpointing, and collaboration with your team. See the full Weights & Biases Documentation **[here](https://docs.wandb.ai/guides/integrations/keras)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤩 A shared dashboard for your experiments

    With just a few lines of code,
    you'll get rich, interactive, shareable dashboards [which you can see yourself here](https://wandb.ai/wandb/wandb_example).
    ![](https://i.imgur.com/Pell4Oo.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔒 Data & Privacy

    We take security very seriously, and our cloud-hosted dashboard uses industry standard best practices for encryption. If you're working with datasets that cannot leave your enterprise cluster, we have [on-prem](https://docs.wandb.com/self-hosted) installations available.

    It's also easy to download all your data and export it to other tools — like custom analysis in a Jupyter notebook. Here's [more on our API](https://docs.wandb.com/library/api).

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start by installing the library and logging in to your free account.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell
def _():
    # Log in to your W&B account
    import wandb

    # Use wandb-core
    wandb.require("core")
    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 👟 Run an experiment
    1️⃣. **Start a new run** and pass in hyperparameters to track

    2️⃣. **Log metrics** from training or evaluation

    3️⃣. **Visualize results** in the dashboard
    """)
    return


@app.cell
def _(wandb):
    import random
    for _run in range(5):
    # Launch 5 simulated experiments
        wandb.init(project='basic-intro', config={'learning_rate': 0.02, 'architecture': 'CNN', 'dataset': 'CIFAR-100'})
        offset = random.random() / 5  # 1️⃣ Start a new run to track this script
        for ii in range(2, 10):
            acc = 1 - 2 ** (-ii) - random.random() / ii - offset  # Set entity to specify your username or team name
            loss = 2 ** (-ii) + random.random() / ii + offset  # ex: entity="carey",
            wandb.log({'acc': acc, 'loss': loss})  # Set the project where this run will be logged
        wandb.finish()  # Track hyperparameters and run metadata  # This simple block simulates a training loop logging metrics  # 2️⃣ Log metrics from your script to W&B  # Mark the run as finished
    return (random,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You have now trained your first model using wandb! 👆 Click on the wandb link above to see your metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🥕 Simple Keras Classifier
    Run this model to train a simple MNIST classifier, and click on the project page link to see your results stream in live to a W&B project. For a full guide on how to use Weights & Biases with Keras, **[see here](https://docs.wandb.ai/guides/integrations/keras)**
    """)
    return


@app.cell
def _(random, wandb):
    import numpy as np
    import tensorflow as tf
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
    for _run in range(5):
        wandb.init(project='keras-intro', config={'layer_1': 512, 'activation_1': 'relu', 'dropout': random.uniform(0.01, 0.8), 'layer_2': 10, 'activation_2': 'softmax', 'optimizer': 'sgd', 'loss': 'sparse_categorical_crossentropy', 'metric': 'accuracy', 'epoch': 6, 'batch_size': 256})
        config = wandb.config
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = (x_train / 255.0, x_test / 255.0)
        x_train, y_train = (x_train[::5], y_train[::5])
        x_test, y_test = (x_test[::20], y_test[::20])
        labels = [str(digit) for digit in range(np.max(y_train) + 1)]
        model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), tf.keras.layers.Dense(config.layer_1, activation=config.activation_1), tf.keras.layers.Dropout(config.dropout), tf.keras.layers.Dense(config.layer_2, activation=config.activation_2)])
        model.compile(optimizer=config.optimizer, loss=config.loss, metrics=[config.metric])
        wandb_callbacks = [WandbMetricsLogger(), WandbModelCheckpoint(filepath='my_model_{epoch:02d}.keras')]
        model.fit(x=x_train, y=y_train, epochs=config.epoch, batch_size=config.batch_size, validation_data=(x_test, y_test), callbacks=wandb_callbacks)
        wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You have now trained your first model using wandb! 👆 Click on the wandb link above to see your metrics.

    For a full guide on how to use Weights & Biases with Keras, **[see here](https://docs.wandb.ai/guides/integrations/keras)**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔔 Try W&B Alerts

    **[W&B Alerts](https://docs.wandb.ai/guides/track/alert)** allows you to send alerts, triggered from your Python code, to your Slack or email. There are 2 steps to follow the first time you'd like to send a Slack or email alert, triggered from your code:

    1) Turn on Alerts in your W&B [User Settings](https://wandb.ai/settings)

    2) Add `wandb.alert()` to your code:

    ```python
    wandb.alert(
        title="Low accuracy",
        text=f"Accuracy is below the acceptable threshold"
    )
    ```

    See the minimal example below to see how to use `wandb.alert`. You can find the full docs for **[W&B Alerts here](https://docs.wandb.ai/guides/track/alert)**
    """)
    return


@app.cell
def _(random, wandb):
    # Start a wandb run
    wandb.init(project="keras-intro")

    # Simulating a model training loop
    acc_threshold = 0.3
    for training_step in range(1000):

        # Generate a random number for accuracy
        accuracy = round(random.random() + random.random(), 3)
        print(f"Accuracy is: {accuracy}, {acc_threshold}")

        # 🐝 Log accuracy to wandb
        wandb.log({"Accuracy": accuracy})

        # 🔔 If the accuracy is below the threshold, fire a W&B Alert and stop the run
        if accuracy <= acc_threshold:
            # 🐝 Send the wandb Alert
            wandb.alert(
                title="Low Accuracy",
                text=f"Accuracy {accuracy} at step {training_step} is below the acceptable theshold, {acc_threshold}",
            )
            print("Alert triggered")
            break

    # Mark the run as finished (useful in Jupyter notebooks)
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # What's next 🚀 ?
    The next tutorial you will learn how to do hyperparameter optimization using W&B Sweeps:
    ## 👉 [Hyperparameters sweeps using PyTorch](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb)
    """)
    return


if __name__ == "__main__":
    app.run()
