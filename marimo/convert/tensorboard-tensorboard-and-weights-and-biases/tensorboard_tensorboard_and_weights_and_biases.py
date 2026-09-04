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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorboard/TensorBoard_and_Weights_and_Biases.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{tensorboard} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases"/> <br>

    <!--- @wandbcode{tensorboard, v=examples} -->

    <img src="http://wandb.me/mini-diagram" width="600" alt="Weights & Biases"/>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By the end of this colab you will have a TensorBoard server running in Weights & Biases, just like this:

    <img src="https://i.imgur.com/fjOsTLO.png" width="600" alt="W&B in TensorBoard"/>

    This code is modified from the offical TensorBoard [getting started](https://www.tensorflow.org/tensorboard/get_started) code
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🪴 Start a Weights & Biases run
    When using Weights & Biases for the first time you will need to:

    1️⃣. Sign-up for a free W&B [account here](https://wandb.ai/site)

    2️⃣. Create a new W&B [API key at your settings page](https://wandb.ai/settings) and store it securely. API keys can only be viewed once when created.

    3️⃣. Initialise a W&B run with wandb.init and you will be prompted to enter your API key to log in
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qqq wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Initialising a Weights & Biases run with `sync_tensorboard=True` will enable wandb to pick up
    """)
    return


@app.cell
def _():
    import wandb
    run = wandb.init(project="my-wonderful-project", sync_tensorboard=True)
    return (run,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🌿 Create Your Dataset and Model
    """)
    return


@app.cell
def _():
    import tensorflow as tf

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    def create_model():
      return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
      ])

    model = create_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, tf, x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🌲 Train Your Model and Log to TensorBoard AND Weights & Biases

    The tensorboard logs will be automatically picked up by Weights & Biases and logged
    """)
    return


@app.cell
def _(model, tf, x_test, x_train, y_test, y_train):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(histogram_freq=1)

    model.fit(x=x_train, 
              y=y_train, 
              epochs=5, 
              validation_data=(x_test, y_test), 
              callbacks=[tensorboard_callback])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (Notebook only) Finish the Weights & Biases Run
    """)
    return


@app.cell
def _(run):
    run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Documentation

    You can find additional documentation of how to use [Weights & Biases with Tensorboard here](https://docs.wandb.ai/guides/integrations/tensorboard)
    """)
    return


if __name__ == "__main__":
    app.run()
