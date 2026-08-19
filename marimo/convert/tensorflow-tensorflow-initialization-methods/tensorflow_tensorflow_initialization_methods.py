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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Tensorflow_Initialization_Methods.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Packages 📦 and Basic Setup
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install -Uq wandb
    return


@app.cell
def _():
    import wandb
    import numpy as np
    import tensorflow as tf
    from wandb.keras import WandbCallback

    return WandbCallback, tf, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 💿 Dataset

    For the sake of simplicity, MNIST was chosen
    """)
    return


@app.cell
def _(tf):
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The Model 👷‍♀️

    Initializers used (from `tf.keras.initializers`):-

    * Zeros
    * LeCunNormal
    * GlorotNormal
    * HeNormal
    """)
    return


@app.cell
def _(tf):
    initializer = tf.keras.initializers.LecunNormal()

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='tanh', kernel_initializer=initializer),
      tf.keras.layers.Dense(64, activation='tanh', kernel_initializer=initializer),
      tf.keras.layers.Dense(32, activation='tanh', kernel_initializer=initializer),
      tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training 💪🏻

    The weights and gradients were logged using the helpful `log_gradients` and `log_weights` parameters of `WandbCallback()`
    """)
    return


@app.cell
def _():
    PROJECT = "tensorflow_initialization_methods"
    return (PROJECT,)


@app.cell
def _(PROJECT, WandbCallback, model, wandb, x_train, y_train):
    run = wandb.init(project=PROJECT)

    model.fit(x_train, y_train, epochs=20, callbacks=[WandbCallback(training_data = (x_train, y_train),log_weights = True, log_gradients = True, save_model = False)])

    run.finish()
    return


if __name__ == "__main__":
    app.run()
