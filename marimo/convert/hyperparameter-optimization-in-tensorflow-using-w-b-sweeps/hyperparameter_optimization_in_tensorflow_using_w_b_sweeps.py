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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/Hyperparameter_Optimization_in_TensorFlow_using_W&B_Sweeps.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{tf-sweeps} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{tf-sweeps} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧹 Weights & Biases Sweep + ‍🌊 TensorFlow 2.x
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Use Weights & Biases Sweeps to automate hyperparameter optimization and explore the space of possible models, complete with interactive dashboards like this:

    ![](https://i.imgur.com/AN0qnpC.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Why Should I Use Sweeps?

    * **Quick setup**: With just a few lines of code you can run W&B sweeps.
    * **Transparent**: We cite all the algorithms we're using, and [our code is open source](https://github.com/wandb/client/tree/master/wandb/sweeps).
    * **Powerful**: Our sweeps are completely customizable and configurable. You can launch a sweep across dozens of machines, and it's just as easy as starting a sweep on your laptop.

    **[Check out the official documentation $\rightarrow$](https://docs.wandb.com/sweeps)**

    ## What this notebook covers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * Simple steps to get started with W&B Sweep with custom training loop in TensorFlow.
    * We will find best hyperparameters for our image classification task.

    **Note**: Sections starting with _Step_ are all you need to perform hyperparameter sweep in existing code.
    The rest of the code is there to set up a simple example.
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
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1️⃣: Import W&B and Login
    """)
    return


@app.cell
def _():
    import tqdm
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.datasets import cifar10

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return keras, np, tf, tqdm


@app.cell
def _():
    import wandb
    from wandb.integration.keras import WandbCallback

    return (wandb,)


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
    # 👩‍🍳 Prepare Dataset
    """)
    return


@app.cell
def _(keras, np):
    # Prepare the training dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train/255.
    x_test = x_test/255.
    x_train = np.reshape(x_train, (-1, 784))
    x_test = np.reshape(x_test, (-1, 784))
    return x_test, x_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧠 Define the Model and Training Loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏗️ Build a Simple Classifier MLP
    """)
    return


@app.cell
def _(keras, tf):
    def Model():
        inputs = keras.Input(shape=(784,), name="digits")
        x1 = keras.layers.Dense(64, activation="relu")(inputs)
        x2 = keras.layers.Dense(64, activation="relu")(x1)
        outputs = keras.layers.Dense(10, name="predictions")(x2)

        return keras.Model(inputs=inputs, outputs=outputs)

    
    def train_step(x, y, model, optimizer, loss_fn, train_acc_metric):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_acc_metric.update_state(y, logits)

        return loss_value

    
    def test_step(x, y, model, loss_fn, val_acc_metric):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        val_acc_metric.update_state(y, val_logits)

        return loss_value

    return Model, test_step, train_step


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔁 Write a Training Loop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3️⃣: Log metrics with `wandb.log`
    """)
    return


@app.cell
def _(np, test_step, tqdm, train_step, wandb):
    def train(train_dataset,
              val_dataset, 
              model,
              optimizer,
              loss_fn,
              train_acc_metric,
              val_acc_metric,
              epochs=10, 
              log_step=200, 
              val_log_step=50):
  
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))

            train_loss = []   
            val_loss = []

            # Iterate over the batches of the dataset
            for step, (x_batch_train, y_batch_train) in tqdm.tqdm(enumerate(train_dataset), total=len(train_dataset)):
                loss_value = train_step(x_batch_train, y_batch_train, 
                                        model, optimizer, 
                                        loss_fn, train_acc_metric)
                train_loss.append(float(loss_value))

            # Run a validation loop at the end of each epoch
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                val_loss_value = test_step(x_batch_val, y_batch_val, 
                                           model, loss_fn, 
                                           val_acc_metric)
                val_loss.append(float(val_loss_value))
            
            # Display metrics at the end of each epoch
            train_acc = train_acc_metric.result()
            print("Training acc over epoch: %.4f" % (float(train_acc),))

            val_acc = val_acc_metric.result()
            print("Validation acc: %.4f" % (float(val_acc),))

            # Reset metrics at the end of each epoch
            train_acc_metric.reset_states()
            val_acc_metric.reset_states()

            # 3️⃣ log metrics using wandb.log
            wandb.log({'epochs': epoch,
                       'loss': np.mean(train_loss),
                       'acc': float(train_acc), 
                       'val_loss': np.mean(val_loss),
                       'val_acc':float(val_acc)})

    return (train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 4️⃣: Configure the Sweep

    This is where you will:
    * Define the hyperparameters you're sweeping over
    * Provide your hyperparameter optimization method. We have `random`, `grid` and `bayes` methods.
    * Provide an objective and a `metric` if using `bayes`, for example to `minimize` the `val_loss`.
    * Use `hyperband` for early termination of poorly-performing runs

    #### [Check out more on Sweep Configs $\rightarrow$](https://docs.wandb.com/sweeps/configuration)
    """)
    return


@app.cell
def _():
    sweep_config = {
      'method': 'random', 
      'metric': {
          'name': 'val_loss',
          'goal': 'minimize'
      },
      'early_terminate':{
          'type': 'hyperband',
          'min_iter': 5
      },
      'parameters': {
          'batch_size': {
              'values': [32, 64, 128, 256]
          },
          'learning_rate':{
              'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]
          }
      }
    }
    return (sweep_config,)


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
def _(Model, keras, tf, train, wandb, x_test, x_train, y_test, y_train):
    def sweep_train(config_defaults=None):
        # Set default values
        config_defaults = {
            "batch_size": 64,
            "learning_rate": 0.01
        }
        # Initialize wandb with a sample project name
        wandb.init(config=config_defaults)  # this gets over-written in the Sweep

        # Specify the other hyperparameters to the configuration, if any
        wandb.config.epochs = 2
        wandb.config.log_step = 20
        wandb.config.val_log_step = 50
        wandb.config.architecture_name = "MLP"
        wandb.config.dataset_name = "MNIST"

        # build input pipeline using tf.data
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (train_dataset.shuffle(buffer_size=1024)
                                      .batch(wandb.config.batch_size)
                                      .prefetch(buffer_size=tf.data.AUTOTUNE))

        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        val_dataset = (val_dataset.batch(wandb.config.batch_size)
                                  .prefetch(buffer_size=tf.data.AUTOTUNE))

        # initialize model
        model = Model()

        # Instantiate an optimizer to train the model.
        optimizer = keras.optimizers.SGD(learning_rate=wandb.config.learning_rate)
        # Instantiate a loss function.
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # Prepare the metrics.
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        train(train_dataset,
              val_dataset, 
              model,
              optimizer,
              loss_fn,
              train_acc_metric,
              val_acc_metric,
              epochs=wandb.config.epochs, 
              log_step=wandb.config.log_step, 
              val_log_step=wandb.config.val_log_step)

    return (sweep_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 6️⃣: Initialize Sweep and Run Agent
    """)
    return


@app.cell
def _(sweep_config, wandb):
    sweep_id = wandb.sweep(sweep_config, project="sweeps-tensorflow")
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
    # 🎨 Example Gallery

    See examples of projects tracked and visualized with W&B in our [Gallery →](https://app.wandb.ai/gallery)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 📏 Best Practices
    1. **Projects**: Log multiple runs to a project to compare them. `wandb.init(project="project-name")`
    2. **Groups**: For multiple processes or cross validation folds, log each process as a runs and group them together. `wandb.init(group='experiment-1')`
    3. **Tags**: Add tags to track your current baseline or production model.
    4. **Notes**: Type notes in the table to track the changes between runs.
    5. **Reports**: Take quick notes on progress to share with colleagues and make dashboards and snapshots of your ML projects.
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
