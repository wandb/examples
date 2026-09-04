# /// script
# dependencies = ["flax", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/jax/Simple_Training_Loop_in_JAX_and_Flax.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{stylegan-nada-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Writing a Simple Training Loop in JAX and FLAX
    <!--- @wandbcode{stylegan-nada-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Packages 📦 and Basic Setup
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ❤️ Install Packages
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb flax !pip install -q wandb flax
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp

    import optax

    from flax import linen as nn
    from flax.training import train_state
    from flax.serialization import (
        to_state_dict, msgpack_serialize, from_bytes
    )

    import os
    import wandb
    import numpy as np
    from typing import Callable
    from tqdm.auto import tqdm

    import tensorflow as tf
    import tensorflow_datasets as tfds

    return (
        Callable,
        from_bytes,
        jax,
        jnp,
        msgpack_serialize,
        nn,
        np,
        optax,
        os,
        tf,
        tfds,
        to_state_dict,
        tqdm,
        train_state,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ⚙️ Project Configuration using **`wandb.config`**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can now call [**`wandb.init`**](https://docs.wandb.ai/guides/track/launch) to initialize a new job. This creates a new run in [**Weights & Biases**](https://wandb.ai/site) and launches a background process to sync data. We will also sync all the configs of our experiments with the W&B run, which makes it far easier for us to reproduce the results of the experiment later.
    """)
    return


@app.cell
def _(nn, wandb):
    wandb.init(
        project="simple-training-loop",
        entity="jax-series",
        job_type="simple-train-loop"
    )

    config = wandb.config
    config.seed = 42
    config.batch_size = 64
    config.validation_split = 0.2
    config.pooling = "avg"
    config.learning_rate = 1e-4
    config.epochs = 15

    MODULE_DICT = {
        "avg": nn.avg_pool,
        "max": nn.max_pool,
    }
    return MODULE_DICT, config


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💿 The Dataset
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    most JAX practitioners prefer to use the **`tf.data`** API for building data loading pipelines for JAX and Flax-based machine learning workflow. In this notebook, we will build a simple data loading pipeline for the CIFAR-10 dataset using Tensorflow Datasets for Image Classification.
    """)
    return


@app.cell
def _(config, tf, tfds):
    (full_train_set, test_dataset), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        image = tf.cast(image, tf.float32) / 255.
        return image, label

    full_train_set = full_train_set.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )

    num_data = tf.data.experimental.cardinality(
        full_train_set
    ).numpy()
    print("Total number of data points:", num_data)
    train_dataset = full_train_set.take(
        num_data * (1 - config.validation_split)
    )
    val_dataset = full_train_set.take(
        num_data * (config.validation_split)
    )
    print(
        "Number of train data points:",
        tf.data.experimental.cardinality(train_dataset).numpy()
    )
    print(
        "Number of val data points:",
        tf.data.experimental.cardinality(val_dataset).numpy()
    )

    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(
        tf.data.experimental.cardinality(train_dataset).numpy()
    )
    train_dataset = train_dataset.batch(config.batch_size)

    val_dataset = val_dataset.cache()
    val_dataset = val_dataset.shuffle(
        tf.data.experimental.cardinality(val_dataset).numpy()
    )
    val_dataset = val_dataset.batch(config.batch_size)


    test_dataset = test_dataset.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE
    )
    print(
        "Number of test data points:",
        tf.data.experimental.cardinality(test_dataset).numpy()
        )
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.batch(config.batch_size)
    return test_dataset, train_dataset, val_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✍️ Model Architecture
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us now define a very simple classification convolution based neural network. Instead of some famous architecture we'll create a simple custom architecture by subclassing [**`linen.Module`**](https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module).
    """)
    return


@app.cell
def _(Callable, nn):
    class CNN(nn.Module):
        pool_module: Callable = nn.avg_pool

        def setup(self):
            self.conv_1 = nn.Conv(features=32, kernel_size=(3, 3))
            self.conv_2 = nn.Conv(features=32, kernel_size=(3, 3))
            self.conv_3 = nn.Conv(features=64, kernel_size=(3, 3))
            self.conv_4 = nn.Conv(features=64, kernel_size=(3, 3))
            self.conv_5 = nn.Conv(features=128, kernel_size=(3, 3))
            self.conv_6 = nn.Conv(features=128, kernel_size=(3, 3))
            self.dense_1 = nn.Dense(features=1024)
            self.dense_2 = nn.Dense(features=512)
            self.dense_output = nn.Dense(features=10)

        @nn.compact
        def __call__(self, x):
            x = nn.relu(self.conv_1(x))
            x = nn.relu(self.conv_2(x))
            x = self.pool_module(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.relu(self.conv_3(x))
            x = nn.relu(self.conv_4(x))
            x = self.pool_module(x, window_shape=(2, 2), strides=(2, 2))
            x = nn.relu(self.conv_5(x))
            x = nn.relu(self.conv_6(x))
            x = self.pool_module(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))
            x = nn.relu(self.dense_1(x))
            x = nn.relu(self.dense_2(x))
            return self.dense_output(x)

    return (CNN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have defined, the CNN Module, we would need to initialize it. However, unlike Tensorflow or PyTorch, the parameters of a Flax Module are not stored with the models themselves. We would need to initialize parameters by calling the init function, using a PRNG Key and a dummy input parameter with the same shape as the expected input.
    """)
    return


@app.cell
def _(CNN, MODULE_DICT, config, jax, jnp):
    rng = jax.random.PRNGKey(config.seed)
    x = jnp.ones(shape=(config.batch_size, 32, 32, 3))
    model = CNN(pool_module=MODULE_DICT[config.pooling])
    params = model.init(rng, x)
    jax.tree_map(lambda x: x.shape, params)
    return model, rng, x


@app.cell
def _(model, nn, rng, x):
    nn.tabulate(model, rng)(x)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After we initialize the model we'll use the variables to create a [**TrainState**](https://flax.readthedocs.io/en/latest/flax.training.html#flax.training.train_state.TrainState), a utility class for handling parameter and gradient updates. This is a key feature of the new Flax version. Instead of initializing the model again and again with new variables we just update the "state" of the model and pass this as inputs to functions.
    """)
    return


@app.cell
def _(config, jnp, model, optax, rng, train_state):
    def init_train_state(
        model, random_key, shape, learning_rate
    ) -> train_state.TrainState:
        # Initialize the Model
        variables = model.init(random_key, jnp.ones(shape))
        # Create the optimizer
        optimizer = optax.adam(learning_rate)
        # Create a State
        return train_state.TrainState.create(
            apply_fn = model.apply,
            tx=optimizer,
            params=variables['params']
        )


    state = init_train_state(
        model, rng, (config.batch_size, 32, 32, 3), config.learning_rate
    )
    print(type(state))
    return (state,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ⚙️ Utility Functions
    """)
    return


@app.cell
def _(jax, optax):
    def cross_entropy_loss(*, logits, labels):
        one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=10)
        return optax.softmax_cross_entropy(
            logits=logits, labels=one_hot_encoded_labels
        ).mean()

    return (cross_entropy_loss,)


@app.cell
def _(cross_entropy_loss, jnp):
    def compute_metrics(*, logits, labels):
      loss = cross_entropy_loss(logits=logits, labels=labels)
      accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
      metrics = {
          'loss': loss,
          'accuracy': accuracy,
      }
      return metrics

    return (compute_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧱 + 🏗 = 🏠 Training
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    * Any train step should take in two basic parameters; the state and the batch (or whatever format the input is) in question.

    * We usually define the loss function within this function as best practice. We get the logits from the model, using the `apply_fn` from the TrainState (which is just the apply method of the model) by passing the parameters and the input. We then compute the loss by using the logits and input and return the loss as well as the logits (this is key).

    * We then transform the function using `jax.value_and_grad()` transformation. Instead of `jax.grad()` which just creates a function which returns the derivative of the function. We use `jax.value_and_grad()` which returns the gradient as well as the evaluation of the function. (Notice the `has_aux` parameter, we set this to True because the loss function returns the loss as well as the logits, an auxiliary object)

    * We then calculate the gradients and obtain the logits by passing in the parameters of the state. Notice how the function returns both the gradients and the logits (because we used `jax.value_and_grad()` instead of `jax.grad()`) we'll later need these logits to calculate metrics after the step

    * We then essentially perform backpropagation by updating the TrainState using the calculated gradients by using the `.apply_gradients()` method

    * Calculate the metrics using the utility `compute_metrics` function.
    """)
    return


@app.cell
def _(compute_metrics, cross_entropy_loss, jax, jnp, train_state):
    @jax.jit
    def train_step(
        state: train_state.TrainState, batch: jnp.ndarray
    ):
        image, label = batch

        def loss_fn(params):
            logits = state.apply_fn({'params': params}, image)
            loss = cross_entropy_loss(logits=logits, labels=label)
            return loss, logits

        gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, logits), grads = gradient_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = compute_metrics(logits=logits, labels=label)
        return state, metrics

    return (train_step,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similar to our `train_step` this function also takes the state and the batch. We simply perform a forward pass using the data and obtain the logits and then compute the corresponding metrics. As this is the `eval_step` we don't compute the gradients or update the parameters of the TrainState.
    """)
    return


@app.cell
def _(compute_metrics, jax):
    @jax.jit
    def eval_step(state, batch):
        image, label = batch
        logits = state.apply_fn({'params': state.params}, image)
        return compute_metrics(logits=logits, labels=label)

    return (eval_step,)


@app.cell
def _(from_bytes, jax, msgpack_serialize, np, os, to_state_dict, wandb):
    def save_checkpoint(ckpt_path, state, epoch):
        with open(ckpt_path, "wb") as outfile:
            outfile.write(msgpack_serialize(to_state_dict(state)))
        artifact = wandb.Artifact(
            f'{wandb.run.name}-checkpoint', type='dataset'
        )
        artifact.add_file(ckpt_path)
        wandb.log_artifact(artifact, aliases=["latest", f"epoch_{epoch}"])


    def load_checkpoint(ckpt_file, state):
        artifact = wandb.use_artifact(
            f'{wandb.run.name}-checkpoint:latest'
        )
        artifact_dir = artifact.download()
        ckpt_path = os.path.join(artifact_dir, ckpt_file)
        with open(ckpt_path, "rb") as data_file:
            byte_data = data_file.read()
        return from_bytes(state, byte_data)


    def accumulate_metrics(metrics):
        metrics = jax.device_get(metrics)
        return {
            k: np.mean([metric[k] for metric in metrics])
            for k in metrics[0]
        }

    return accumulate_metrics, load_checkpoint, save_checkpoint


@app.cell
def _(
    accumulate_metrics,
    eval_step,
    load_checkpoint,
    save_checkpoint,
    tf,
    tfds,
    tqdm,
    train_state,
    train_step,
    wandb,
):
    def train_and_evaluate(
        train_dataset,
        eval_dataset,
        test_dataset,
        state: train_state.TrainState,
        epochs: int,
    ):
        num_train_batches = tf.data.experimental.cardinality(train_dataset)
        num_eval_batches = tf.data.experimental.cardinality(eval_dataset)
        num_test_batches = tf.data.experimental.cardinality(test_dataset)
    
        for epoch in tqdm(range(1, epochs + 1)):

            best_eval_loss = 1e6
        
            train_batch_metrics = []
            train_datagen = iter(tfds.as_numpy(train_dataset))
            for batch_idx in range(num_train_batches):
                batch = next(train_datagen)
                state, metrics = train_step(state, batch)
                train_batch_metrics.append(metrics)
        
            train_batch_metrics = accumulate_metrics(train_batch_metrics)
            print(
                'TRAIN (%d/%d): Loss: %.4f, accuracy: %.2f' % (
                    epoch, epochs, train_batch_metrics['loss'],
                    train_batch_metrics['accuracy'] * 100
                )
            )

            eval_batch_metrics = []
            eval_datagen = iter(tfds.as_numpy(eval_dataset))
            for batch_idx in range(num_eval_batches):
                batch = next(eval_datagen)
                metrics = eval_step(state, batch)
                eval_batch_metrics.append(metrics)
        
            eval_batch_metrics = accumulate_metrics(eval_batch_metrics)
            print(
                'EVAL (%d/%d):  Loss: %.4f, accuracy: %.2f\n' % (
                    epoch, epochs, eval_batch_metrics['loss'],
                    eval_batch_metrics['accuracy'] * 100
                )
            )

            wandb.log({
                "Train Loss": train_batch_metrics['loss'],
                "Train Accuracy": train_batch_metrics['accuracy'],
                "Validation Loss": eval_batch_metrics['loss'],
                "Validation Accuracy": eval_batch_metrics['accuracy']
            }, step=epoch)

            if eval_batch_metrics['loss'] < best_eval_loss:
                save_checkpoint("checkpoint.msgpack", state, epoch)
    
        restored_state = load_checkpoint("checkpoint.msgpack", state)
        test_batch_metrics = []
        test_datagen = iter(tfds.as_numpy(test_dataset))
        for batch_idx in range(num_test_batches):
            batch = next(test_datagen)
            metrics = eval_step(restored_state, batch)
            test_batch_metrics.append(metrics)
    
        test_batch_metrics = accumulate_metrics(test_batch_metrics)
        print(
            'Test: Loss: %.4f, accuracy: %.2f' % (
                test_batch_metrics['loss'],
                test_batch_metrics['accuracy'] * 100
            )
        )

        wandb.log({
            "Test Loss": test_batch_metrics['loss'],
            "Test Accuracy": test_batch_metrics['accuracy']
        })
    
        return state, restored_state

    return (train_and_evaluate,)


@app.cell
def _(
    config,
    state,
    test_dataset,
    train_and_evaluate,
    train_dataset,
    val_dataset,
):
    state_1, best_state = train_and_evaluate(train_dataset, val_dataset, test_dataset, state, epochs=config.epochs)
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
