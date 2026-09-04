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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/jax/training_with_tfrecords_in_jax_imagenette.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb flax !pip install -q wandb flax
    return


@app.cell
def _():
    import os
    import wandb
    import numpy as np
    from glob import glob
    from typing import Callable
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    import jax
    import jax.numpy as jnp

    import optax

    from flax import linen as nn
    from flax.training import train_state
    from flax.serialization import (
        to_state_dict, msgpack_serialize, from_bytes
    )

    import tensorflow as tf
    import tensorflow_datasets as tfds
    AUTOTUNE = tf.data.AUTOTUNE
    return (
        AUTOTUNE,
        Callable,
        from_bytes,
        glob,
        jax,
        jnp,
        msgpack_serialize,
        nn,
        np,
        optax,
        os,
        plt,
        tf,
        tfds,
        to_state_dict,
        tqdm,
        train_state,
        wandb,
    )


@app.cell
def _(nn, wandb):
    wandb.init(
        project="simple-training-loop",
        entity="jax-series",
        job_type="tfrecord"
    )

    config = wandb.config
    config.seed = 42
    config.image_size = 227
    config.batch_size = 64
    config.pooling = "max"
    config.learning_rate = 1e-4
    config.epochs = 15
    config.artifact_address = 'jax-series/simple-training-loop/imagenette-tfrecords:v3'
    config.labels = [
        'tench', 'english_springer', 'english_springer', 'chain_saw',
        'church', 'french_horn', 'grabage_truck', 'gas_pump',
        'golf_ball', 'parachute'
    ]

    MODULE_DICT = {
        "avg": nn.avg_pool,
        "max": nn.max_pool,
    }
    return MODULE_DICT, config


@app.cell
def _(AUTOTUNE, tf):
    def decode_image(image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        return image

    def read_labeled_tfrecord(example):
        feature = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }

        example = tf.io.parse_single_example(example, feature)
        image = decode_image(example['image'])
        label = tf.cast(example['label'], tf.int32)
        return image, label

    def load_dataset(filenames, ordered = False):
    
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False 
        
        dataset = tf.data.TFRecordDataset(
            filenames, num_parallel_reads=AUTOTUNE
        )
        dataset_len = sum(1 for _ in dataset)
        dataset = dataset.with_options(ignore_order)
        dataset = dataset.map(
            read_labeled_tfrecord, num_parallel_calls=AUTOTUNE
        ) 
        return dataset, dataset_len

    return (load_dataset,)


@app.cell
def _(config, glob, os, wandb):
    artifact = wandb.use_artifact(
        config.artifact_address, type='dataset'
    )
    artifact_dir = artifact.download()
    train_files = glob(os.path.join(artifact_dir, "train", "*.tfrec"))
    val_files = glob(os.path.join(artifact_dir, "val", "*.tfrec"))
    return train_files, val_files


@app.cell
def _(load_dataset, train_files):
    sample_dataset, _ = load_dataset(train_files)
    sample_dataset = sample_dataset.shuffle(1024)
    sample_dataset.element_spec
    return (sample_dataset,)


@app.cell
def _(config, plt, sample_dataset):
    plt.figure(figsize=(16, 16))
    for i in range(16):
        x, y = next(iter(sample_dataset))
        x, y = x.numpy(), y.numpy().tolist()
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(x)
        plt.axis("off")
        name = config.labels[y]
        ax.set_title(name, fontsize=20)
    return


@app.cell
def _(AUTOTUNE, config, load_dataset, tf):
    def resize_image(image, label):
        image = tf.image.resize(
            image, [config.image_size, config.image_size]
        )
        return image, label


    def data_augment(image, label):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, 0.01)
        image = tf.image.random_saturation(image, 0.70, 1.30)
        image = tf.image.random_contrast(image, 0.80, 1.20)
        image = tf.image.random_brightness(image, 0.10)
        return image, label

    def get_training_dataset(filenames, batch_size):
        dataset, dataset_len = load_dataset(filenames, ordered = False)
        dataset = dataset.map(
            resize_image, num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            data_augment, num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.repeat()
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset, dataset_len // batch_size

    def get_val_dataset(filenames, batch_size):
        dataset, dataset_len = load_dataset(filenames, ordered = True)
        dataset = dataset.map(
            resize_image, num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            data_augment, num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset, dataset_len // batch_size

    return (get_training_dataset,)


@app.cell
def _(config, get_training_dataset, train_files, val_files):
    train_dataset, num_train_batches = get_training_dataset(train_files, config.batch_size)
    val_dataset, num_val_batches = get_training_dataset(val_files, config.batch_size)
    return num_train_batches, num_val_batches, train_dataset, val_dataset


@app.cell
def _(Callable, nn):
    class AlexNet(nn.Module):
        num_classes: int
        pool_module: Callable = nn.avg_pool

        def setup(self):
            self.conv_1 = nn.Conv(
                features=96, kernel_size=(11, 11), strides=4, padding="VALID"
            )
            self.conv_2 = nn.Conv(
                features=256, kernel_size=(5, 5), strides=1, padding="VALID"
            )
            self.conv_3 = nn.Conv(
                features=384, kernel_size=(3, 3), strides=1, padding="VALID"
            )
            self.conv_4 = nn.Conv(
                features=384, kernel_size=(3, 3), strides=1, padding="VALID"
            )
            self.conv_5 = nn.Conv(
                features=256, kernel_size=(3, 3), strides=1, padding="VALID"
            )
            self.dense_1 = nn.Dense(features=1024)
            self.dense_2 = nn.Dense(features=512)
            self.dense_output = nn.Dense(features=self.num_classes)
    
        def __call__(self, x):
            x = nn.relu(self.conv_1(x))
            x = self.pool_module(x, window_shape=(3, 3), strides=(2, 2))
            x = nn.relu(self.conv_2(x))
            x = self.pool_module(x, window_shape=(3, 3), strides=(2, 2))
            x = nn.relu(self.conv_3(x))
            x = nn.relu(self.conv_4(x))
            x = nn.relu(self.conv_5(x))
            x = self.pool_module(x, window_shape=(3, 3), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))
            x = nn.relu(self.dense_1(x))
            x = nn.relu(self.dense_2(x))
            return self.dense_output(x)

    return (AlexNet,)


@app.cell
def _(AlexNet, MODULE_DICT, config, jax, jnp):
    rng = jax.random.PRNGKey(config.seed)
    x_1 = jnp.ones(shape=(config.batch_size, config.image_size, config.image_size, 3))
    model = AlexNet(num_classes=len(config.labels), pool_module=MODULE_DICT[config.pooling])
    params = model.init(rng, x_1)
    jax.tree_map(lambda x: x.shape, params)
    return model, rng, x_1


@app.cell
def _(model, nn, rng, x_1):
    nn.tabulate(model, rng)(x_1)
    return


@app.cell
def _(config, jnp, model, optax, rng, train_state):
    def init_train_state(
        model, random_key, shape, learning_rate
    ) -> train_state.TrainState:
        variables = model.init(random_key, jnp.ones(shape))
        optimizer = optax.adam(learning_rate)
        return train_state.TrainState.create(
            apply_fn = model.apply,
            tx=optimizer,
            params=variables['params']
        )


    state = init_train_state(
        model=model,
        random_key=rng,
        shape=(config.batch_size, config.image_size, config.image_size, 3),
        learning_rate=config.learning_rate
    )
    print(type(state))
    return (state,)


@app.cell
def _(config, jax, optax):
    def cross_entropy_loss(*, logits, labels):
        one_hot_encoded_labels = jax.nn.one_hot(
            labels, num_classes=len(config.labels)
        )
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

    return accumulate_metrics, save_checkpoint


@app.cell
def _(
    accumulate_metrics,
    eval_step,
    save_checkpoint,
    tfds,
    tqdm,
    train_state,
    train_step,
    wandb,
):
    def train_and_evaluate(
        train_dataset,
        eval_dataset,
        num_train_batches,
        num_eval_batches,
        state: train_state.TrainState,
        epochs: int,
    ):
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
    
        return state

    return (train_and_evaluate,)


@app.cell
def _(
    config,
    num_train_batches,
    num_val_batches,
    state,
    train_and_evaluate,
    train_dataset,
    val_dataset,
):
    state_1 = train_and_evaluate(train_dataset, val_dataset, num_train_batches, num_val_batches, state, epochs=config.epochs)
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
