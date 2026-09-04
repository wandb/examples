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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/cosine_decay_using_keras.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{keras-cosine-decay} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using Cosine Decay with Keras
    <!--- @wandbcode{keras-cosine-decay} -->
    This notebook demonstrates how to use the Cosine Decay learning rate schedule with Keras.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qq wandb
    return


@app.cell
def _():
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import models
    import tensorflow_datasets as tfds

    # Weights and Biases related imports
    import wandb
    from wandb.keras import WandbMetricsLogger

    return WandbMetricsLogger, layers, models, tf, tfds, wandb


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    configs = dict(
        num_classes = 10,
        shuffle_buffer = 1024,
        batch_size = 64,
        image_size = 28,
        image_channels = 1,
        earlystopping_patience = 3,
        learning_rate = 1e-3,
        epochs = 10,
        num_steps = 0.7,
    )
    return (configs,)


@app.cell
def _(configs, tf, tfds):
    AUTOTUNE = tf.data.AUTOTUNE


    def parse_data(example):
        # Get image
        image = example["image"]

        # Get label
        label = example["label"]
        label = tf.one_hot(label, depth=configs["num_classes"])

        return image, label


    def get_dataloader(ds, configs, dataloader_type="train"):
        dataloader = ds.map(parse_data, num_parallel_calls=AUTOTUNE)

        if dataloader_type=="train":
            dataloader = dataloader.shuffle(configs["shuffle_buffer"])
      
        dataloader = (
            dataloader
            .batch(configs["batch_size"])
            .prefetch(AUTOTUNE)
        )

        return dataloader

    train_ds, valid_ds = tfds.load('fashion_mnist', split=['train', 'test'])

    trainloader = get_dataloader(train_ds, configs)
    validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
    return trainloader, validloader


@app.cell
def _(configs, layers, models, tf):
    def get_model(configs):
        backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', include_top=False)
        backbone.trainable = True

        inputs = layers.Input(shape=(configs["image_size"], configs["image_size"], configs["image_channels"]))
        resize = layers.Resizing(32, 32)(inputs)
        neck = layers.Conv2D(3, (3,3), padding="same")(resize)
        preprocess_input = tf.keras.applications.mobilenet.preprocess_input(neck)
        x = backbone(preprocess_input)
        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(configs["num_classes"], activation="softmax")(x)

        return models.Model(inputs=inputs, outputs=outputs)


    tf.keras.backend.clear_session()
    model = get_model(configs)
    model.summary()
    return (model,)


@app.cell
def _(configs, tf, trainloader):
    # Learning Rate
    total_steps = len(trainloader)*configs["epochs"]
    decay_steps = total_steps * configs["num_steps"]

    cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate = configs["learning_rate"],
        decay_steps = decay_steps,
        alpha=0.1
    )
    return (cosine_decay_scheduler,)


@app.cell
def _(cosine_decay_scheduler, model, tf):
    model.compile(
        optimizer = tf.keras.optimizers.Adam(cosine_decay_scheduler),
        loss = "categorical_crossentropy",
        metrics = ["accuracy"]
    )
    return


@app.cell
def _(WandbMetricsLogger, configs, model, trainloader, validloader, wandb):
    # Initialize a W&B run
    run = wandb.init(
        project = "cosine_decay",
        config = configs,
    )

    # Train your model
    model.fit(
        trainloader,
        epochs = configs["epochs"],
        validation_data = validloader,
        callbacks = [
            WandbMetricsLogger(log_freq=2),
        ]
    )
    return (run,)


@app.cell
def _(model, validloader, wandb):
    eval_loss, eval_acc = model.evaluate(validloader)

    wandb.log({
        "eval_loss": eval_loss,
        "eval_acc": eval_acc
    })
    return


@app.cell
def _(run):
    # Close the W&B run
    run.finish()
    return


if __name__ == "__main__":
    app.run()
