# /// script
# dependencies = ["pip", "restorers @ git+https://github.com/soumik12345/restorers.git", "setuptools"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/restorers/Train_Zero_DCE_Restorers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{restorers-zero-dce-train} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌈 Restorers + WandB 🪄🐝

    <!--- @wandbcode{restorers-mirnetv2-train} -->

    This notebook shows how to train a [Zero-DCE](https://arxiv.org/abs/2001.06826) model for zero-reference low-light enhancement using [**restorers**](https://github.com/soumik12345/restorers) and [**wandb**](https://wandb.ai/site). For more details regarding usage of restorers, refer to the following report:

    [![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/ml-colabs/low-light-enhancement/reports/Lighting-up-Images-in-the-Deep-Learning-Era--VmlldzozNzE4Njkz)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: pip setuptools !pip install -q --upgrade pip setuptools
    # packages added via marimo's package management: git+https://github.com/soumik12345/restorers.git !pip install git+https://github.com/soumik12345/restorers.git
    return


@app.cell
def _():
    import os
    from glob import glob

    import tensorflow as tf

    import wandb
    # import the wandb callbacks for keras
    from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

    from restorers.model.zero_dce import ZeroDCE

    return (
        WandbMetricsLogger,
        WandbModelCheckpoint,
        ZeroDCE,
        glob,
        os,
        tf,
        wandb,
    )


@app.cell
def _(glob, os, tf, wandb):
    wandb.init(project="low-light-enhancement", job_type="train")


    def load_data(image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(
            images=image,
            size=[256, 256]
        )
        image = image / ((2 ** 8) - 1)
        return image


    def data_generator(low_light_images):
        dataset = tf.data.Dataset.from_tensor_slices((low_light_images))
        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(8, drop_remainder=True)
        return dataset


    artifact = wandb.use_artifact("ml-colabs/dataset/LoL:v0", type='dataset')
    artifact_dir = artifact.download()

    train_low_light_images = sorted(glob(os.path.join(artifact_dir, "our485", "low", "*")))
    num_train_images = int((1 - 0.2) * len(train_low_light_images))
    val_low_light_images = train_low_light_images[num_train_images:]
    train_low_light_images = train_low_light_images[:num_train_images]

    train_dataset = data_generator(train_low_light_images)
    val_dataset = data_generator(val_low_light_images)
    return train_dataset, val_dataset


@app.cell
def _(ZeroDCE, tf):
    # define the ZeroDCE model; this gives us a `tf.keras.Model`
    model = ZeroDCE(
        num_intermediate_filters=32, # number of filters in the intermediate convolutional layers
        num_iterations=8, # number of iterations of enhancement
        decoder_channel_factor=1 # factor by which number filters in the decoder of deep curve estimation layer is multiplied
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        weight_exposure_loss=1.0, # weight of the exposure control loss
        weight_color_constancy_loss=0.5, # weight of the color constancy loss
        weight_illumination_smoothness_loss=20, # weight of the illumination smoothness loss
    )
    return (model,)


@app.cell
def _(
    WandbMetricsLogger,
    WandbModelCheckpoint,
    model,
    train_dataset,
    val_dataset,
):
    callbacks = [
        # define the metrics logger callback;
        # we set the `log_freq="batch"` explicitly
        # to the metrics are logged both batch-wise and epoch-wise
        WandbMetricsLogger(log_freq="batch"),
        # define the model checkpoint callback
        WandbModelCheckpoint(
            filepath="checkpoint",
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
            initial_value_threshold=None,
        )
    ]

    # call model.fit()
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks,
    )
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
