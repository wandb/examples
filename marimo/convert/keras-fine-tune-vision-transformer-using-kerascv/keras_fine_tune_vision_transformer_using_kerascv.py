# /// script
# dependencies = ["keras-cv", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/Fine_tune_Vision_Transformer_using_KerasCV.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{keras-vit} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Installations and Imports
    <!--- @wandbcode{keras-vit} -->
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: keras-cv !pip install -qq keras-cv
    # packages added via marimo's package management: wandb !pip install -qq wandb
    return


@app.cell
def _():
    import numpy as np
    from argparse import Namespace

    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import layers
    from tensorflow.keras import models

    import keras_cv as kcv
    from keras_cv.models import ViTTiny16
    from keras_cv.layers import preprocessing

    import wandb
    from wandb.keras import WandbMetricsLogger
    from wandb.keras import WandbEvalCallback

    return (
        Namespace,
        ViTTiny16,
        WandbEvalCallback,
        WandbMetricsLogger,
        np,
        preprocessing,
        tf,
        tfds,
        wandb,
    )


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hyperparameters
    """)
    return


@app.cell
def _(Namespace):
    configs = Namespace(
        learning_rate = 1e-4,
        batch_size = 64,
        num_epochs = 10,
        image_size = 224,
        num_classes = 120,
        num_steps = 1.0,
    )
    return (configs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset and Dataloaders
    """)
    return


@app.cell
def _(configs, preprocessing, tf, tfds):
    AUTOTUNE = tf.data.AUTOTUNE


    def parse_data(example):
        "Apply preprocessing to one data sample at a time."
        image = example["image"]
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (configs.image_size, configs.image_size))

        label = example["label"]
        label = tf.one_hot(label, configs.num_classes)

        return image, label


    base_augmentations = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(factor=0.02),
            tf.keras.layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="base_augmentation",
    )

    mixup = preprocessing.MixUp(alpha=0.8)


    def apply_base_augmentations(images, labels):
        images = base_augmentations(images)
        return images, labels


    ds_train, ds_test = tfds.load('stanford_dogs', split=['train', 'test'])

    trainloader = (
        ds_train
        .map(parse_data, num_parallel_calls=AUTOTUNE)
        .batch(configs.batch_size)
        .map(apply_base_augmentations, num_parallel_calls=AUTOTUNE)
        .map(lambda images, labels: mixup({"images": images, "labels": labels}), num_parallel_calls=AUTOTUNE)
        .map(lambda x: (x["images"], x["labels"]), num_parallel_calls=AUTOTUNE)
        .shuffle(1024)
        .prefetch(AUTOTUNE)
    )

    testloader = (
        ds_test
        .map(parse_data, num_parallel_calls=AUTOTUNE)
        .batch(configs.batch_size)
        .prefetch(AUTOTUNE)
    )
    return testloader, trainloader


@app.cell
def _(ViTTiny16, configs, tf):
    def get_model():
        inputs = tf.keras.layers.Input(shape=(configs.image_size, configs.image_size, 3))

        vit = ViTTiny16(
            include_rescaling=False,
            include_top=False,
            name="ViTTiny32",
            weights="imagenet",
            input_tensor=inputs,
            pooling="token_pooling",
            activation=tf.keras.activations.gelu,
        )
    
        vit.trainable = True

        outputs = tf.keras.layers.Dense(configs.num_classes, activation="softmax")(vit.output)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    model = get_model()
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compile the Model

    We will use `CosineDecay` learning rate scheduler.
    """)
    return


@app.cell
def _(configs, model, tf, trainloader):
    total_steps = len(trainloader)*configs.num_epochs
    decay_steps = total_steps * configs.num_steps

    cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
        configs.learning_rate, decay_steps, alpha=0.1
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay_scheduler),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # [OPTIONAL] Model Prediction Visualization

    We will build a custom Keras callback by subclassing `WandbEvalCallback` for model prediction visualization.
    """)
    return


@app.cell
def _(WandbEvalCallback, np, tf, wandb):
    class WandbClfEvalCallback(WandbEvalCallback):
        def __init__(
            self, validloader, data_table_columns, pred_table_columns, num_samples=100
        ):
            super().__init__(data_table_columns, pred_table_columns)

            self.val_data = validloader.unbatch().take(num_samples)

        def add_ground_truth(self, logs=None):
            for idx, (image, label) in enumerate(self.val_data):
                self.data_table.add_data(
                    idx,
                    wandb.Image(image),
                    np.argmax(label, axis=-1)
                )

        def add_model_predictions(self, epoch, logs=None):
            # Get predictions
            preds = self._inference()
            table_idxs = self.data_table_ref.get_index()

            for idx in table_idxs:
                pred = preds[idx]
                self.pred_table.add_data(
                    epoch,
                    self.data_table_ref.data[idx][0],
                    self.data_table_ref.data[idx][1],
                    self.data_table_ref.data[idx][2],
                    pred
                )

        def _inference(self):
          preds = []
          for image, label in self.val_data:
              pred = self.model(tf.expand_dims(image, axis=0))
              argmax_pred = tf.argmax(pred, axis=-1).numpy()[0]
              preds.append(argmax_pred)

          return preds

    return (WandbClfEvalCallback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train the model with W&B
    """)
    return


@app.cell
def _(
    WandbClfEvalCallback,
    WandbMetricsLogger,
    configs,
    model,
    testloader,
    trainloader,
    wandb,
):
    # Initialize a W&B run
    run = wandb.init(
        project="keras_cv_vit",
        save_code=False,
        config=vars(configs),
    )

    # Fine-tune the model
    model.fit(
        trainloader,
        epochs=configs.num_epochs,
        validation_data=testloader,
        callbacks=[
            WandbMetricsLogger(log_freq=2),
            WandbClfEvalCallback(
                validloader = testloader,
                data_table_columns = ["idx", "image", "label"],
                pred_table_columns = ["epoch", "idx", "image", "label", "pred"]
            )
        ],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Evaluation
    """)
    return


@app.cell
def _(model, testloader, wandb):
    eval_loss, eval_acc = model.evaluate(testloader)
    wandb.log({
        "eval_loss": eval_loss,
        "eval_acc": eval_acc
    })
    return


@app.cell
def _(wandb):
    # Close the W&B run
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
