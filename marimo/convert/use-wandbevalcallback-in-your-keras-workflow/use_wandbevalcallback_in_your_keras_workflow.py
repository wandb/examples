# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.2",
#     "tensorflow>=2.16,<3",
#     "tensorflow-datasets>=4.9,<5",
#     "importlib-resources>=6",
#     "wandb>=0.18,<1",
#     "pillow>=10",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Keras evaluation with W&B")

with app.setup:
    import marimo as mo
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import layers, models

    # Weights and Biases related imports
    import wandb
    from wandb.integration.keras import WandbEvalCallback, WandbMetricsLogger


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    # Using Keras Evaluation Callbacks with Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/use-wandbevalcallback-in-your-keras-workflow/use_wandbevalcallback_in_your_keras_workflow.py/server)

    This notebook introduces `WandbEvalCallback`, an abstract callback you can subclass to visualize datasets and model predictions using Weights & Biases Tables. See the [Keras integration](https://docs.wandb.ai/models/integrations/keras).

    Use Weights & Biases for machine learning experiment tracking, dataset
    versioning, and project collaboration.

    A CPU is sufficient; a GPU speeds up training. Running the data and model
    cells downloads Fashion-MNIST and ImageNet weights the first time.

    Run the cells from top to bottom in the marimo editor. Editing a cell
    marks its dependents as stale until you run them again.
    """)
    return


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Set up your experiment and log in

    Get your [Weights & Biases API key](https://wandb.ai/authorize) and set `WANDB_API_KEY` in
    your runtime's environment or **Secrets** panel. Locally, you can also run
    `wandb login` in a terminal. A hosted runtime needs its own credentials.

    Set `entity` to your Weights & Biases team slug, or leave it as `None` to use your default
    entity. The slug appears in `wandb.ai/<entity>/<project>`. Choose a project
    to group your experiments.

    This lesson requires an online Weights & Biases run: the evaluation callback uploads
    validation images as an Artifact and retrieves their table for reuse in
    each epoch's prediction table.
    """)
    return


@app.cell
def _():
    entity = None
    project = "intro-keras"
    wandb.login()
    return entity, project


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Hyperparameters

    Use a Python dictionary to define the experiment. Passing it to
    `wandb.init(config=configs)` records these settings with the run, so you
    can compare experiments and reproduce the model's configuration.
    """)
    return


@app.cell
def _():
    configs = dict(
        dataset="fashion_mnist",
        num_classes=10,
        shuffle_buffer=1024,
        batch_size=64,
        image_size=28,
        image_channels=1,
        learning_rate=0.001,
        epochs=10,
        seed=42,
    )
    return (configs,)


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Dataset

    We use [Fashion-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)
    from the TensorFlow Datasets catalog to build a simple image classification
    pipeline. Labels become one-hot vectors; images keep their 0–255 range
    until the model applies MobileNetV2 preprocessing. Validation examples stay
    in a fixed order.

    > Note: The `get_dataloader()` function is defined at the end of the notebook to keep the focus on the main concepts rather than data-loading implementation details.
    """)
    return


@app.cell
def _(configs):
    train_ds, valid_ds = tfds.load(configs["dataset"], split=["train", "test"])
    trainloader = get_dataloader(train_ds, configs)
    validloader = get_dataloader(valid_ds, configs, dataloader_type="valid")
    return trainloader, validloader


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Model

    A frozen ImageNet MobileNetV2 supplies image features. The model resizes each
    grayscale image to 32 × 32, learns a three-channel convolution, applies
    MobileNetV2 preprocessing, and trains a ten-class classification head.
    """)
    return


@app.function
def get_model(configs, weights="imagenet"):
    backbone = tf.keras.applications.mobilenet_v2.MobileNetV2(
        weights=weights, include_top=False
    )
    backbone.trainable = False

    inputs = layers.Input(
        shape=(configs["image_size"], configs["image_size"], configs["image_channels"])
    )
    resize = layers.Resizing(32, 32)(inputs)
    neck = layers.Conv2D(3, (3, 3), padding="same")(resize)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input(neck)
    x = backbone(preprocess_input, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(configs["num_classes"], activation="softmax")(x)
    return models.Model(inputs=inputs, outputs=outputs)


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Compile model

    Adam uses the learning rate recorded in `configs`. Alongside cross-entropy
    loss, track accuracy and top-five accuracy on training and validation data.
    Rerun this cell before training when you want to start a fresh experiment.
    Running only `model.fit` again continues from the model's current weights.
    """)
    return


@app.cell
def _(configs):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(configs["seed"])
    model = get_model(configs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top@5_accuracy"),
        ],
    )
    model.summary()
    return (model,)


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## `WandbEvalCallback`

    `WandbEvalCallback` is a dataset- and task-independent abstract base class
    for model prediction and dataset visualization. Subclass it and implement
    `add_ground_truth` and `add_model_predictions`.

    The base callback creates data and prediction `wandb.Table` instances,
    logs the ground truth at `on_train_begin`, and logs a new prediction
    Artifact version at `on_epoch_end`. Below, `WandbClfEvalCallback` records
    100 validation images and their predicted classes at every epoch.

    Once the validation table uploads, the base callback waits for its Artifact
    and exposes the stored table as `data_table_ref`. Prediction rows reuse
    `self.data_table_ref.data[idx][n]` instead of uploading each image again.
    """)
    return


@app.class_definition
class WandbClfEvalCallback(WandbEvalCallback):
    def __init__(
        self, validloader, data_table_columns, pred_table_columns, num_samples=100
    ):
        super().__init__(data_table_columns, pred_table_columns)
        self.val_data = validloader.unbatch().take(num_samples).cache()

    def add_ground_truth(self, logs=None):
        for idx, (image, label) in enumerate(self.val_data):
            self.data_table.add_data(
                idx, wandb.Image(image.numpy()), int(tf.argmax(label, axis=-1))
            )

    def add_model_predictions(self, epoch, logs=None):
        # Get predictions
        preds = self._inference()
        table_idxs = self.data_table_ref.get_index()
        for idx in table_idxs:
            self.pred_table.add_data(
                epoch,
                self.data_table_ref.data[idx][0],
                self.data_table_ref.data[idx][1],
                self.data_table_ref.data[idx][2],
                preds[idx],
            )

    def _inference(self):
        preds = []
        for images, labels in self.val_data.batch(32):
            predictions = self.model(images, training=False)
            preds.extend(tf.argmax(predictions, axis=-1).numpy().tolist())
        return preds


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Train

    Create the Weights & Biases run before constructing its callbacks. `WandbMetricsLogger`
    records metrics every ten batches and at epoch end; `WandbClfEvalCallback`
    logs ground truth once and predictions after each epoch. The context
    manager closes the run when training finishes or raises an exception.
    """)
    return


@app.cell
def _(configs, entity, model, project, trainloader, validloader):
    with wandb.init(entity=entity, project=project, config=configs) as run:
        history = model.fit(
            trainloader,
            epochs=configs["epochs"],
            validation_data=validloader,
            shuffle=False,  # The tf.data training pipeline already shuffles.
            callbacks=[
                WandbMetricsLogger(log_freq=10),
                WandbClfEvalCallback(
                    validloader,
                    data_table_columns=["idx", "image", "ground_truth"],
                    pred_table_columns=["epoch", "idx", "image", "ground_truth", "prediction"],
                    num_samples=100,
                ),
            ],
        )
    return


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Verify and next steps

    Open the run link printed by Weights & Biases in the previous cell. Select the **Artifacts** tab in the Weights & Biases UI to view the `val` artifact you logged. In it, you will find a table with 100 validation images and labels.

    Select the **Workspaces** from the proejct sidebar to view the training and validation loss,
    accuracy, and top-five accuracy charts.

    Try changing the learning rate, recreate and compile the model, then train
    again. Use the same project to compare the runs. To adapt the callback to
    another task, change the table columns and the two methods that add ground
    truth and predictions.

    ## Helper functions

    The input pipeline maps images and labels, shuffles only training examples,
    then batches and prefetches them.
    """)
    return


@app.function
def parse_data(example, num_classes):
    # Get image
    image = example["image"]
    image = tf.cast(image, tf.float32)

    # Get label
    label = example["label"]
    label = tf.one_hot(label, depth=num_classes)
    return image, label


@app.function
def get_dataloader(ds, configs, dataloader_type="train"):
    dataloader = ds.map(
        lambda example: parse_data(example, configs["num_classes"]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if dataloader_type == "train":
        dataloader = dataloader.shuffle(configs["shuffle_buffer"], seed=configs["seed"])
    return dataloader.batch(configs["batch_size"]).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    app.run()
