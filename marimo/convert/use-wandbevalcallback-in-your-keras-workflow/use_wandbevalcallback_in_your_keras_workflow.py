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
app = marimo.App(
    width="medium",
    app_title="Keras evaluation with Weights & Biases",
)

with app.setup:
    import marimo as mo
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow.keras import layers, models

    # Weights and Biases related imports
    import wandb
    from wandb.integration.keras import WandbEvalCallback, WandbMetricsLogger


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-config-logo--dark {
      display: none;
    }

    body.dark .wandb-config-logo--light {
      display: none;
    }

    body.dark .wandb-config-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-config-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-config-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />

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


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This lesson requires an online Weights & Biases run: the evaluation callback uploads
    validation images as an Artifact and retrieves their table for reuse in
    each epoch's prediction table.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `intro-keras`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="intro-keras", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    entity = _entity
    project = _project
    mo.md("✓ Connected to Weights & Biases.")
    return entity, project


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
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
