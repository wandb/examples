# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "torch==2.14.0",
#     "torchvision==0.29.0",
#     "wandb==0.30.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Weights & Biases Tables Quickstart",
)

with app.setup:
    import marimo as mo

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T
    import wandb


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-header-logo--dark {
      display: none;
    }

    body.dark .wandb-header-logo--light {
      display: none;
    }

    body.dark .wandb-header-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-header-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-header-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />

    # View & analyze model predictions during training

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/w-b-tables-quickstart/w_b_tables_quickstart.py/server)

    Train an MNIST classifier and log a Table of labeled prediction examples
    after each epoch. Use the images, predictions, and class probabilities to
    inspect mistakes that an aggregate accuracy score cannot explain.

    Running the following code cells downloads MNIST, trains a model, and logs
    metrics and prediction tables to Weights & Biases.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _notify_submission(_value):
        mo.status.toast(
            "Credentials submitted",
            "Run the next cell to verify them with W&B.",
        )

    wandb_api_form = (
        mo.md("{api_key}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="Forge API key",
                full_width=True,
            )
        )
        .form(
            submit_button_label="Connect to Weights & Biases",
            bordered=True,
            on_change=_notify_submission,
        )
    )

    wandb_api_form
    return (wandb_api_form,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After submitting the API key, run the following cell to verify the
    credentials and load your available W&B entities.
    """)
    return


@app.cell(hide_code=True)
def _():
    from wandb.sdk.lib.service.service_connection import (
        WandbApiFailedError as _WandbApiFailedError,
    )

    mo.stop(wandb_api_form.value is None)

    _submission = wandb_api_form.value
    _api_key = _submission["api_key"].strip()

    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
            verify=True,
            prompt=False,
        )
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    try:
        _api = wandb.Api(timeout=15)
        _viewer = _api.viewer
    except (wandb.errors.Error, _WandbApiFailedError, ValueError):
        mo.stop(
            True,
            mo.callout(
                "W&B verified the credentials but could not retrieve your account entities.",
                kind="danger",
            ),
        )

    available_entities = sorted({
        _value
        for _value in (
            getattr(_viewer, "username", None),
            getattr(_viewer, "entity", None),
            *_viewer.teams,
        )
        if _value
    })
    mo.stop(
        not available_entities,
        mo.callout(
            "W&B authenticated successfully, but no personal or team entities were found.",
            kind="danger",
        ),
    )

    mo.callout(
        mo.md("✓ Credentials verified. Choose a personal or team entity below."),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases personal or team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `table-quickstart`.

    Then select **Use this W&B target**. A project does not need to exist yet;
    W&B creates it when the first run starts if you have write access.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _validate_wandb_target(value):
        if value is None:
            return None
        if not value["entity"]:
            return "Select a W&B entity."
        if not value["project"].strip():
            return "Enter a project name."
        return None

    def _notify_wandb_submission(_value):
        mo.status.toast(
            "W&B target selected",
            "Run the next cell to confirm the destination.",
        )

    wandb_setup_form = (
        mo.md("{entity}\n\n{project}")
        .batch(
            entity=mo.ui.dropdown(
                options=available_entities,
                value=available_entities[0],
                label="W&B entity",
                searchable=True,
                full_width=True,
            ),
            project=mo.ui.text(
                value="table-quickstart",
                label="Project",
                full_width=True,
            ),
        )
        .form(
            submit_button_label="Use this W&B target",
            validate=_validate_wandb_target,
            on_change=_notify_wandb_submission,
            bordered=True,
        )
    )

    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After submitting the target form, run the next cell to confirm the selected
    destination. W&B verifies project write access when the first run starts.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

    mo.callout(
        mo.md(
            f"✓ Using `{entity}/{project}`. W&B will confirm write access "
            "when the first run starts."
        ),
        kind="success",
    )
    return entity, project




@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set the Training and Table Sizes

    The next cell selects a GPU when available and defines the model's
    hyperparameters. One epoch keeps the demo short. Increase `EPOCHS` to compare
    predictions at different points in training.

    We evaluate all 10,000 test images, but log only 32 images from each of the
    first ten test batches. That creates a table with 320 rows per epoch and
    keeps image uploads small.
    """)
    return


@app.cell
def _():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    EPOCHS = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_CLASSES = 10
    L1_SIZE = 32
    L2_SIZE = 64
    CONV_KERNEL_SIZE = 5
    NUM_BATCHES_TO_LOG = 10
    NUM_IMAGES_PER_BATCH = 32
    return (
        BATCH_SIZE,
        CONV_KERNEL_SIZE,
        EPOCHS,
        L1_SIZE,
        L2_SIZE,
        LEARNING_RATE,
        NUM_BATCHES_TO_LOG,
        NUM_CLASSES,
        NUM_IMAGES_PER_BATCH,
        device,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define the Classifier

    The next cell defines a small convolutional network for MNIST. It returns
    ten logits per image, one for each digit. During evaluation, we apply
    softmax to convert those logits into per-class probabilities for the table.

    This is ordinary PyTorch model code. The Weights & Biases integration happens later,
    when we start a run and log its metrics and predictions.
    """)
    return


@app.cell
def _(CONV_KERNEL_SIZE, L1_SIZE, L2_SIZE):
    class ConvNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, L1_SIZE, CONV_KERNEL_SIZE, stride=1, padding=CONV_KERNEL_SIZE // 2),
                nn.BatchNorm2d(L1_SIZE),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(L1_SIZE, L2_SIZE, CONV_KERNEL_SIZE, stride=1, padding=CONV_KERNEL_SIZE // 2),
                nn.BatchNorm2d(L2_SIZE),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.fc = nn.Linear(7 * 7 * L2_SIZE, num_classes)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.reshape(out.size(0), -1)
            return self.fc(out)

    return (ConvNet,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load MNIST

    The next cell downloads MNIST and creates training and test data loaders.
    Training batches are shuffled. Test batches stay in a fixed order so each
    logged image has the same identifier across epochs and experiments.

    The test batch size is twice the training batch size. We use
    `num_workers=0` so loading also works in notebook runtimes without separate
    worker processes.
    """)
    return


@app.cell
def _(BATCH_SIZE):
    train_dataset = torchvision.datasets.MNIST(
        root="./data", train=True, transform=T.ToTensor(), download=True,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, transform=T.ToTensor(), download=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2 * BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, train_loader


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train and Log Prediction Tables

    The next cell starts a Weights & Biases run with `wandb.init()` and records the training
    configuration. Each execution builds a fresh model and optimizer.

    During training, `training_run.log()` sends the loss to Weights & Biases. At the end of
    each epoch, we evaluate the model and build a new `wandb.Table`:

    - `columns` defines the schema: image ID, image, predicted digit, true digit,
      and ten class probabilities.
    - `add_data()` adds one row per image. `wandb.Image()` makes the image
      viewable in Weights & Biases. The remaining values let you filter errors and compare
      class scores.
    - `training_run.log()` uploads accuracy and the table under
      `test_predictions`. Logging a fresh table under the same key each epoch
      lets you inspect predictions over time.

    The context manager finishes the run after training and evaluation complete.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    CONV_KERNEL_SIZE,
    ConvNet,
    EPOCHS,
    L1_SIZE,
    L2_SIZE,
    LEARNING_RATE,
    NUM_BATCHES_TO_LOG,
    NUM_CLASSES,
    NUM_IMAGES_PER_BATCH,
    device,
    entity,
    project,
    test_loader,
    train_loader,
):
    training_config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "l1_size": L1_SIZE,
        "l2_size": L2_SIZE,
        "conv_kernel": CONV_KERNEL_SIZE,
        "num_batches_to_log": NUM_BATCHES_TO_LOG,
        "num_images_per_batch": NUM_IMAGES_PER_BATCH,
    }

    model = ConvNet(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    with wandb.init(project=project, entity=entity, config=training_config) as training_run:
        for epoch in range(EPOCHS):
            model.train()
            for step, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                training_run.log({"loss": loss.item()})
                if (step + 1) % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}/{EPOCHS}, "
                        f"batch {step + 1}/{len(train_loader)}, loss: {loss.item():.4f}"
                    )

            columns = ["id", "image", "guess", "truth"]
            columns.extend(f"score_{digit}" for digit in range(NUM_CLASSES))
            test_table = wandb.Table(columns=columns)

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_index, (images, labels) in enumerate(test_loader):
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    predicted = outputs.argmax(dim=1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if batch_index < NUM_BATCHES_TO_LOG:
                        scores = F.softmax(outputs, dim=1).cpu()
                        for image_index in range(min(NUM_IMAGES_PER_BATCH, len(images))):
                            image_id = batch_index * test_loader.batch_size + image_index
                            test_table.add_data(
                                image_id,
                                wandb.Image(images[image_index].cpu()),
                                predicted[image_index].item(),
                                labels[image_index].item(),
                                *scores[image_index].tolist(),
                            )

            accuracy = 100 * correct / total
            training_run.log({
                "epoch": epoch + 1,
                "acc": accuracy,
                "test_predictions": test_table,
            })
            print(
                f"Epoch {epoch + 1}: test accuracy {accuracy:.2f}% "
                f"on {total:,} images; {len(test_table.data)} table rows"
            )

        run_url = training_run.url
    print(f"View metrics and prediction tables: {run_url}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Explore Your Predictions in Weights & Biases

    Open the run URL printed by the training cell and find the `test_predictions` Table.
    Each row connects an image to its true label, predicted label, and class
    probabilities.

    Filter to rows where `guess != truth` to inspect mistakes. Group by
    `guess` to find patterns in misclassified digits, or compare the
    `score_0` through `score_9` columns to inspect the model's confidence.

    With multiple epochs, use the table's step selector to compare snapshots.
    Across experiments, stable image IDs let you match predictions for the same
    test image. Try changing the learning rate or layer sizes, rerun training,
    and compare the runs in your project. Repeatedly using the test results to
    choose settings turns this split into validation data. Keep an independent
    test set for a final evaluation in your own project.

    ## Next steps

    - See the [Weights & Biases Tables guide](https://docs.wandb.ai/models/tables)
    for more ways to query and compare your results.
    - Explore the [CNN-2: Progress over Training Time report](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess) for examples on how to compare, filter, and group predictions.
    """)
    return


if __name__ == "__main__":
    app.run()
