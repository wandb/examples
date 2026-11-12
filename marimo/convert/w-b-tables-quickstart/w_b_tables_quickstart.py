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
            "Run the next cell to verify them with Weights & Biases.",
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
    credentials and load your available Weights & Biases entities.
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
                "Weights & Biases verified the credentials but could not retrieve your account entities.",
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
            "Weights & Biases authenticated successfully, but no personal or team entities were found.",
            kind="danger",
        ),
    )

    mo.callout(
        mo.md("✓ Credentials verified. Specify your team entity below."),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `table-quickstart`.

    Then select **Use this Weights & Biases target**.

    > If the project doesn't exist, Weights & Biases creates it when the first run starts, provided you have write access.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _validate_wandb_target(value):
        if value is None:
            return None
        if not value["entity"]:
            return "Select a Weights & Biases entity."
        if not value["project"].strip():
            return "Enter a project name."
        return None

    def _notify_wandb_submission(_value):
        mo.status.toast(
            "Weights & Biases target selected",
            "Run the next cell to confirm the destination.",
        )

    wandb_setup_form = (
        mo.md("{entity}\n\n{project}")
        .batch(
            entity=mo.ui.dropdown(
                options=available_entities,
                value=available_entities[0],
                label="Weights & Biases entity",
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
            submit_button_label="Use this Weights & Biases target",
            validate=_validate_wandb_target,
            on_change=_notify_wandb_submission,
            bordered=True,
        )
    )

    wandb_setup_form
    return (wandb_setup_form,)




@app.cell(hide_code=True)
def _():
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

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
    ## Verify the prediction Table

    Open the run URL and find `test_predictions`. Each row should connect a stable
    image ID with the image, `guess`, `truth`, and ten `score_*` probabilities.
    Filter to `guess != truth`, group by `guess`, and inspect low-margin cases where
    the two largest scores are close.

    Use the step selector to compare epoch snapshots. Across runs, stable IDs let
    you find the same image. Inspect loss and accuracy beside the Table so aggregate
    changes can be connected to predictions. See the
    [Tables guide](https://docs.wandb.ai/models/tables).

    ## Next steps

    Change the learning rate or layer sizes and rerun with the same seed and image
    IDs. Compare fixed, persistent, and new errors. Repeatedly selecting settings
    from these results turns the test split into validation data, so keep an
    independent test set for final evaluation. The
    [progress-over-time Report](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)
    shows additional comparisons.
    """)
    return


if __name__ == "__main__":
    app.run()
