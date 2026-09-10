# /// script
# dependencies = [
#     "torch==2.14.0",
#     "torchvision==0.29.0",
#     "wandb==0.30.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])

with app.setup:
    import marimo as mo

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as T
    import wandb

    WANDB_PROJECT = "table-quickstart"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # View & analyze model predictions during training

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/w-b-tables-quickstart/w_b_tables_quickstart.py/server)

    <!--- @wandbcode{tables_quickstart} -->
    """)
    return


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

    This quickstart guide covers how to track, visualize, and compare model predictions over the course of training, using PyTorch on MNIST data.

    With [W&B Tables](https://docs.wandb.com/datasets-and-predictions):
    1. Log metrics, images, text, etc. to a `wandb.Table()` during model training or evaluation
    2. View, sort, filter, group, join, interactively query, and explore these tables
    3. Compare model predictions or results: dynamically across specific images, hyperparameters/model versions, or time steps.

    ## Examples
    ### Compare predicted scores for specific images

    [Live example: compare predictions after 1 vs 5 epochs of training →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#compare-predictions-after-1-vs-5-epochs)
    <img src="https://i.imgur.com/NMme6Qj.png" alt="1 epoch vs 5 epochs of training"/>
    The histograms compare per-class scores between the two models. The top green bar in each histogram represents model "CNN-2, 1 epoch" (id 0), which only trained for 1 epoch. The bottom purple bar represents model "CNN-2, 5 epochs" (id 1), which trained for 5 epochs. The images are filtered to cases where the models disagree. For example, in the first row, the "4" gets high scores across all the possible digits after 1 epoch, but after 5 epochs it scores highest on the correct label and very low on the rest.

    ### Focus on top errors over time
    [Live example →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#top-errors-over-time)

    See incorrect predictions (filter to rows where "guess" != "truth") on the full test data. Note that there are 229 wrong guesses after 1 training epoch, but only 98 after 5 epochs.
    <img src="https://i.imgur.com/7g8nodn.png" alt="side by side, 1 vs 5 epochs of training"/>

    ### Compare model performance and find patterns

    [See full detail in a live example →](https://wandb.ai/stacey/table-quickstart/reports/CNN-2-Progress-over-Training-Time--Vmlldzo3NDY5ODU#false-positives-grouped-by-guess)

    Filter out correct answers, then group by the guess to see examples of misclassified images and the underlying distribution of true labels—for two models side-by-side. A model variant with 2X the layer sizes and learning rate is on the left, and the baseline is on the right. Note that the baseline makes slightly more mistakes for each guessed class.
    <img src="https://i.imgur.com/i5PP9AE.png" alt="grouped errors for baseline vs double variant"/>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, your team or entity. You can leave the key blank when this environment already has W&B credentials, including a `WANDB_API_KEY` configured in the Molab Secrets panel.

    If you do not yet have an account, [sign up for W&B](https://wandb.ai/signup). Your entity is the first path segment after `wandb.ai` when you open a W&B workspace.

    This example runs in Molab as a convenient hosted environment, but you can run the same training code anywhere and visualize its metrics and predictions in W&B.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use cached credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="W&B entity or team (optional)",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(api_key=_api_key_input, entity=_entity_input)
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            "Submit the form above to authenticate before creating a run.",
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _entity = wandb_login_form.value["entity"].strip()

    try:
        wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except wandb.errors.Error:
        mo.stop(
            True,
            mo.callout(
                "W&B authentication failed. Check the API key or your Molab Secrets configuration, then submit the form again.",
                kind="danger",
            ),
        )

    wandb_entity = _entity or None
    _entity_note = (
        f" Runs will be logged to `{wandb_entity}`."
        if wandb_entity
        else " Runs will use your default W&B entity."
    )
    mo.callout(
        mo.md(f"W&B is connected.{_entity_note}"),
        kind="success",
    )
    return (wandb_entity,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 0. Setup

    The dependencies are declared in the notebook metadata. The next cells define the PyTorch data loader, model, and training schedule. MNIST is downloaded only after you explicitly start the training workflow.
    """)
    return


@app.cell
def _():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create train and test data loaders.
    def get_dataloader(is_train, batch_size, slice=5):
        "Get a training data loader."
        ds = torchvision.datasets.MNIST(
            root=".",
            train=is_train,
            transform=T.ToTensor(),
            download=True,
        )
        loader = torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=True if is_train else False,
            pin_memory=torch.cuda.is_available(),
            num_workers=2,
        )
        return loader

    return device, get_dataloader


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1. Define the model and training schedule

    * Set the number of epochs to run, where each epoch consists of a training step and a validation (test) step. Optionally configure the amount of data to log per test step. Here the number of batches and number of images per batch to visualize are set low to simplify the demo.
    * Define a simple convolutional neural net (following [pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial) code).
    * Load in train and test sets using PyTorch
    """)
    return


@app.cell
def _(device):
    # Number of epochs to run. Each epoch includes a training step and a
    # test step, so this sets the number of prediction tables to log.
    EPOCHS = 1

    # Keep the amount of prediction data low to simplify the demo.
    NUM_BATCHES_TO_LOG = 10
    NUM_IMAGES_PER_BATCH = 32

    # Training configuration and hyperparameters.
    NUM_CLASSES = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    L1_SIZE = 32
    L2_SIZE = 64
    # Changing this may require changing the shape of adjacent layers.
    CONV_KERNEL_SIZE = 5


    # Define a two-layer convolutional neural network.
    class ConvNet(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, L1_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
                nn.BatchNorm2d(L1_SIZE),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(L1_SIZE, L2_SIZE, CONV_KERNEL_SIZE, stride=1, padding=2),
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

    model = ConvNet(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model
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
        criterion,
        model,
        optimizer,
    )


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md(r"""
    ## 2. Run training and log test predictions

    For every epoch, run a training step and a test step. For each test step, create a `wandb.Table()` in which to store test predictions. These can be visualized, dynamically queried, and compared side by side in your browser.

    The button below downloads MNIST, trains the model for the configured number of epochs, and creates a W&B run containing metrics and prediction tables.
    """),
            mo.callout(
                mo.md(
                    "For faster training in molab, open **Configure compute** and select a GPU before starting the run."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _():
    run_training = mo.ui.run_button(
        label="Train model and log predictions to W&B",
        kind="success",
        tooltip="Downloads MNIST and creates a W&B run",
    )
    run_training
    return (run_training,)


@app.cell
def _(
    BATCH_SIZE,
    CONV_KERNEL_SIZE,
    EPOCHS,
    L1_SIZE,
    L2_SIZE,
    LEARNING_RATE,
    NUM_BATCHES_TO_LOG,
    NUM_CLASSES,
    NUM_IMAGES_PER_BATCH,
    criterion,
    device,
    get_dataloader,
    model,
    optimizer,
    run_training,
    wandb_entity,
):
    mo.stop(
        not run_training.value,
        mo.callout(
            mo.md(
                "Click **Train model and log predictions to W&B** when you are ready to download MNIST and start the run."
            ),
            kind="info",
        ),
    )

    train_loader = get_dataloader(is_train=True, batch_size=BATCH_SIZE)
    test_loader = get_dataloader(is_train=False, batch_size=2 * BATCH_SIZE)

    training_config = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LEARNING_RATE,
        "l1_size": L1_SIZE,
        "l2_size": L2_SIZE,
        "conv_kernel": CONV_KERNEL_SIZE,
        "img_count": min(10000, NUM_IMAGES_PER_BATCH * NUM_BATCHES_TO_LOG),
    }

    # ✨ W&B: Initialize a run and log the training configuration.
    with wandb.init(
        project=WANDB_PROJECT,
        entity=wandb_entity,
        config=training_config,
    ) as run:
        run_url = run.url

        def log_test_predictions(
            images,
            labels,
            outputs,
            predicted,
            test_table,
            log_counter,
        ):
            "Log predictions and per-class scores for one test batch."
            scores = F.softmax(outputs.data, dim=1)
            log_scores = scores.cpu().numpy()
            log_images = images.cpu().numpy()
            log_labels = labels.cpu().numpy()
            log_preds = predicted.cpu().numpy()

            for image_id, (image, label, prediction, score) in enumerate(
                zip(log_images, log_labels, log_preds, log_scores)
            ):
                row_id = f"{image_id}_{log_counter}"
                test_table.add_data(
                    row_id,
                    wandb.Image(image),
                    prediction,
                    label,
                    *score,
                )
                if image_id + 1 == NUM_IMAGES_PER_BATCH:
                    break

        total_step = len(train_loader)
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

                # ✨ W&B: Log loss over training steps for live visualization.
                run.log({"loss": loss.item()})
                if (step + 1) % 100 == 0:
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1,
                            EPOCHS,
                            step + 1,
                            total_step,
                            loss.item(),
                        )
                    )

            columns = ["id", "image", "guess", "truth"]
            columns.extend(f"score_{digit}" for digit in range(NUM_CLASSES))
            test_table = wandb.Table(columns=columns)

            model.eval()
            log_counter = 0
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    if log_counter < NUM_BATCHES_TO_LOG:
                        log_test_predictions(
                            images,
                            labels,
                            outputs,
                            predicted,
                            test_table,
                            log_counter,
                        )
                        log_counter += 1

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            # ✨ W&B: Log accuracy and the prediction table for this epoch.
            run.log(
                {
                    "epoch": epoch,
                    "acc": accuracy,
                    "test_predictions": test_table,
                }
            )
            print(
                "Test accuracy on the 10,000 MNIST test images: "
                f"{accuracy:.2f}%"
            )

    training_result = {
        "run_url": run_url,
        "accuracy": accuracy,
        "epochs": EPOCHS,
    }
    return (training_result,)


@app.cell
def _(training_result):
    _run_url = training_result["run_url"]
    _accuracy = training_result["accuracy"]
    mo.callout(
        mo.md(
            f"""
    Training is complete. [Open the W&B run]({_run_url}) to explore the logged loss, accuracy, and prediction tables.

    Final test accuracy: **{_accuracy:.2f}%**
    """
        ),
        kind="success",
        title="W&B Tables run ready",
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
