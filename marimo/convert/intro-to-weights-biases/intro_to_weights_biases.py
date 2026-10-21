# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "torch>=2.1",
#     "torchvision>=0.16",
#     "wandb>=0.18",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Intro to Weights & Biases")

with app.setup:
    import random
    import tempfile
    from pathlib import Path

    import marimo as mo
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
    from torchvision.datasets import MNIST
    import wandb

    MNIST.mirrors = [
        mirror for mirror in MNIST.mirrors
        if "http://yann.lecun.com/" not in mirror
    ]


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

    # Introduction to Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/intro-to-weights-biases/intro_to_weights_biases.py/server)

    A Weights & Biases run records one experiment's configuration, metrics, and outputs.
    This lesson starts with simulated metrics, then tracks real PyTorch training,
    prediction tables, and model checkpoints. An optional final example shows
    how to send a Weights & Biases alert.

    Running the code cells creates experiment runs, downloads MNIST, and uploads
    metrics and checkpoints. The optional alert has its own Python setting
    near the end of the lesson.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose projects for simulated metrics and training runs. The fields below contain this lesson’s defaults. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{simulation_project}\n\n{training_project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        simulation_project=mo.ui.text(value="basic-intro", label="Simulation project", full_width=True),
        training_project=mo.ui.text(value="pytorch-intro", label="Training project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _simulation_project = _submission["simulation_project"].strip()
    _training_project = _submission["training_project"].strip()
    mo.stop(
        not _entity or not _simulation_project or not _training_project,
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
    simulation_project = _simulation_project
    training_project = _training_project
    mo.md("✓ Connected to Weights & Biases.")
    return entity, simulation_project, training_project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Track Simulated Experiments

    The next cell creates five `simulation_run` objects in sequence. Each call
    to `wandb.init()` starts a separate experiment and records its configuration.
    `simulation_run.log()` sends one accuracy and loss measurement per epoch.

    The fixed seeds make the examples reproducible while giving each run a
    different curve. These values are generated mathematically, without training
    a model or loading a dataset.
    Each context manager finishes its run before the next experiment begins.
    """)
    return


@app.cell
def _(entity, simulation_project):
    for experiment_index in range(5):
        simulation_config = {"epochs": 10, "seed": experiment_index, "dataset": "synthetic"}
        simulation_rng = random.Random(simulation_config["seed"])
        offset = simulation_rng.random() / 5
        with wandb.init(
            project=simulation_project,
            entity=entity,
            name=f"simulation-{experiment_index + 1}",
            config=simulation_config,
        ) as simulation_run:
            for simulation_epoch in range(1, simulation_config["epochs"] + 1):
                simulated_accuracy = 1 - 2 ** -simulation_epoch - simulation_rng.random() / 10 - offset
                simulated_loss = 2 ** -simulation_epoch + simulation_rng.random() / 10 + offset
                simulation_run.log({
                    "epoch": simulation_epoch,
                    "acc": simulated_accuracy,
                    "loss": simulated_loss,
                })
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compare runs in your project

    Open a run link printed by Weights & Biases in the previous cell. In the simulation project you selected, select the five simulation runs and compare their `acc` and `loss`
    curves. Choose `epoch` as the horizontal axis to compare the same training
    point across runs. Inspect each run's config to see its seed and epoch count.

    Next, we apply the same logging pattern to a real MNIST classifier.

    ## Set the PyTorch Experiment Configuration

    The next cell chooses CUDA when available, otherwise CPU, and defines common
    training settings. We train three fresh models with dropout values 0.1, 0.3,
    and 0.5 so you can compare them in Weights & Biases.

    To keep the lesson short, we use every fifth MNIST example. This gives us
    12,000 training images and 2,000 held-out images. We repeatedly evaluate the
    held-out subset when comparing dropout rates, so it serves as validation
    data. An independent final evaluation would require another test set.
    """)
    return


@app.cell
def _():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    training_config = {
        "epochs": 5,
        "batch_size": 128,
        "lr": 1e-3,
        "seed": 42,
        "sample_stride": 5,
        "architecture": "MLP",
        "dataset": "MNIST",
    }
    dropout_values = [0.1, 0.3, 0.5]
    return device, dropout_values, training_config


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load MNIST batches

    `get_dataloader()` loads MNIST into temporary storage and selects every
    `sample_stride`-th image. `ToTensor()` converts each grayscale image to a
    tensor with pixel values between 0 and 1.

    Training batches are shuffled. Validation batches retain the same order
    across runs so we can compare predictions on the same images. Calling
    this helper downloads missing data. `num_workers=0` works across notebook
    runtimes, and pinned memory helps transfer batches to CUDA when available.
    """)
    return


@app.function
def get_dataloader(is_train, batch_size, sample_stride):
    dataset = MNIST(
        root=Path(tempfile.gettempdir()) / "wandb-intro-mnist",
        train=is_train,
        transform=T.ToTensor(),
        download=True,
    )
    sampled_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), sample_stride),
    )
    return torch.utils.data.DataLoader(
        sampled_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Build a classifier to compare dropout rates

    The MLP flattens each 28-by-28 image, learns 256 hidden features, and outputs
    ten logits, one per digit. Cross-entropy loss accepts these logits directly.
    `dropout` controls how many hidden activations are randomly zeroed during
    training. Each call creates a fresh model on the selected device.
    """)
    return


@app.function
def get_model(dropout, device):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10),
    ).to(device)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Evaluate on the held-out images

    `model.eval()` turns off dropout and uses batch normalization's learned
    statistics. `torch.inference_mode()` avoids tracking gradients during
    validation. The helper averages loss by image count, so a smaller final
    batch contributes the correct weight, and computes accuracy across the
    entire validation subset.

    It also returns the first batch's images, predictions, labels, and softmax
    probabilities on the CPU. We will use that fixed batch to build a Weights & Biases Table.
    """)
    return


@app.function
def evaluate_model(model, validation_loader, loss_function, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    prediction_batch = None
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(validation_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            loss_sum += loss_function(outputs, labels).item() * len(labels)
            correct += (predictions == labels).sum().item()
            if batch_index == 0:
                prediction_batch = (
                    images.cpu(), predictions.cpu(), labels.cpu(),
                    outputs.softmax(dim=1).cpu(),
                )
    return (
        loss_sum / len(validation_loader.dataset),
        correct / len(validation_loader.dataset),
        prediction_batch,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Build a Prediction Table

    The next helper function creates a `wandb.Table` with one row per image. Each row
    contains a viewable `wandb.Image`, the predicted digit, the true label,
    and ten class probabilities.
    """)
    return


@app.function
def make_prediction_table(images, predictions, labels, probabilities):
    prediction_table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{digit}" for digit in range(10)]
    )
    for image, prediction, label, scores in zip(images, predictions, labels, probabilities):
        prediction_table.add_data(
            wandb.Image(image),
            prediction.item(),
            label.item(),
            *scores.tolist(),
        )
    return prediction_table


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train, Log Metrics, and Save Checkpoints

    The next cell runs one experiment per dropout value. `training_run.config`
    records the settings, and every run starts with a fresh model and optimizer.

    `training_run.log()` records training loss by batch, validation metrics by
    epoch, and a prediction table after the final epoch. The scalar summary
    `final_validation_accuracy` makes the final result easy to compare across runs.

    At each epoch, `torch.save(model.state_dict(), ...)` saves model weights
    to a separate file so later epochs cannot overwrite an earlier checkpoint.
    `training_run.log_model()` uploads that checkpoint as a model Artifact with
    an epoch alias. The model architecture and dropout are recorded in the run
    configuration. Recreate that architecture before loading the saved weights.

    Checkpoints are kept in a temporary directory until the run finishes logging.
    """)
    return


@app.cell
def _(device, dropout_values, entity, training_config, training_project):
    for dropout in dropout_values:
        torch.manual_seed(training_config["seed"])
        with tempfile.TemporaryDirectory(prefix="wandb-mnist-checkpoints-") as checkpoint_directory:
            with wandb.init(
                project=training_project,
                entity=entity,
                config={**training_config, "dropout": dropout},
            ) as training_run:
                run_config = training_run.config
                train_loader = get_dataloader(True, run_config.batch_size, run_config.sample_stride)
                validation_loader = get_dataloader(False, 2 * run_config.batch_size, run_config.sample_stride)
                model = get_model(run_config.dropout, device)
                loss_function = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=run_config.lr)
                example_count = 0

                for epoch in range(run_config.epochs):
                    model.train()
                    for step, (images, labels) in enumerate(train_loader):
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        train_loss = loss_function(outputs, labels)
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()
                        example_count += len(images)
                        training_run.log({
                            "train/train_loss": train_loss.item(),
                            "train/epoch": epoch + (step + 1) / len(train_loader),
                            "train/example_ct": example_count,
                        })

                    validation_loss, validation_accuracy, prediction_batch = evaluate_model(
                        model, validation_loader, loss_function, device,
                    )
                    validation_metrics = {
                        "epoch": epoch + 1,
                        "val/val_loss": validation_loss,
                        "val/val_accuracy": validation_accuracy,
                    }
                    if epoch == run_config.epochs - 1:
                        validation_metrics["predictions_table"] = make_prediction_table(*prediction_batch)
                    training_run.log(validation_metrics)

                    checkpoint_path = Path(checkpoint_directory) / f"model_weights_epoch_{epoch + 1}.pt"
                    torch.save(model.state_dict(), checkpoint_path)
                    training_run.log_model(
                        str(checkpoint_path),
                        name=f"mnist-{training_run.id}",
                        aliases=[f"epoch-{epoch + 1}"],
                    )
                    print(
                        f"Dropout {dropout}, epoch {epoch + 1}: "
                        f"validation loss {validation_loss:.3f}, accuracy {validation_accuracy:.2%}"
                    )

                training_run.summary["final_validation_accuracy"] = validation_accuracy
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Inspect Metrics, Predictions, and Models

    In your selected training project, compare the three runs' `val/val_loss` and
    `val/val_accuracy` curves. Use the config column `dropout` to identify each
    experiment and `final_validation_accuracy` to compare their final results.
    Open a run's `predictions_table` and filter `pred != target` to inspect
    mistakes alongside the ten class probabilities. In **Artifacts**, open its
    `mnist-<run-id>` model collection and inspect the versions with `epoch-1`,
    `epoch-2`, and later aliases.

    ## Optional: Send an Alert

    The next cell demonstrates `alert_run.alert()`. To try it, enable notifications
    in your [Weights & Biases user settings](https://wandb.ai/settings) and change
    `send_example_alert` to `True`. This creates a separate run and can send a
    real notification through the email or Slack settings you configured.

    We log three synthetic accuracy values. The final value crosses a threshold
    of 0.3 and triggers an alert. This is independent of the MNIST results and
    does not run until you opt in with the Python setting below.
    """)
    return


@app.cell
def _(entity, training_project):
    send_example_alert = False

    if send_example_alert:
        with wandb.init(project=training_project, entity=entity, job_type="alert-demo") as alert_run:
            accuracy_threshold = 0.3
            for alert_step, alert_accuracy in enumerate([0.8, 0.65, 0.25]):
                alert_run.log({"step": alert_step, "accuracy": alert_accuracy})
                if alert_accuracy < accuracy_threshold:
                    alert_run.alert(
                        title="Low accuracy",
                        text=f"Demo accuracy {alert_accuracy} is below {accuracy_threshold} at step {alert_step}.",
                    )
                    print(f"Alert triggered: {alert_run.url}")
                    break
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Next Steps

    You have used the same run/log pattern for simulated metrics and real
    training, then added Tables, checkpoint Artifacts, and an optional alert.

    Next, automate hyperparameter comparisons with
    [Weights & Biases Sweeps using PyTorch](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/organizing-hyperparameter-sweeps-in-pytorch-with-w-b/organizing_hyperparameter_sweeps_in_pytorch_with_w_b.py/server).
    See the [Weights & Biases Alerts guide](https://docs.wandb.ai/guides/track/alert)
    for notification settings.
    """)
    return


if __name__ == "__main__":
    app.run()
