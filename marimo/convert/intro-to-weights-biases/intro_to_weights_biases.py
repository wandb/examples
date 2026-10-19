# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "torch>=2.1",
#     "torchvision>=0.16",
#     "wandb>=0.18",
# ]
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


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    # Intro to Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/intro-to-weights-biases/intro_to_weights_biases.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    A Weights & Biases run records one experiment's configuration, metrics, and outputs.
    This lesson starts with simulated metrics, then tracks real PyTorch training,
    prediction tables, and model checkpoints. An optional final example shows
    how to send a Weights & Biases alert.

    ## 1. Authenticate

    Open the notebook with its declared dependencies:

    ```bash
    uvx marimo edit --sandbox intro_to_weights_biases.py
    ```

    The next cell calls `wandb.login()` and selects the projects used by the
    two examples. Set `entity` to your team name, or use `None` for your
    default entity.

    Configure `WANDB_API_KEY` in your environment or molab Secrets. Get a key
    from [Weights & Biases](https://wandb.ai/authorize), or
    [create an account](https://wandb.ai/signup). Terminal-backed sessions can
    also prompt for a key.

    Running the code cells creates experiment runs, downloads MNIST, and uploads
    metrics and checkpoints. The optional alert requires a separate click.
    """)
    return


@app.cell
def _():
    entity = None
    simulation_project = "basic-intro"
    training_project = "pytorch-intro"
    if not wandb.login():
        raise RuntimeError("Configure Weights & Biases credentials before running the lesson.")
    return entity, simulation_project, training_project


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 2. Track Simulated Experiments

    The next cell creates five `simulation_run` objects in sequence. Each call
    to `wandb.init()` starts a separate experiment and records its configuration.
    `simulation_run.log()` sends one accuracy and loss measurement per epoch.

    The fixed seeds make the examples reproducible while giving each run a
    different curve. These values are synthetic: no model or dataset is used.
    Each context manager finishes its run before the next experiment begins.
    """)
    return


@app.cell
def _(entity, simulation_project):
    simulation_run_urls = []
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
            simulation_run_urls.append(simulation_run.url)
    return


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 3. Compare runs in your project

    Open any run returned as stdout in the previous cell. [Compare accuracy and loss across experiments](https://docs.coreweave.com/products/wandb/runs/compare-runs). Each run has its own configuration and metric history.

    Next, we apply the same logging pattern to a real MNIST classifier.

    ## 4. Set the PyTorch Experiment Configuration

    The next cell chooses CUDA when available, otherwise CPU, and defines common
    training settings. We train three fresh models with dropout values 0.1, 0.3,
    and 0.5 so you can compare them in Weights & Biases.

    To keep the lesson short, we use every fifth MNIST example: 12,000 training
    images and 2,000 held-out images. We use the held-out subset for validation
    throughout this example; it is not an independent final test.
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


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 5. Define the Data, Model, and Evaluation Helpers

    The following helpers contain ordinary PyTorch code. They load sampled
    MNIST batches, build an MLP, and evaluate loss and accuracy.

    Validation also returns the first batch's images, predictions, labels, and
    class probabilities for a Weights & Biases prediction table. Its order stays fixed across
    runs. We use `num_workers=0` for compatibility with notebook runtimes.

    Defining these helpers does not download data or train a model.
    """)
    return


@app.cell
def get_dataloader(device):
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
            pin_memory=device.startswith("cuda"),
            num_workers=0,
        )

    return (get_dataloader,)


@app.cell
def get_model(device):
    def get_model(dropout):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10),
        ).to(device)

    return (get_model,)


@app.cell
def evaluate_model(device):
    def evaluate_model(model, validation_loader, loss_function):
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

    return (evaluate_model,)


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 6. Build a Prediction Table

    The next helper creates a `wandb.Table` with one row per image. Each row
    contains a viewable `wandb.Image`, the predicted digit, the true label,
    and ten class probabilities.

    This function builds the table; the training cell logs it to the active run
    after the final validation pass. In Weights & Biases, filter rows where `pred != target`
    to inspect mistakes.
    """)
    return


@app.function
def make_prediction_table(images, predictions, labels, probabilities):
    prediction_table = wandb.Table(
        columns=["image", "pred", "target"] + [f"score_{digit}" for digit in range(10)]
    )
    for image, prediction, label, scores in zip(images, predictions, labels, probabilities):
        prediction_table.add_data(
            wandb.Image(image[0].numpy() * 255),
            prediction.item(),
            label.item(),
            *scores.tolist(),
        )
    return prediction_table


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 7. Train, Log Metrics, and Save Checkpoints

    The next cell runs one experiment per dropout value. `training_run.config`
    records the settings, and every run starts with a fresh model and optimizer.

    `training_run.log()` records training loss by batch, validation metrics by
    epoch, and a prediction table after the final epoch. The scalar summary
    `final_validation_accuracy` makes the final result easy to compare across runs.

    At each epoch, `torch.save(model.state_dict(), ...)` saves model weights.
    `training_run.log_model()` uploads that checkpoint as a model Artifact with
    an epoch alias. The model architecture and dropout are recorded in the run
    configuration; recreate that architecture before loading its weights.

    Checkpoints are kept in a temporary directory until the run finishes logging.
    """)
    return


@app.cell
def _(
    device,
    dropout_values,
    entity,
    evaluate_model,
    get_dataloader,
    get_model,
    training_config,
    training_project,
):
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
                model = get_model(run_config.dropout)
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
                        model, validation_loader, loss_function,
                    )
                    validation_metrics = {
                        "epoch": epoch + 1,
                        "val/val_loss": validation_loss,
                        "val/val_accuracy": validation_accuracy,
                    }
                    if epoch == run_config.epochs - 1:
                        validation_metrics["predictions_table"] = make_prediction_table(*prediction_batch)
                    training_run.log(validation_metrics)

                    checkpoint_path = Path(checkpoint_directory) / "model_weights.pt"
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


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## 8. Inspect Metrics, Predictions, and Models

    Compare dropout settings against validation loss and accuracy in your project.
    Open a run's `predictions_table` to inspect images and class scores.
    In **Artifacts**, find its saved model weights and epoch aliases.

    Each run also records terminal output and available system metrics, which
    can help explain differences in runtime.

    ## 9. Optional: Send an Alert

    The next two cells demonstrate `alert_run.alert()`. Enable notifications
    in your [Weights & Biases user settings](https://wandb.ai/settings), then click
    **Send example alert** to create a separate run and test the notification.

    We log three synthetic accuracy values. The final value crosses a threshold
    of 0.3 and triggers an alert. This is independent of the MNIST results and
    can send a real email or Slack notification through your Weights & Biases settings.
    """)
    return


@app.cell
def _(entity, training_project):
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


@app.cell(hide_code=False)
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
