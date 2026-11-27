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

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Intro to Weights & Biases")


with app.setup:
    import math
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
        mirror
        for mirror in MNIST.mirrors
        if "http://yann.lecun.com/" not in mirror
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Intro to Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/intro-to-weights-biases/intro_to_weights_biases.py/server)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Use [W&B](https://wandb.ai/site?utm_source=intro_colab&utm_medium=code&utm_campaign=intro) for machine learning experiment tracking, model checkpointing, collaboration with your team and more. See the full W&B Documentation [here](https://docs.wandb.ai/).

    In this notebook, you will create and track a machine learning experiment using a simple PyTorch model. By the end of the notebook, you will have an interactive project dashboard that you can share and customize with other members of your team. [View an example dashboard here](https://wandb.ai/wandb/wandb_example).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    The notebook's dependencies install the W&B SDK automatically. Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, your team entity. You can leave the key blank when this environment already has credentials, such as `WANDB_API_KEY` configured in the marimo Secrets panel. If you are new to W&B, [create a free account](https://wandb.ai/signup).
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use configured credentials",
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
            mo.md("Connect to W&B before creating runs."),
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _entity = wandb_login_form.value["entity"].strip()
    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
        )
        _login_error = None
    except wandb.errors.Error as _error:
        _login_ok = False
        _login_error = str(_error)

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key and "
                f"try again.\n\nW&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {"entity": _entity or None}
    mo.callout(mo.md("Connected to W&B."), kind="success")
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Simulate and track a machine learning experiment with W&B

    Create, track, and visualize a machine learning experiment. To do this:

    1. Initialize a [W&B run](https://docs.wandb.ai/guides/runs) and pass in the hyperparameters you want to track.
    2. Within your training loop, log metrics such as the accuracy and loss.
    """)
    return


@app.function
def run_simulated_experiments(wandb_settings):
    """Create five runs with synthetic accuracy and loss metrics."""
    if wandb.run is not None:
        wandb.finish()

    # Launch 5 simulated experiments
    total_runs = 5
    run_urls = []
    for run_index in range(total_runs):
        # 1. Start a new run to track this script
        with wandb.init(
            # Set the project where this run will be logged
            project="basic-intro",
            entity=wandb_settings["entity"],
            # Pass a run name (otherwise it is randomly assigned)
            name=f"experiment_{run_index}",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.02,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 10,
            },
        ) as basic_run:
            # This simple block simulates a training loop logging metrics
            epochs = 10
            offset = random.random() / 5
            for epoch in range(2, epochs):
                accuracy = 1 - 2 ** (-epoch) - random.random() / epoch - offset
                loss = 2 ** (-epoch) + random.random() / epoch + offset

                # 2. Log metrics from your script to W&B
                basic_run.log({"acc": accuracy, "loss": loss})
            run_urls.append(basic_run.url)
        # Exiting the context manager marks the run as finished

    return run_urls


@app.cell(hide_code=True)
def _(run_simulated_experiments, wandb_settings):
    simulate_runs_button = mo.ui.run_button(
        label="Create five simulated W&B runs"
    )
    _entity_note = (
        f" for entity `{wandb_settings['entity']}`"
        if wandb_settings["entity"]
        else " using your default entity"
    )
    mo.vstack(
        [
            mo.md(
                f"Clicking the button runs `{run_simulated_experiments.__name__}` "
                "to create five runs in the `basic-intro` project and log "
                f"synthetic accuracy and loss metrics{_entity_note}."
            ),
            simulate_runs_button,
        ]
    )
    return (simulate_runs_button,)


@app.cell(hide_code=True)
def _(simulate_runs_button, wandb_settings):
    mo.stop(
        not simulate_runs_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to create the runs."),
            kind="info",
        ),
    )

    _simulated_run_urls = run_simulated_experiments(wandb_settings)
    _run_links = "\n".join(
        f"- [Experiment {index}]({url})"
        for index, url in enumerate(_simulated_run_urls, start=1)
    )
    mo.callout(
        mo.md(f"Created five runs:\n\n{_run_links}"),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    View how your machine learning peformed in your W&B project. Copy and paste the URL link that is printed from the previous cell. The URL will redirect you to a W&B project that contains a dashboard showing graphs the show how

    The following image shows what a dashboard can look like:
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ![](https://i.imgur.com/Pell4Oo.png)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we know how to integrate W&B into a pseudo machine learning training loop, let's track a machine learning experiment using a basic PyTorch neural network. The following code will also upload model checkpoints to W&B that you can then share with other teams in in your organization.

    ## Track a machine learning experiment using PyTorch

    The following code cell defines and trains a simple MNIST classifier. During training, you will see W&B prints out URLs. Click on the project page link to see your results stream in live to a W&B project.

    W&B runs automatically log [metrics](https://docs.wandb.ai/ref/app/pages/run-page#charts-tab),
    [system information](https://docs.wandb.ai/ref/app/pages/run-page#system-tab),
    [hyperparameters](https://docs.wandb.ai/ref/app/pages/run-page#overview-tab),
    [terminal output](https://docs.wandb.ai/ref/app/pages/run-page#logs-tab) and
    you'll see an [interactive table](https://docs.wandb.ai/guides/data-vis)
    with model inputs and outputs.

    ### Set up PyTorch Dataloader
    The following cell defines some useful functions that we will need to train our machine learning model. The functions themselves are not unique to W&B so we'll not cover them in detail here. See the PyTorch documentation for more information on how to define [forward and backward training loop](https://pytorch.org/tutorials/beginner/nn_tutorial.html), how to use [PyTorch DataLoaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) to load data in for training, and how define PyTorch models using the [`torch.nn.Sequential` Class](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html).
    """)
    return


@app.function
def get_dataloader(is_train, batch_size, sample_stride=5):
    """Get a sampled MNIST training or validation DataLoader."""
    dataset = MNIST(
        root=Path(tempfile.gettempdir()) / "wandb-intro-mnist",
        train=is_train,
        transform=T.ToTensor(),
        download=True,
    )
    sampled_dataset = torch.utils.data.Subset(
        dataset,
        indices=range(0, len(dataset), sample_stride),
    )
    return torch.utils.data.DataLoader(
        dataset=sampled_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=device.startswith("cuda"),
        num_workers=2,
    )


@app.function
def get_model(dropout):
    """Build the small multilayer perceptron used in this tutorial."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10),
    ).to(device)


@app.function
def validate_model(
    training_run,
    model,
    valid_dl,
    loss_func,
    log_images=False,
    batch_idx=0,
):
    """Evaluate the model and optionally log one prediction table."""
    model.eval()
    validation_loss = 0.0
    correct = 0
    with torch.inference_mode():
        for index, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            validation_loss += loss_func(outputs, labels).item() * labels.size(0)

            # Compute accuracy and accumulate
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

            # Log one batch of images to the dashboard, always same batch_idx.
            if index == batch_idx and log_images:
                log_image_table(
                    training_run,
                    images,
                    predicted,
                    labels,
                    outputs.softmax(dim=1),
                )

    return (
        validation_loss / len(valid_dl.dataset),
        correct / len(valid_dl.dataset),
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Create a table to compare the predicted values versus the true value

    The following cell is unique to W&B, so let's go over it.

    In the cell we define a function called `log_image_table`. Though technically, optional, this function creates a W&B Table object. We will use the table object to create a table that shows what the model predicted for each image.

    More specifically, each row will conists of the image fed to the model, along with predicted value and the actual value (label).
    """)
    return


@app.function
def log_image_table(training_run, images, predicted, labels, probabilities):
    """Log a W&B Table with images, predictions, labels, and class scores."""
    # Create a W&B Table to log images, labels, and predictions
    table = wandb.Table(
        columns=["image", "pred", "target"]
        + [f"score_{index}" for index in range(10)]
    )
    for image, prediction, target, probability in zip(
        images.to("cpu"),
        predicted.to("cpu"),
        labels.to("cpu"),
        probabilities.to("cpu"),
    ):
        table.add_data(
            wandb.Image(image[0].numpy() * 255),
            int(prediction),
            int(target),
            *probability.numpy(),
        )
    training_run.log({"predictions_table": table}, commit=False)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Train your model and upload checkpoints

    The following code trains and saves model checkpoints to your project. Use model checkpoints like you normally would to assess how the model performed during training.

    W&B also makes it easy to share your saved models and model checkpoints with other members of your team or organization. To learn how to share your model and model checkpoints with members outside of your team, see [W&B Registry](https://docs.wandb.ai/guides/registry).
    """)
    return


@app.function
def train_mnist_experiments(wandb_settings):
    """Train three MNIST models and log metrics and checkpoints to W&B."""
    if wandb.run is not None:
        wandb.finish()

    run_urls = []
    # Launch 3 experiments, trying different dropout rates
    for _experiment_index in range(3):
        with tempfile.TemporaryDirectory() as checkpoint_directory:
            # Initialize a W&B run. The context manager closes it after training.
            with wandb.init(
                project="pytorch-intro",
                entity=wandb_settings["entity"],
                config={
                    "epochs": 5,
                    "batch_size": 128,
                    "lr": 1e-3,
                    "dropout": random.uniform(0.01, 0.80),
                },
            ) as training_run:
                # Copy your config
                config = training_run.config

                # Get the data
                train_dl = get_dataloader(
                    is_train=True,
                    batch_size=config.batch_size,
                )
                valid_dl = get_dataloader(
                    is_train=False,
                    batch_size=2 * config.batch_size,
                )
                n_steps_per_epoch = math.ceil(
                    len(train_dl.dataset) / config.batch_size
                )

                # A simple MLP model
                model = get_model(config.dropout)

                # Make the loss and optimizer
                loss_func = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

                # Training
                example_count = 0

                for epoch in range(config.epochs):
                    model.train()
                    for step, (images, labels) in enumerate(train_dl):
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        train_loss = loss_func(outputs, labels)
                        optimizer.zero_grad()
                        train_loss.backward()
                        optimizer.step()

                        example_count += len(images)
                        train_metrics = {
                            "train/train_loss": train_loss.item(),
                            "train/epoch": (
                                step + 1 + n_steps_per_epoch * epoch
                            )
                            / n_steps_per_epoch,
                            "train/example_ct": example_count,
                        }
                        if step + 1 < n_steps_per_epoch:
                            # Log train metrics to W&B
                            training_run.log(train_metrics)

                    validation_loss, accuracy = validate_model(
                        training_run,
                        model,
                        valid_dl,
                        loss_func,
                        log_images=epoch == config.epochs - 1,
                    )

                    # Log train and validation metrics to W&B
                    validation_metrics = {
                        "val/val_loss": validation_loss,
                        "val/val_accuracy": accuracy,
                    }
                    training_run.log({**train_metrics, **validation_metrics})

                    # Save the model checkpoint to W&B
                    checkpoint_path = Path(checkpoint_directory) / "my_model.pt"
                    torch.save(model, checkpoint_path)
                    training_run.log_model(
                        str(checkpoint_path),
                        "my_mnist_model",
                        aliases=[
                            f"epoch-{epoch + 1}_dropout-{round(config.dropout, 4)}"
                        ],
                    )
                    print(
                        f"Epoch: {epoch + 1}, "
                        f"Train Loss: {train_loss.item():.3f}, "
                        f"Valid Loss: {validation_loss:.3f}, "
                        f"Accuracy: {accuracy:.2f}"
                    )

                # If you had a test set, this is where you could log it as a
                # Summary metric. This tutorial records validation accuracy.
                training_run.summary["final_validation_accuracy"] = accuracy
                run_urls.append(training_run.url)

    return run_urls


@app.cell(hide_code=True)
def _(train_mnist_experiments, wandb_settings):
    train_models_button = mo.ui.run_button(
        label="Download MNIST, train three models, and log to W&B"
    )
    _entity_note = (
        f" for entity `{wandb_settings['entity']}`"
        if wandb_settings["entity"]
        else " using your default entity"
    )
    mo.vstack(
        [
            mo.md(
                f"This runs `{train_mnist_experiments.__name__}` to download "
                "MNIST, train three five-epoch models, and upload metrics, "
                f"prediction tables, and model checkpoints{_entity_note}."
            ),
            train_models_button,
        ]
    )
    return (train_models_button,)


@app.cell(hide_code=True)
def _(train_models_button, wandb_settings):
    mo.stop(
        not train_models_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to train."),
            kind="info",
        ),
    )

    _training_run_urls = train_mnist_experiments(wandb_settings)
    _training_links = "\n".join(
        f"- [Training run {index}]({url})"
        for index, url in enumerate(_training_run_urls, start=1)
    )
    mo.callout(
        mo.md(f"Training finished:\n\n{_training_links}"),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You have now trained your first model using W&B. Click on one of the links above to see your metrics and see your saved model checkpoints in the Artifacts tab in the W&B App UI
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## (Optional) Set up a W&B Alert

    Create a [W&B Alerts](https://docs.wandb.ai/guides/track/alert) to send alerts to your Slack or email from your Python code.

    There are 2 steps to follow the first time you'd like to send a Slack or email alert, triggered from your code:

    1) Turn on Alerts in your W&B [User Settings](https://wandb.ai/settings)
    2) Add `wandb.alert()` to your code. For example:

    ```python
    wandb.alert(
        title="Low accuracy",
        text=f"Accuracy is below the acceptable threshold"
    )
    ```

    The following cell shows a minimal example below to see how to use `wandb.alert`
    """)
    return


@app.function
def create_low_accuracy_alert(wandb_settings):
    """Log synthetic accuracy values and alert when one is too low."""
    if wandb.run is not None:
        wandb.finish()

    # Start a W&B run. The context manager marks it finished on exit.
    with wandb.init(
        project="pytorch-intro",
        entity=wandb_settings["entity"],
    ) as alert_run:
        # Simulate a model training loop
        accuracy_threshold = 0.3
        for training_step in range(1000):
            # Generate a random number for accuracy
            accuracy = round(random.random() + random.random(), 3)
            print(f"Accuracy is: {accuracy}, threshold: {accuracy_threshold}")

            # Log accuracy to W&B
            alert_run.log({"Accuracy": accuracy})

            # If accuracy is below the threshold, fire a W&B Alert and stop.
            if accuracy <= accuracy_threshold:
                # Send the W&B Alert
                alert_run.alert(
                    title="Low Accuracy",
                    text=(
                        f"Accuracy {accuracy} at step {training_step} is below "
                        f"the acceptable threshold, {accuracy_threshold}"
                    ),
                )
                print("Alert triggered")
                return alert_run.url, training_step, accuracy

        return alert_run.url, None, None


@app.cell(hide_code=True)
def _(create_low_accuracy_alert, wandb_settings):
    send_alert_button = mo.ui.run_button(
        label="Create a run and send a low-accuracy alert"
    )
    _entity_note = (
        f" for entity `{wandb_settings['entity']}`"
        if wandb_settings["entity"]
        else " using your default entity"
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"This runs `{create_low_accuracy_alert.__name__}` to create "
                    f"a run in `pytorch-intro`{_entity_note}. If W&B Alerts are "
                    "enabled, it may also send a real Slack or email notification."
                ),
                kind="warn",
            ),
            send_alert_button,
        ]
    )
    return (send_alert_button,)


@app.cell(hide_code=True)
def _(send_alert_button, wandb_settings):
    mo.stop(
        not send_alert_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to test an alert."),
            kind="info",
        ),
    )

    _alert_run_url, _alert_step, _alert_accuracy = create_low_accuracy_alert(
        wandb_settings
    )
    if _alert_step is None:
        _alert_result = mo.callout(
            mo.md(
                "The simulation completed without crossing the threshold. "
                f"[Open the run in W&B]({_alert_run_url})."
            ),
            kind="warn",
        )
    else:
        _alert_result = mo.callout(
            mo.md(
                f"Alert triggered at step {_alert_step} with accuracy "
                f"{_alert_accuracy:.3f}. [Open the run in W&B]({_alert_run_url})."
            ),
            kind="success",
        )
    _alert_result
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    You can find the full docs for [W&B Alerts here](https://docs.wandb.ai/guides/track/alert).

    ## Next steps
    The next tutorial you will learn how to do hyperparameter optimization using W&B Sweeps:
    [Hyperparameters sweeps using PyTorch](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/organizing-hyperparameter-sweeps-in-pytorch-with-w-b/organizing_hyperparameter_sweeps_in_pytorch_with_w_b.py/server)
    """)
    return


if __name__ == "__main__":
    app.run()
