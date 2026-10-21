# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.23",
#     "numpy>=1.26",
#     "onnx>=1.16",
#     "torch>=2.5",
#     "torchvision>=0.20",
#     "tqdm>=4.66",
#     "wandb>=0.18",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Simple PyTorch Integration")

with app.setup:
    import random
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import wandb
    from tqdm.auto import tqdm

    # Device configuration
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")


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

    # Weights & Biases + PyTorch

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/simple-pytorch-integration/simple_pytorch_integration.py/server)

    In this notebook you will learn how to integrate Weights & Biases with your PyTorch code to add experiment tracking to your pipeline.

    The example trains a small convolutional network on every fifth MNIST image:
    12,000 training examples and 2,000 test examples. A GPU is optional. Running
    the experiment downloads MNIST into `data/`, trains for five epochs, and
    creates one Weights & Biases run with metrics and an ONNX model.

    In pseudocode, what we'll do is:

    ```python
    # import the library
    import wandb

    # capture a dictionary of hyperparameters with config
    config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 128}

    # start a new experiment
    with wandb.init(project="new-sota-model", config=config) as run:
        # set up model and data
        model, dataloader = get_model(), get_data()

        # optional: track gradients
        run.watch(model)

        for batch in dataloader:
            metrics = model.training_step(batch)
            # log metrics inside your training loop to visualize model performance
            run.log(metrics)

        # optional: save model at the end
        model.to_onnx()
        run.save("model.onnx")
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `pytorch_demo`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="pytorch_demo", label="Project", full_width=True),
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
    ## Define the experiment settings

    The `config` dictionary holds the settings the training code will use, including
    the learning rate, batch size, and epoch count. It also records the dataset
    and architecture so you can distinguish this experiment from other runs.

    Pass this dictionary to `wandb.init(config=config)`. Inside the run, read
    settings from `run.config` so the logged values and training inputs agree.
    """)
    return


@app.cell
def _():
    config = dict(
        epochs=5,
        classes=10,
        kernels=[16, 32],
        batch_size=128,
        learning_rate=0.005,
        dataset="MNIST",
        architecture="CNN",
        seed=42,
    )
    return (config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The `model_pipeline` function keeps one experiment inside a Weights & Biases
    run. It calls `make` to build the model, data loaders, loss, and optimizer,
    then calls `train` and `evaluate`. Exiting the run’s context manager finishes
    logging. The helper functions are defined below.
    """)
    return


@app.function
def model_pipeline(hyperparameters, entity, project):
    seed_everything(hyperparameters["seed"])

    # tell wandb to get started
    with wandb.init(entity=entity, project=project, config=hyperparameters) as run:
        # access all HPs through run.config, so logging matches execution!
        config = run.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config, run)

        # and test its final performance
        evaluate(model, test_loader, run)

    return model


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `wandb.init()` starts the run and records the supplied configuration.
    The model, optimizer, and data loaders use `run.config` to read those settings.

    The `with` block keeps metric logging and the final model file associated
    with this run. A new call to `model_pipeline` creates a fresh model and run,
    which lets you compare experiments after changing a setting.
    """)
    return


@app.function
def make(config):
    # Make the data
    train, test = get_data(train=True), get_data(train=False)
    train_loader = make_loader(train, batch_size=config.batch_size, shuffle=True)
    test_loader = make_loader(test, batch_size=config.batch_size, shuffle=False)

    # Make the model
    model = ConvNet(config.kernels, config.classes).to(device)

    # Make the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return model, train_loader, test_loader, criterion, optimizer


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define the Data Loading and Model

    Now, we need to specify how the data is loaded and what the model looks like.
    This part is very important, but it is no different from what it would be without Weights & Biases.

    `get_data` uses every fifth MNIST image to keep this tutorial small. `make` combines the data, model, loss, and optimizer. The loaders use the notebook process so the same code works in local and hosted sessions.
    """)
    return


@app.function
def get_data(slice=5, train=True):
    # remove slow mirror from list of MNIST mirrors
    torchvision.datasets.MNIST.mirrors = [
        mirror for mirror in torchvision.datasets.MNIST.mirrors
        if not mirror.startswith("http://yann.lecun.com")
    ]
    full_dataset = torchvision.datasets.MNIST(
        root="data", train=train, transform=transforms.ToTensor(), download=True
    )
    #  equiv to slicing with [::slice]
    sub_dataset = torch.utils.data.Subset(
        full_dataset, indices=range(0, len(full_dataset), slice)
    )
    return sub_dataset


@app.function
def make_loader(dataset, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
        num_workers=0,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The ConvNet applies two convolution-and-pooling stages, then maps the flattened
    features to ten digit scores. It returns logits for cross-entropy loss.
    Change the channel sizes in `config["kernels"]` to compare model capacity
    while keeping the dataset and training budget fixed.
    """)
    return


@app.class_definition
class ConvNet(nn.Module):
    def __init__(self, kernels, classes=10):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, kernels[0], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(kernels[0], kernels[1], kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * kernels[-1], classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define the training loop

    For each batch, compute predictions and cross-entropy loss, clear the old
    gradients, call `loss.backward()`, then update the weights with `optimizer.step()`.
    The order matters because the optimizer uses the gradients from the backward pass.

    `run.watch(..., log="all", log_freq=10)` records parameter and gradient
    histograms. The loop logs loss every 25 batches and on the final batch,
    using the number of examples seen as the logging step.
    """)
    return


@app.function
def train_batch(images, labels, model, optimizer, criterion):
    images, labels = images.to(device), labels.to(device)

    # Forward pass ➡
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()
    return loss.item()


@app.function
def train(model, loader, criterion, optimizer, config, run):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    run.watch(model, criterion, log="all", log_freq=10)
    model.train()

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    for epoch in tqdm(range(config.epochs)):
        for images, labels in loader:
            loss = train_batch(images, labels, model, optimizer, criterion)
            example_ct += len(images)
            batch_ct += 1

            # Report metrics every 25th batch
            if batch_ct % 25 == 0 or batch_ct == total_batches:
                run.log({"epoch": epoch, "loss": loss}, step=example_ct)
                print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `run.log()` accepts a dictionary whose keys name the recorded metrics.
    This example logs `loss` and `epoch`, with the number of examples seen passed
    as `step`. Use a consistent step convention when comparing runs.

    ## Evaluate and save the model

    The evaluation helper switches the model to evaluation mode, predicts the
    held-out MNIST digits, and logs the fraction classified correctly as
    `test_accuracy`. It then exports the trained network to ONNX and uploads
    `model.onnx` with `run.save()`.

    If you repeatedly use this score to choose hyperparameters, that held-out
    subset serves as validation data. Use a separate test set for a final estimate
    of generalization. For versioned model collections and lineage, see the
    [Weights & Biases Artifacts documentation](https://docs.wandb.ai/models/artifacts).
    """)
    return


@app.function
def evaluate(model, test_loader, run):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Accuracy of the model on the {total} test images: {test_accuracy:%}")
    run.log({"test_accuracy": test_accuracy})

    # Save the trained model alongside this run's metrics.
    model_path = Path(run.dir) / "model.onnx"
    torch.onnx.export(model, images[:1], str(model_path), dynamo=False)
    run.save(str(model_path), base_path=run.dir, policy="now")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run training and inspect the logged results

    Run the experiment cell after submitting the setup form. It builds a fresh
    model, trains it, evaluates the held-out images, and uploads the ONNX file.

    Open the run URL printed by Weights & Biases. Inspect the loss curve and the
    parameter and gradient histograms, then check `test_accuracy` in the summary.
    Under **Files**, find `model.onnx`. The configuration records the settings
    used to create these results.
    """)
    return


@app.cell
def _(config, entity, project):
    model_pipeline(config, entity, project)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify your experiment

    Open the run link printed by Weights & Biases. Check that its **Config** matches the
    Python dictionary above, that the loss and gradient charts contain data,
    and that `test_accuracy` appears in the run summary. Under **Files**, look
    for `model.onnx`.

    Change `learning_rate` or `batch_size` in `config`, then run the experiment
    cell again. This creates a new run. Compare both runs in your selected project to see how the change affected loss and test accuracy.

    ## Next steps: Test Hyperparameters with Sweeps

    We only looked at a single set of hyperparameters in this example.
    But an important part of most ML workflows is iterating over
    a number of hyperparameters.

    You can use [Weights & Biases Sweeps](https://docs.coreweave.com/products/wandb/sweeps) to automate hyperparameter testing and explore the space of possible models and optimization strategies.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Helper functions

    Resetting the random seeds before each experiment makes runs easier to
    compare on the same device and software stack.
    """)
    return


@app.function
def seed_everything(seed):
    # Reset random number generators before each experiment.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    app.run()
