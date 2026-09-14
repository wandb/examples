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
# ///

import marimo

__generated_with = "0.24.2"
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
    # W&B + PyTorch

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/simple-pytorch-integration/simple_pytorch_integration.py/server)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Use [Weights & Biases](https://wandb.com) for machine learning experiment tracking, dataset versioning, and project collaboration.

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## What this notebook covers:

    We show you how to integrate Weights & Biases with your PyTorch code to add experiment tracking to your pipeline.

    ### The resulting interactive W&B dashboard will look like:
    ![Example PyTorch training dashboard](https://i.imgur.com/z8TK2Et.png)

    ### In pseudocode, what we'll do is:
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
    Follow along with a [video tutorial](https://wandb.me/pytorch-video)!
    **Note**: Sections starting with _Step_ are all you need to integrate W&B in an existing pipeline. The rest just loads data and defines a model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Install, Import, and Log In

    Open this notebook in molab, or run `uvx marimo edit simple_pytorch_integration.py --sandbox` locally. Its script metadata installs the required packages.

    The example trains a small convolutional network on every fifth MNIST image: 12,000 training examples and 2,000 test examples. A CUDA GPU is optional; CPU training is supported. MNIST downloads into `data/` only after you submit **Train model and log to W&B**. That submission creates one W&B run and, by default, saves its ONNX model.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.callout(mo.md(f"**Training device:** `{device}`."), kind="info")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    ### Step 1: Import W&B and Login

    To log data, [create a W&B account](https://wandb.ai) and get your [API key](https://wandb.ai/authorize). Enter the key below, or leave it blank to use `WANDB_API_KEY` from marimo's **Secrets** panel or credentials already configured in this runtime with `wandb login`. A fresh molab runtime does not inherit your computer's login.

    Set **W&B entity or team** to a team you can write to. You can find the entity in a project URL, `wandb.ai/<entity>/<project>`. Leave it blank to use your default entity. Connecting authenticates this session; training has a separate submission below.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(
            api_key=mo.ui.text(
                kind="password", label="W&B API key (optional)", full_width=True
            ),
            entity=mo.ui.text(label="W&B entity or team (optional)", full_width=True),
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Connect to W&B above before running the training example."),
            kind="info",
        ),
    )
    _api_key = wandb_login_form.value["api_key"].strip()
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False
    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check your API key or this runtime's credentials and connection, then submit again."
            ),
            kind="danger",
        ),
    )
    wandb_settings = {"entity": wandb_login_form.value["entity"].strip() or None}
    mo.callout(
        mo.md("Connected to W&B. Configure and submit training below."), kind="success"
    )
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define the Experiment and Pipeline
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Step 2: Track metadata and hyperparameters with `wandb.init`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Programmatically, the first thing we do is define our experiment:
    what are the hyperparameters? what metadata is associated with this run?

    It's a pretty common workflow to store this information in a `config` dictionary
    (or similar object)
    and then access it as needed.

    For this example, we're only letting a few hyperparameters vary
    and hand-coding the rest.
    But any part of your model can be part of the `config`!

    We also include some metadata: we're using the MNIST dataset and a convolutional
    architecture. If we later work with, say,
    fully-connected architectures on CIFAR in the same project,
    this will help us separate our runs.
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
    Now, let's define the overall pipeline,
    which is pretty typical for model-training:

    1. we first `make` a model, plus associated data and optimizer, then
    2. we `train` the model accordingly and finally
    3. `test` it to see how training went.

    We'll implement these functions below.
    """)
    return


@app.function
def model_pipeline(hyperparameters, project, entity=None, name=None, export_onnx=True):
    seed_everything(hyperparameters["seed"])
    if wandb.run is not None:
        wandb.run.finish()

    # tell wandb to get started
    with wandb.init(
        project=project, entity=entity, name=name, config=hyperparameters
    ) as run:
        # access all HPs through run.config, so logging matches execution!
        config = run.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer = make(config)
        print(model)

        # and use them to train the model
        train(model, train_loader, criterion, optimizer, config, run)

        # and test its final performance
        test_accuracy = test(model, test_loader, run, export_onnx=export_onnx)
        result = {
            "url": run.url,
            "name": run.name,
            "test_accuracy": test_accuracy,
            "export_onnx": export_onnx,
        }

    return model, result


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The only difference here from a standard pipeline
    is that it all occurs inside the context of `wandb.init`.
    Calling this function sets up a line of communication
    between your code and our servers.

    Passing the `config` dictionary to `wandb.init`
    immediately logs all that information to us,
    so you'll always know what hyperparameter values
    you set your experiment to use.

    To ensure the values you chose and logged are always the ones that get used
    in your model, we recommend using the `run.config` copy of your object.
    Check the definition of `make` below to see some examples.

    > *Side Note*: We take care to run our code in separate processes,
    so that any issues on our end
    (e.g. a giant sea monster attacks our data centers)
    don't crash your code.
    Once the issue is resolved (e.g. the Kraken returns to the deep)
    you can log the data with `wandb sync`.
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
    This part is very important, but it is no different from what it would be without W&B.

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
    Defining the model is normally the fun part!

    But nothing changes with `wandb`,
    so we're gonna stick with a standard ConvNet architecture.

    Don't be afraid to mess around with this and try some experiments --
    all your results will be logged on [wandb.ai](https://wandb.ai)!
    """)
    return


@app.class_definition
# Conventional and convolutional neural network

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


@app.cell
def _(config):
    model = ConvNet(config["kernels"], config["classes"])
    model
    return (model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define Training Logic
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Moving on in our `model_pipeline`, it's time to specify how we `train`.

    Two `wandb` functions come into play here: `watch` and `log`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Step 3. Track gradients with `run.watch` and everything else with `run.log`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    `run.watch` will log the gradients and the parameters of your model,
    every `log_freq` steps of training.

    All you need to do is call it before you start training.

    The rest of the training code remains the same:
    we iterate over epochs and batches,
    running forward and backward passes
    and applying our `optimizer`.
    """)
    return


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
                train_log(loss, example_ct, epoch, run)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The only difference is in the logging code:
    where previously you might have reported metrics by printing to the terminal,
    now you pass the same information to `run.log`.

    `run.log` expects a dictionary with strings as keys.
    These strings identify the objects being logged, which make up the values.
    You can also optionally log which `step` of training you're on.

    > *Side Note*: I like to use the number of examples the model has seen,
    since this makes for easier comparison across batch sizes,
    but you can use raw steps or batch count. For longer training runs, it can also make sense to log by `epoch`.
    """)
    return


@app.function
def train_log(loss, example_ct, epoch, run):
    # Where the magic happens
    run.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Define Testing Logic
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Once the model is done training, we want to test it:
    run it against some fresh data from production, perhaps,
    or apply it to some hand-curated "hard examples".
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Optional Step 4: Call `run.save`

    This is also a great time to save the model's architecture
    and final parameters to disk.
    For maximum compatibility, we'll `export` our model in the
    [Open Neural Network eXchange (ONNX) format](https://onnx.ai/).

    Passing that filename to `run.save` ensures that the model parameters
    are saved to W&B's servers: no more losing track of which `.h5` or `.pb`
    corresponds to which training runs!

    For more advanced `wandb` features for storing, versioning, and distributing
    models, check out our [Artifacts tools](https://www.wandb.com/artifacts).
    """)
    return


@app.function
def test(model, test_loader, run, export_onnx=True):
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

    if export_onnx:
        # Save the model in the exchangeable ONNX format
        model_path = Path(run.dir) / "model.onnx"
        # Use the classic exporter, which needs only the onnx dependency.
        torch.onnx.export(model, images[:1], str(model_path), dynamo=False)
        run.save(str(model_path), base_path=run.dir, policy="now")

    return test_accuracy


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run training and watch your metrics live on wandb.ai!
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Now that we've defined the whole pipeline and slipped in
    those few lines of W&B code,
    we're ready to run our fully-tracked experiment.

    We'll report a few links to you:
    our documentation,
    the Project page, which organizes all the runs in a project, and
    the Run page, where this run's results will be stored.

    Navigate to the Run page and check out these tabs:

    1. **Charts**, where the model gradients, parameter values, and loss are logged throughout training
    2. **System**, which contains a variety of system metrics, including Disk I/O utilization, CPU and GPU metrics, and more
    3. **Logs**, which has a copy of anything pushed to standard out during training
    4. **Files**, where, if model saving is enabled and training is complete, you can click on the `model.onnx` to view our network with the [Netron model viewer](https://github.com/lutzroeder/netron).

    Once the run is finished
    (i.e. the `with wandb.init` block is exited),
    we'll also print a summary of the results in the cell output.
    """)
    return


@app.cell(hide_code=True)
def _(config, wandb_login_form):
    _connection_submission = wandb_login_form.value
    _entity = (
        _connection_submission["entity"].strip()
        if _connection_submission is not None
        else ""
    )
    _entity_note = (
        f"Runs will use **{_entity or 'your default entity'}** after authentication."
    )
    training_form = (
        mo.md(
            "{project}\n\n{run_name}\n\n{epochs}\n\n{batch_size}\n\n{learning_rate}\n\n{seed}\n\n{export_onnx}"
        )
        .batch(
            project=mo.ui.text(value="pytorch-demo", label="W&B project"),
            run_name=mo.ui.text(value="", label="Run name (blank auto-generates)"),
            epochs=mo.ui.number(
                start=1, stop=20, step=1, value=config["epochs"], label="Epochs"
            ),
            batch_size=mo.ui.dropdown(
                options=[32, 64, 128, 256],
                value=config["batch_size"],
                label="Batch size",
            ),
            learning_rate=mo.ui.number(
                start=0.0001,
                stop=0.1,
                step=0.0001,
                value=config["learning_rate"],
                label="Learning rate",
            ),
            seed=mo.ui.number(
                start=0,
                stop=2**32 - 1,
                step=1,
                value=config["seed"],
                label="Random seed",
            ),
            export_onnx=mo.ui.checkbox(value=True, label="Save model.onnx to the run"),
        )
        .form(submit_button_label="Train model and log to W&B", bordered=True)
    )
    mo.vstack([mo.md(_entity_note), training_form])
    return (training_form,)


@app.cell(hide_code=True)
def _(config, training_form, wandb_settings):
    mo.stop(
        training_form.value is None,
        mo.md(
            "Submit **Train model and log to W&B** to download MNIST and create one training run."
        ),
    )
    _values = training_form.value
    mo.stop(
        not _values["project"].strip(), mo.md("Enter a W&B project and submit again.")
    )
    training_config = dict(
        config,
        **{
            key: _values[key]
            for key in ("epochs", "batch_size", "learning_rate", "seed")
        },
    )
    training_settings = {
        "project": _values["project"].strip(),
        "entity": wandb_settings["entity"],
        "name": _values["run_name"].strip() or None,
        "export_onnx": _values["export_onnx"],
    }
    return training_config, training_settings


@app.cell
def _(training_config, training_settings):
    # Build, train and analyze the model with the pipeline
    trained_model, training_result = model_pipeline(training_config, **training_settings)
    return trained_model, training_result


@app.cell(hide_code=True)
def _(training_result):
    _run_link = (
        f"[Open run {training_result['name']}]({training_result['url']})"
        if training_result["url"]
        else "The run completed offline; results are saved in the local wandb directory."
    )
    mo.callout(
        mo.md(
            f"**Test accuracy:** {training_result['test_accuracy']:.2%}. {_run_link}"
        ),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Test Hyperparameters with Sweeps

    We only looked at a single set of hyperparameters in this example.
    But an important part of most ML workflows is iterating over
    a number of hyperparameters.

    You can use Weights & Biases Sweeps to automate hyperparameter testing and explore the space of possible models and optimization strategies.

    ### [Check out Hyperparameter Optimization in PyTorch using W&B Sweeps $\rightarrow$](http://wandb.me/sweeps-colab)

    Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

    1. **Define the sweep:** We do this by creating a dictionary or a [YAML file](https://docs.wandb.ai/models/sweeps/sweep-config-keys) that specifies the parameters to search through, the search strategy, the optimization metric et all.

    2. **Initialize the sweep:**
    `sweep_id = wandb.sweep(sweep_config)`

    3. **Run the sweep agent:**
    `wandb.agent(sweep_id, function=train)`

    And voila! That's all there is to running a hyperparameter sweep!
    <img src="https://imgur.com/UiQKg0L.png" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Example Gallery

    See examples of projects tracked and visualized with W&B in our [Gallery →](https://app.wandb.ai/gallery)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Advanced Setup

    1. [Environment variables](https://docs.wandb.ai/models/track/environment-variables): Set API keys in environment variables so you can run training on a managed cluster.
    2. [Offline mode](https://docs.wandb.ai/models/track/run): Set `WANDB_MODE=offline` to train offline and sync results later.
    3. [On-prem](https://docs.wandb.ai/platform/hosting): Install W&B in a private cloud or air-gapped servers in your own infrastructure. We have local installations for everyone from academics to enterprise teams.
    4. [Sweeps](https://docs.wandb.ai/models/sweeps): Set up hyperparameter search quickly with our lightweight tool for tuning.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Helper functions

    ### Training and reproducibility helpers

    The forward pass, backward pass, and optimizer step are standard PyTorch. Resetting numeric seeds for every submitted experiment makes comparisons repeatable on the same device and software stack.
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
def seed_everything(seed):
    # Ensure deterministic behavior
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    app.run()
