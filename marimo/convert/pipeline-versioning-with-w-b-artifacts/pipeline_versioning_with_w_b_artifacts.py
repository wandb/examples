# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "torch>=2.6",
#     "torchvision>=0.21",
#     "wandb>=0.19.10",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Pipeline Versioning with Weights & Biases Artifacts",
)


@app.cell
def _():
    import os
    import tempfile

    import marimo as mo
    import torch
    import torch.nn.functional as F
    import torchvision
    import wandb
    from torch.utils.data import TensorDataset

    return F, TensorDataset, mo, os, tempfile, torch, torchvision, wandb


@app.cell(hide_code=True)
def _(mo):
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

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/pipeline-versioning-with-w-b-artifacts/pipeline_versioning_with_w_b_artifacts.py/server)

    # Pipeline Versioning with Weights & Biases Artifacts

    Track an MNIST pipeline from raw data to an evaluated model. This Weights & Biases
    lesson is based on the PyTorch [Basic MNIST Example](https://github.com/pytorch/examples/tree/master/mnist/).
    Watch the [original video tutorial](https://youtu.be/Hd94gatGMic).

    An **Artifact** is a versioned collection of files, such as a dataset or model.
    A run records one step of a pipeline: `log_artifact()` declares an output, and
    `use_artifact()` declares an input. Together these relationships form a directed
    acyclic graph showing which data and model versions produced a result.

    We'll create five runs: load data, preprocess data, initialize a model, train,
    and evaluate. The initialized model is saved separately so later experiments
    can start from the same weights.

    > Running the pipeline downloads MNIST, trains a small CNN for five epochs, and
    uploads datasets, checkpoints, metrics, and example images.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `artifacts-example`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="artifacts-example", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(mo, wandb, wandb_setup_form):
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


@app.cell
def _(torch):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Log the raw dataset

    First define the data-loading logic. MNIST provides 60,000 training images and
    10,000 test images. We reserve the last 10,000 training images for validation,
    leaving the official test set for final evaluation. Each split contains image
    and label tensors.
    """)
    return


@app.cell
def _(TensorDataset, torchvision):
    def load_mnist(data_root, train_size=50_000):
        # Skip the slow original mirror, keeping the fallback mirrors.
        torchvision.datasets.MNIST.mirrors = [
            mirror for mirror in torchvision.datasets.MNIST.mirrors
            if not mirror.startswith("http://yann.lecun.com")
        ]
        train_data = torchvision.datasets.MNIST(data_root, train=True, download=True)
        test_data = torchvision.datasets.MNIST(data_root, train=False, download=True)
        return (
            TensorDataset(train_data.data[:train_size], train_data.targets[:train_size]),
            TensorDataset(train_data.data[train_size:], train_data.targets[train_size:]),
            TensorDataset(test_data.data, test_data.targets),
        )

    return (load_mnist,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create `loading_run` and an Artifact named `mnist-raw` with type `dataset`.
    Names identify collections of versions; types group related Artifacts. A run's
    `job_type` labels its role in the pipeline. Keeping related runs in one project
    makes their lineage easier to explore.

    `new_file()` both creates a file inside the Artifact and attaches it. We save
    one `.pt` file per split and record the source and split sizes as JSON metadata.
    After `log_artifact()`, `wait()` ensures the upload has completed before the
    next stage fetches it. We pass its fully qualified, versioned name downstream,
    rather than the mutable `latest` alias.

    Each stage uses a temporary directory for downloaded data and Weights & Biases runtime
    files. The context managers finish the run before cleaning up the directory;
    Weights & Biases's separate Artifact cache can persist across stages.
    """)
    return


@app.cell
def _(entity, load_mnist, project, tempfile, torch, wandb):
    with tempfile.TemporaryDirectory(prefix="wandb-mnist-load-") as loading_workspace:
        with wandb.init(
            project=project, entity=entity, job_type="load-data",
            name="mnist-load-data", dir=loading_workspace,
        ) as loading_run:
            raw_datasets = load_mnist(loading_workspace)
            raw_data = wandb.Artifact(
                "mnist-raw", type="dataset",
                description="Raw MNIST dataset, split into train/validation/test",
                metadata={
                    "source": "torchvision.datasets.MNIST",
                    "sizes": [len(dataset) for dataset in raw_datasets],
                },
            )
            for _split, _dataset in zip(["training", "validation", "test"], raw_datasets):
                with raw_data.new_file(_split + ".pt", mode="wb") as _file:
                    torch.save(_dataset.tensors, _file)

            uploaded_raw_data = loading_run.log_artifact(raw_data)
            uploaded_raw_data.wait()
            raw_artifact_path = uploaded_raw_data.qualified_name
    print(f"Raw dataset: {raw_artifact_path}")
    return (raw_artifact_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Consume the raw Artifact and log preprocessed data

    The next helpers handle tensor files and preprocessing. `read_dataset()` loads
    only tensors using `weights_only=True`. `preprocess()` converts pixel values to
    floating point in `[0, 1]` and adds a channel dimension, changing image shapes
    from `(28, 28)` to `(1, 28, 28)` for the CNN.
    """)
    return


@app.cell
def _(TensorDataset, os, torch):
    def read_dataset(data_dir, split):
        images, labels = torch.load(
            os.path.join(data_dir, split + ".pt"), weights_only=True,
        )
        return TensorDataset(images, labels)

    def preprocess(dataset, normalize=True, expand_dims=True):
        images, labels = dataset.tensors
        if normalize:
            images = images.to(torch.float32) / 255
        if expand_dims:
            images = images.unsqueeze(1)
        return TensorDataset(images, labels)

    return preprocess, read_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `preprocessing_run.use_artifact()` records the raw dataset as this run's input.
    `download()` materializes its files under the supplied directory, reusing Weights & Biases's
    content-addressed cache when possible. Download time depends on the cache and
    network.

    We save the transformed splits in `mnist-preprocess` and record the preprocessing
    options in its metadata. These settings are also the CNN's input contract:
    keep normalization and channel expansion enabled for the model below.

    Artifact references look like `entity/project/mnist-raw:v0`. You can also request
    `mnist-raw:latest` or a custom alias such as `best`, but aliases can move to a new
    version. This pipeline uses the exact version produced by the preceding cell.
    """)
    return


@app.cell
def _(
    entity,
    os,
    preprocess,
    project,
    raw_artifact_path,
    read_dataset,
    tempfile,
    torch,
    wandb,
):
    preprocessing_steps = {"normalize": True, "expand_dims": True}
    with tempfile.TemporaryDirectory(prefix="wandb-mnist-preprocess-") as preprocessing_workspace:
        with wandb.init(
            project=project, entity=entity, job_type="preprocess-data",
            name="mnist-preprocess-data", dir=preprocessing_workspace,
        ) as preprocessing_run:
            raw_input = preprocessing_run.use_artifact(raw_artifact_path, type="dataset")
            raw_directory = raw_input.download(root=os.path.join(preprocessing_workspace, "raw"))
            processed_data = wandb.Artifact(
                "mnist-preprocess", type="dataset",
                description="Normalized MNIST images with a channel dimension",
                metadata=preprocessing_steps,
            )
            for _split in ["training", "validation", "test"]:
                _dataset = preprocess(read_dataset(raw_directory, _split), **preprocessing_steps)
                with processed_data.new_file(_split + ".pt", mode="wb") as _file:
                    torch.save(_dataset.tensors, _file)

            uploaded_processed_data = preprocessing_run.log_artifact(processed_data)
            uploaded_processed_data.wait()
            processed_artifact_path = uploaded_processed_data.qualified_name
    print(f"Preprocessed dataset: {processed_artifact_path}")
    return (processed_artifact_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Log an initialized model

    This small ConvNet has two convolution/pooling blocks, dropout, and a linear
    classifier. Its constructor calculates the flattened feature size from the
    input shape and layer settings. Saving the architecture settings alongside the
    weights lets another run reconstruct the model.
    """)
    return


@app.cell
def _():
    from math import floor

    import torch.nn as nn

    class ConvNet(nn.Module):
        def __init__(
            self, hidden_layer_sizes=(32, 64), kernel_sizes=(3,),
            activation="ReLU", pool_sizes=(2,), dropout=0.5,
            num_classes=10, input_shape=(1, 28, 28),
        ):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(input_shape[0], hidden_layer_sizes[0], kernel_sizes[0]),
                getattr(nn, activation)(),
                nn.MaxPool2d(pool_sizes[0]),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(hidden_layer_sizes[0], hidden_layer_sizes[-1], kernel_sizes[-1]),
                getattr(nn, activation)(),
                nn.MaxPool2d(pool_sizes[-1]),
            )
            self.layer3 = nn.Sequential(nn.Flatten(), nn.Dropout(dropout))
            feature_width = floor((input_shape[1] - kernel_sizes[0] + 1) / pool_sizes[0])
            feature_width = floor((feature_width - kernel_sizes[-1] + 1) / pool_sizes[-1])
            self.fc = nn.Linear(feature_width * feature_width * hidden_layer_sizes[-1], num_classes)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            return self.fc(x)

    return (ConvNet,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Choose the model settings below. `wandb.init(config=...)` records them with
    `initialization_run`; we copy that configuration into the model Artifact's
    metadata for later reconstruction.

    Here we demonstrate `add_file()`: save the checkpoint with `torch.save()` first,
    then attach it to `convnet`. In contrast, `new_file()` above writes directly into
    the Artifact. The initialized model is independent of the dataset preprocessing
    stage; the training stage will consume both outputs.
    """)
    return


@app.cell
def _():
    model_config = {
        "hidden_layer_sizes": [32, 64],
        "kernel_sizes": [3],
        "activation": "ReLU",
        "pool_sizes": [2],
        "dropout": 0.5,
        "num_classes": 10,
    }
    return (model_config,)


@app.cell
def _(ConvNet, entity, model_config, os, project, tempfile, torch, wandb):
    with tempfile.TemporaryDirectory(prefix="wandb-mnist-initialize-") as initialization_workspace:
        with wandb.init(
            project=project, entity=entity, job_type="initialize",
            name="mnist-initialize-model", config=model_config, dir=initialization_workspace,
        ) as initialization_run:
            initial_model = ConvNet(**initialization_run.config)
            initialized_model_artifact = wandb.Artifact(
                "convnet", type="model", description="Initialized MNIST ConvNet",
                metadata=dict(initialization_run.config),
            )
            initialized_checkpoint = os.path.join(initialization_workspace, "initialized_model.pth")
            torch.save(initial_model.state_dict(), initialized_checkpoint)
            initialized_model_artifact.add_file(initialized_checkpoint, name="initialized_model.pth")

            uploaded_initial_model = initialization_run.log_artifact(initialized_model_artifact)
            uploaded_initial_model.wait()
            initialized_artifact_path = uploaded_initial_model.qualified_name
    print(f"Initialized model: {initialized_artifact_path}")
    return (initialized_artifact_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Train using dataset and model Artifacts

    Before training, define an evaluation helper that computes mean cross-entropy
    loss and accuracy across a loader without tracking gradients. Training will use
    it on validation data after each epoch; final evaluation will use the same
    logic on the held-out test set.
    """)
    return


@app.cell
def _(F, device, torch):
    def evaluate_loader(model, loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
        return total_loss / len(loader.dataset), 100.0 * correct / len(loader.dataset)

    return (evaluate_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The training cell declares **two Artifact inputs**: the preprocessed dataset and
    the initialized model. It loads weights and reconstructs the architecture from
    model metadata, then trains using the editable settings below.

    `training_run.log()` records training loss and validation loss/accuracy, with the
    epoch and number of examples seen. Weights & Biases advances its default step on each log
    call. `model.train()` at the start of each epoch restores dropout after validation
    switches the model to evaluation mode.

    Finally, save a new `trained-model` Artifact. Its metadata records the architecture
    and training settings, while the run's input edges identify the exact dataset
    and initialization used. The upload must finish before the report run fetches it.
    For more training details, see the
    [PyTorch integration lesson](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/simple-pytorch-integration/simple_pytorch_integration.py/server).
    """)
    return


@app.cell
def _():
    train_config = {
        "batch_size": 128,
        "epochs": 5,
        "batch_log_interval": 25,
        "optimizer": "Adam",
    }
    return (train_config,)


@app.cell
def _(
    ConvNet,
    F,
    device,
    entity,
    evaluate_loader,
    initialized_artifact_path,
    os,
    processed_artifact_path,
    project,
    read_dataset,
    tempfile,
    torch,
    train_config,
    wandb,
):
    with tempfile.TemporaryDirectory(prefix="wandb-mnist-train-") as training_workspace:
        with wandb.init(
            project=project, entity=entity, job_type="train",
            name="mnist-train-convnet", config=train_config, dir=training_workspace,
        ) as training_run:
            training_data_artifact = training_run.use_artifact(processed_artifact_path, type="dataset")
            training_data_directory = training_data_artifact.download(
                root=os.path.join(training_workspace, "data")
            )
            training_dataset = read_dataset(training_data_directory, "training")
            validation_dataset = read_dataset(training_data_directory, "validation")
            train_loader = torch.utils.data.DataLoader(
                training_dataset, batch_size=training_run.config.batch_size,
            )
            validation_loader = torch.utils.data.DataLoader(
                validation_dataset, batch_size=training_run.config.batch_size,
            )

            initialization_input = training_run.use_artifact(initialized_artifact_path, type="model")
            initialization_directory = initialization_input.download(
                root=os.path.join(training_workspace, "initialized-model")
            )
            training_model_config = dict(initialization_input.metadata)
            training_run.config.update(training_model_config)
            training_model = ConvNet(**training_model_config).to(device)
            training_model.load_state_dict(torch.load(
                os.path.join(initialization_directory, "initialized_model.pth"),
                map_location=device, weights_only=True,
            ))

            optimizer = getattr(torch.optim, training_run.config.optimizer)(training_model.parameters())
            example_count = 0
            for epoch in range(training_run.config.epochs):
                training_model.train()
                for batch_index, (batch_images, batch_labels) in enumerate(train_loader):
                    batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                    optimizer.zero_grad()
                    training_loss = F.cross_entropy(training_model(batch_images), batch_labels)
                    training_loss.backward()
                    optimizer.step()
                    example_count += len(batch_images)
                    if batch_index % training_run.config.batch_log_interval == 0:
                        training_run.log({
                            "epoch": epoch, "examples": example_count,
                            "train/loss": training_loss.item(),
                        })

                validation_loss, validation_accuracy = evaluate_loader(training_model, validation_loader)
                training_run.log({
                    "epoch": epoch, "examples": example_count,
                    "validation/loss": validation_loss,
                    "validation/accuracy": validation_accuracy,
                })
                print(f"Epoch {epoch + 1}: validation accuracy {validation_accuracy:.2f}%")

            trained_model_artifact = wandb.Artifact(
                "trained-model", type="model", description="Trained MNIST ConvNet",
                metadata={
                    "model_config": training_model_config,
                    "training_config": train_config,
                },
            )
            trained_checkpoint = os.path.join(training_workspace, "trained_model.pth")
            torch.save(training_model.state_dict(), trained_checkpoint)
            trained_model_artifact.add_file(trained_checkpoint, name="trained_model.pth")
            uploaded_trained_model = training_run.log_artifact(trained_model_artifact)
            uploaded_trained_model.wait()
            trained_artifact_path = uploaded_trained_model.qualified_name
    print(f"Trained model: {trained_artifact_path}")
    return (trained_artifact_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Evaluate the trained model and inspect hard examples

    The final run consumes `trained-model` and the same preprocessed dataset version.
    It evaluates only the test split, keeping test results separate from training and
    hyperparameter selection.

    The next helper ranks individual test images by cross-entropy loss and returns
    the 32 highest-loss examples (or all examples for a smaller test set). This helps
    diagnose confident mistakes or ambiguous images.
    """)
    return


@app.cell
def _(F, device, torch):
    def get_hardest_k_examples(model, testing_set, k=32):
        model.eval()
        loader = torch.utils.data.DataLoader(testing_set, 1, shuffle=False)
        losses = []
        predictions = []
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                losses.append(F.cross_entropy(logits, labels).detach().cpu())
                predictions.append(logits.argmax(dim=1).detach().cpu())
        losses = torch.stack(losses)
        predictions = torch.cat(predictions)
        hardest_indices = torch.argsort(losses)[-k:]
        return (
            losses[hardest_indices],
            testing_set.tensors[0][hardest_indices],
            testing_set.tensors[1][hardest_indices],
            predictions[hardest_indices],
        )

    return (get_hardest_k_examples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `evaluation_run.summary.update()` stores the final test loss and accuracy.
    `evaluation_run.log()` records high-loss images with their predicted label, true
    label, and loss in the captions. These Media panels complement aggregate metrics
    with concrete examples of the model's behavior.

    The trained Artifact stores architecture and training settings separately in
    its metadata. We use `model_config` to reconstruct the CNN; the training
    settings remain available for inspection.
    """)
    return


@app.cell
def _(
    ConvNet,
    device,
    entity,
    evaluate_loader,
    get_hardest_k_examples,
    os,
    processed_artifact_path,
    project,
    read_dataset,
    tempfile,
    torch,
    trained_artifact_path,
    wandb,
):
    with tempfile.TemporaryDirectory(prefix="wandb-mnist-evaluate-") as evaluation_workspace:
        with wandb.init(
            project=project, entity=entity, job_type="report",
            name="mnist-evaluate-convnet", dir=evaluation_workspace,
        ) as evaluation_run:
            evaluation_data_artifact = evaluation_run.use_artifact(processed_artifact_path, type="dataset")
            evaluation_directory = evaluation_data_artifact.download(
                root=os.path.join(evaluation_workspace, "data")
            )
            testing_set = read_dataset(evaluation_directory, "test")
            test_loader = torch.utils.data.DataLoader(testing_set, batch_size=128, shuffle=False)

            evaluation_model_artifact = evaluation_run.use_artifact(trained_artifact_path, type="model")
            evaluation_model_directory = evaluation_model_artifact.download(
                root=os.path.join(evaluation_workspace, "trained-model")
            )
            evaluation_model_config = evaluation_model_artifact.metadata["model_config"]
            evaluation_model = ConvNet(**evaluation_model_config).to(device)
            evaluation_model.load_state_dict(torch.load(
                os.path.join(evaluation_model_directory, "trained_model.pth"),
                map_location=device, weights_only=True,
            ))
            test_loss, test_accuracy = evaluate_loader(evaluation_model, test_loader)
            highest_losses, hardest_images, true_labels, predicted_labels = get_hardest_k_examples(
                evaluation_model, testing_set
            )
            evaluation_run.summary.update({"loss": test_loss, "accuracy": test_accuracy})
            evaluation_run.log({
                "high-loss-examples": [
                    wandb.Image(
                        image.cpu().clamp(0, 1).mul(255).round().to(torch.uint8),
                        caption=f"predicted={int(prediction)}, true={int(label)}, loss={float(loss):.3f}",
                    )
                    for image, prediction, label, loss in zip(
                        hardest_images, predicted_labels, true_labels, highest_losses,
                    )
                ]
            })
    print(f"Test loss: {test_loss:.4f}; test accuracy: {test_accuracy:.2f}%")
    print(f"Evaluation run: {evaluation_run.url}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Inspect the pipeline in Weights & Biases

    Open the run links printed by Weights & Biases and inspect:

    - **Artifacts:** `mnist-raw`, `mnist-preprocess`, `convnet`, and `trained-model`.
    - **Lineage graph:** loading produces raw data; preprocessing consumes it and
      produces processed data; training consumes processed data and the initialized
      model, then produces the trained model; evaluation consumes processed data
      and the trained model.
    - **Metadata:** dataset sizes, preprocessing settings, and model configuration.
    - **Training charts:** `train/loss`, `validation/loss`, and `validation/accuracy`.
    - **Evaluation:** final `loss` and `accuracy`, plus the `high-loss-examples` panel.

    The graph can group nodes by Artifact `type` and run `job_type`, or show individual
    runs and Artifact versions. This lets you trace a result back to its exact inputs.

    Edit the model or training settings and rerun the affected stages to create and
    compare versions. Weights & Biases deduplicates unchanged file contents. For a controlled
    training comparison, keep the initialized model and dataset versions fixed.
    """)
    return


if __name__ == "__main__":
    app.run()
