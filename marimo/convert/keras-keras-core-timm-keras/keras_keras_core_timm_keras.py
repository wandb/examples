# /// script
# dependencies = ["namex", "wandb-addons @ git+https://github.com/soumik12345/wandb-addons"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/keras_core/timm_keras.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{keras_core_timm} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://keras.io/img/logo-k-keras-wb.png" width="200" alt="Keras" />
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{keras_core_timm} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥 Fine-tune a [Timm](https://huggingface.co/docs/timm/index) Model with Keras and WandB 🦄

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/keras_core/timm_keras.ipynb)

    This notebook demonstrates
    - how we can fine-tune a pre-trained model from timm using [KerasCore](https://github.com/keras-team/keras-core).
    - how we can use the backend-agnostic Keras callbacks for [Weights & Biases](https://wandb.ai/site) to manage and track our experiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installing and Importing the Dependencies
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - We install the `main` branch of [KerasCore](https://github.com/keras-team/keras-core), this lets us use the latest feature merged in KerasCore.
    - We install [timm](https://huggingface.co/docs/timm/index), a library containing SOTA computer vision models, layers, utilities, optimizers, schedulers, data-loaders, augmentations, and training/evaluation scripts.
    - We also install [wandb-addons](https://github.com/soumik12345/wandb-addons), a library that hosts the backend-agnostic callbacks compatible with KerasCore
    """)
    return


@app.cell
def _(subprocess):
    # install the `main` branch of KerasCore
    # packages added via marimo's package management: namex !pip install -qq namex
    #! apt install python3.10-venv
    subprocess.call(['apt', 'install', 'python3.10-venv'])
    #! git clone --depth 1 https://github.com/soumik12345/keras-core.git && cd keras-core && python pip_build.py --install
    subprocess.call(['pip_build.py', '--install'])

    # install timm and wandb-addons
    # packages added via marimo's package management: git+https://github.com/soumik12345/wandb-addons !pip install -qq git+https://github.com/soumik12345/wandb-addons
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We specify the Keras backend to be using `torch` by explicitly specifying the environment variable `KERAS_BACKEND`.
    """)
    return


@app.cell
def _():
    import os
    os.environ["KERAS_BACKEND"] = "torch"

    import numpy as np
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import timm
    from timm.data import resolve_data_config

    import torchvision
    from torchvision import datasets, models, transforms
    from torchvision.transforms.functional import InterpolationMode

    import wandb
    from wandb_addons.keras import WandbMetricsLogger, WandbModelCheckpoint

    return (
        InterpolationMode,
        WandbMetricsLogger,
        WandbModelCheckpoint,
        datasets,
        np,
        os,
        plt,
        resolve_data_config,
        timm,
        torch,
        torchvision,
        transforms,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We initialize a [wandb run](https://docs.wandb.ai/guides/runs) and set the configs for the experiment.
    """)
    return


@app.cell
def _(resolve_data_config, wandb):
    wandb.init(project="keras-torch")

    config = wandb.config
    config.model_name = "xception41"
    config.freeze_backbone = False
    config.preprocess_config = resolve_data_config({}, model=config.model_name)
    config.dropout_rate = 0.5
    config.batch_size = 4
    config.num_epochs = 25
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A PyTorch-based Input Pipeline
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We will be using the [ImageNette](https://github.com/fastai/imagenette) dataset for this experiment. Imagenette is a subset of 10 easily classified classes from [Imagenet](https://www.image-net.org/) (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).

    First, let's download this dataset.
    """)
    return


@app.cell
def _(subprocess):
    #! wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz -P imagenette
    subprocess.call(['wget', 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz', '-P', 'imagenette'])
    #! tar zxf imagenette/imagenette2-320.tgz -C imagenette
    subprocess.call(['tar', 'zxf', 'imagenette/imagenette2-320.tgz', '-C', 'imagenette'])
    #! gzip -d imagenette/imagenette2-320.tgz
    subprocess.call(['gzip', '-d', 'imagenette/imagenette2-320.tgz'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we create our standard torch-based data loading pipeline.
    """)
    return


@app.cell
def _(InterpolationMode, config, datasets, os, torch, transforms):
    # Define pre-processing and augmentation transforms for the train and validation sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(
                size=config.preprocess_config["input_size"][1],
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                config.preprocess_config["mean"],
                config.preprocess_config["std"]
            )
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.preprocess_config["input_size"][1]),
            transforms.ToTensor(),
            transforms.Normalize(
                config.preprocess_config["mean"],
                config.preprocess_config["std"]
            )
        ]),
    }

    # Define the train and validation datasets
    data_dir = 'imagenette/imagenette2-320'
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]
        )
        for x in ['train', 'val']
    }

    # Define the torch dataloaders corresponding to the train and validation dataset
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x],
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Specify the global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return class_names, dataloaders, device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's take a look at a few of the samples.
    """)
    return


@app.cell
def _(class_names, config, dataloaders, np, plt, torchvision):
    def imshow(inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array(config.preprocess_config["mean"])
        std = np.array(config.preprocess_config["std"])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)


    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))
    print(inputs.shape, classes.shape)

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating and Training our Classifier
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We typically define a model in PyTorch using [`torch.nn.Module`s](https://pytorch.org/docs/stable/notes/modules.html) which act as the building blocks of stateful computation. Even though Keras supports PyTorch as a backend, it does not mean that we can nest torch modules inside a [`keras_core.Model`](https://keras.io/keras_core/api/models/), because trainable variables inside a Keras Model is tracked exclusively via [Keras Layers](https://keras.io/keras_core/api/layers/).

    KerasCore provides us with a feature called `TorchModuleWrapper` which enables us to do exactly this. The `TorchModuleWrapper` is a Keras Layer that accepts a torch module and tracks its trainable variables, essentially converting the torch module into a Keras Layer. This enables us to put any torch modules inside a Keras Model and train them with a single `model.fit()`!

    The idea of the `TorchModuleWrapper` was proposed by Keras' creator [François Chollet](https://github.com/fchollet) on [this issue thread](https://github.com/keras-team/keras-core/issues/604).
    """)
    return


@app.cell
def _(timm):
    import keras_core as keras
    from keras_core.utils import TorchModuleWrapper


    class TimmClassifier(keras.Model):

        def __init__(self, model_name, freeze_backbone, dropout_rate, num_classes, *args, **kwargs):
            super().__init__(*args, **kwargs)
        
            # Define the pre-trained module from timm
            self.backbone = TorchModuleWrapper(
                timm.create_model(model_name, pretrained=True)
            )
            self.backbone.trainable = not freeze_backbone
        
            # Build the classification head using keras layers
            self.global_average_pooling = keras.layers.GlobalAveragePooling2D()
            self.dropout = keras.layers.Dropout(dropout_rate)
            self.classification_head = keras.layers.Dense(num_classes)

        def call(self, inputs):
            # We get the unpooled features from the timm backbone by calling `forward_features`
            # on the torch module corresponding to the backbone.
            x = self.backbone.module.forward_features(inputs)
            x = self.global_average_pooling(x)
            x = self.dropout(x)
            x = self.classification_head(x)
            return keras.activations.softmax(x, axis=1)

    return TimmClassifier, keras


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:** It is actually possible to use torch modules inside a Keras Model without having to explicitly have them wrapped with the `TorchModuleWrapper` as evident by [this tweet](https://twitter.com/fchollet/status/1697381832164290754) from François Chollet. However, this doesn't seem to work at the point of time this example was created, as reported in [this issue](https://github.com/keras-team/keras-core/issues/834).
    """)
    return


@app.cell
def _(TimmClassifier, class_names, config, device, torch):
    # Now, we define the model and pass a random tensor to check the output shape
    model = TimmClassifier(
        model_name=config.model_name,
        freeze_backbone=config.freeze_backbone,
        dropout_rate=config.dropout_rate,
        num_classes=len(class_names)
    )
    model(torch.ones(1, *config.preprocess_config["input_size"]).to(device)).shape
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, in standard Keras fashion, all we need to do is compile the model and call `model.fit()`!
    """)
    return


@app.cell
def _(
    WandbMetricsLogger,
    WandbModelCheckpoint,
    config,
    dataloaders,
    keras,
    model,
):
    # Create exponential decay learning rate scheduler
    decay_steps = config.num_epochs * len(dataloaders["train"]) // config.batch_size
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, decay_steps=decay_steps, decay_rate=0.1,
    )

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr_scheduler),
        metrics=["accuracy"],
    )

    # Define the backend-agnostic WandB callbacks for KerasCore
    callbacks = [
        # Track experiment metrics
        WandbMetricsLogger(log_freq="batch"),
        # Track and version model checkpoints
        WandbModelCheckpoint("model.keras")
    ]

    # Train the model by calling model.fit
    model.fit(
        dataloaders["train"],
        validation_data=dataloaders["val"],
        epochs=config.num_epochs,
        callbacks=callbacks,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In order to know more about the backend-agnostic Keras callbacks for Weights & Biases, check out the [docs for wandb-addons](https://geekyrakshit.dev/wandb-addons/keras/keras_core/).
    """)
    return


if __name__ == "__main__":
    app.run()
