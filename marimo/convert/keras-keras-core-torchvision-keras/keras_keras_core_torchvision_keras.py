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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/keras_core/torchvision-keras.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{keras_core_torchvision} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://keras.io/img/logo-k-keras-wb.png" width="200" alt="Keras" />
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{keras_core_torchvision} -->

    # 🔥 Fine-tune a TorchVision Model with Keras and WandB 🦄

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/keras_core/torchvision_keras.ipynb)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    [TorchVision](https://pytorch.org/vision/stable/index.html) is a library part of the [PyTorch](http://pytorch.org/) project that consists of popular datasets, model architectures, and common image transformations for computer vision. This example demonstrates how we can perform transfer learning for image classification using a pre-trained backbone model from TorchVision on the [Imagenette dataset](https://github.com/fastai/imagenette) using KerasCore. We will also demonstrate the compatibility of KerasCore with an input system consisting of [Torch Datasets and Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html).

    ### References:

    - [Customizing what happens in `fit()` with PyTorch](https://keras.io/keras_core/guides/custom_train_step_in_torch/)
    - [PyTorch Datasets and Dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
    - [Transfer learning for Computer Vision using PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

    ## Setup

    - We install the `main` branch of [KerasCore](https://github.com/keras-team/keras-core), this lets us use the latest feature merged in KerasCore.
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

    # install wandb-addons
    # packages added via marimo's package management: git+https://github.com/soumik12345/wandb-addons !pip install -qq git+https://github.com/soumik12345/wandb-addons
    return


@app.cell
def _():
    import os
    os.environ["KERAS_BACKEND"] = "torch"

    import numpy as np
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import torchvision
    from torchvision import datasets, models, transforms

    import keras_core as keras
    from keras_core.utils import TorchModuleWrapper

    import wandb
    from wandb_addons.keras import WandbMetricsLogger, WandbModelCheckpoint

    return (
        TorchModuleWrapper,
        WandbMetricsLogger,
        WandbModelCheckpoint,
        datasets,
        keras,
        models,
        nn,
        np,
        os,
        plt,
        torch,
        tqdm,
        transforms,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define the Hyperparameters
    """)
    return


@app.cell
def _(wandb):
    wandb.init(project="keras-torch", entity="ml-colabs", job_type="torchvision/train")

    config = wandb.config
    config.batch_size = 32
    config.image_size = 224
    config.freeze_backbone = True
    config.initial_learning_rate = 1e-3
    config.num_epochs = 5
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Creating the Torch Datasets and Dataloaders

    In this example, we would train an image classification model on the [Imagenette dataset](https://github.com/fastai/imagenette). Imagenette is a subset of 10 easily classified classes from [Imagenet](https://www.image-net.org/) (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute).
    """)
    return


@app.cell
def _(keras):
    # Fetch the imagenette dataset
    data_dir = keras.utils.get_file(
        fname="imagenette2-320.tgz",
        origin="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        extract=True,
    )
    data_dir = data_dir.replace(".tgz", "")
    return (data_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we define pre-processing and augmentation transforms from TorchVision for the train and validation sets.
    """)
    return


@app.cell
def _(config, transforms):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(config.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return (data_transforms,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we will use TorchVision and the [`torch.utils.data`](https://pytorch.org/docs/stable/data.html) packages for creating the dataloaders for trainig and validation.
    """)
    return


@app.cell
def _(config, data_dir, data_transforms, datasets, os, torch):
    # Define the train and validation datasets
    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(data_dir, x), data_transforms[x]
        )
        for x in ['train', 'val']
    }

    # Define the torch dataloaders corresponding to the
    # train and validation dataset
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
    return class_names, dataloaders


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let us visualize a few samples from the training dataloader.
    """)
    return


@app.cell
def _(class_names, dataloaders, np, plt):
    plt.figure(figsize=(10, 10))
    _sample_images, _sample_labels = next(iter(dataloaders['train']))
    _sample_images = _sample_images.numpy()
    _sample_labels = _sample_labels.numpy()
    for _idx in range(9):
        ax = plt.subplot(3, 3, _idx + 1)
        _image = _sample_images[_idx].transpose((1, 2, 0))
        _mean = np.array([0.485, 0.456, 0.406])
        _std = np.array([0.229, 0.224, 0.225])
        _image = _std * _image + _mean
        _image = np.clip(_image, 0, 1)
        plt.imshow(_image)
        plt.title('Ground Truth Label: ' + class_names[int(_sample_labels[_idx])])
        plt.axis('off')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Image Classification Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We typically define a model in PyTorch using [`torch.nn.Module`s](https://pytorch.org/docs/stable/notes/modules.html) which act as the building blocks of stateful computation. Let us define the ResNet18 model from the TorchVision package as a `torch.nn.Module` pre-trained on the [Imagenet1K dataset](https://huggingface.co/datasets/imagenet-1k).
    """)
    return


@app.cell
def _(models, nn):
    # Define the pre-trained resnet18 module from TorchVision
    resnet_18 = models.resnet18(weights='IMAGENET1K_V1')

    # We set the classification head of the pre-trained ResNet18
    # module to an identity module
    resnet_18.fc = nn.Identity()
    return (resnet_18,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ven though Keras supports PyTorch as a backend, it does not mean that we can nest torch modules inside a [`keras_core.Model`](https://keras.io/keras_core/api/models/), because trainable variables inside a Keras Model is tracked exclusively via [Keras Layers](https://keras.io/keras_core/api/layers/).

    KerasCore provides us with a feature called `TorchModuleWrapper` which enables us to do exactly this. The `TorchModuleWrapper` is a Keras Layer that accepts a torch module and tracks its trainable variables, essentially converting the torch module into a Keras Layer. This enables us to put any torch modules inside a Keras Model and train them with a single `model.fit()`!
    """)
    return


@app.cell
def _(TorchModuleWrapper, config, resnet_18):
    # We set the trainable ResNet18 backbone to be a Keras Layer
    # using `TorchModuleWrapper`
    backbone = TorchModuleWrapper(resnet_18)

    # We set this to `False` if you want to freeze the backbone
    backbone.trainable = config.freeze_backbone
    return (backbone,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we will build a Keras functional model with the backbone layer.
    """)
    return


@app.cell
def _(backbone, class_names, config, keras):
    inputs = keras.Input(shape=(3, config.image_size, config.image_size))
    x = backbone(inputs)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(len(class_names))(x)
    outputs = keras.activations.softmax(x, axis=1)
    model = keras.Model(inputs, outputs, name="ResNet18_Classifier")

    model.summary()
    return (model,)


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
        initial_learning_rate=config.initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=0.1,
    )

    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(lr_scheduler),
        metrics=["accuracy"],
    )

    # Define the backend-agnostic WandB callbacks for KerasCore
    callbacks = [
        # Track experiment metrics with WandB
        WandbMetricsLogger(log_freq="batch"),
        # Save best model checkpoints to WandB
        WandbModelCheckpoint(
            filepath="model.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    # Train the model by calling model.fit
    history = model.fit(
        dataloaders["train"],
        validation_data=dataloaders["val"],
        epochs=config.num_epochs,
        callbacks=callbacks,
    )
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluation and Inference

    Now, we let us load the best model weights checkpoint and evaluate the model.
    """)
    return


@app.cell
def _(dataloaders, model, os, wandb):
    wandb.init(
        project="keras-torch", entity="ml-colabs", job_type="torchvision/eval"
    )
    artifact = wandb.use_artifact(
        'ml-colabs/keras-torch/run_hiceci7f_model:latest', type='model'
    )
    artifact_dir = artifact.download()

    model.load_weights(os.path.join(artifact_dir, "model.weights.h5"))

    _, val_accuracy = model.evaluate(dataloaders["val"])
    wandb.log({"Validation-Accuracy": val_accuracy})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, let us visualize the some predictions of the model
    """)
    return


@app.cell
def _(class_names, dataloaders, keras, model, np, tqdm, wandb):
    table = wandb.Table(columns=['Image', 'Ground-Truth', 'Prediction'] + ['Confidence-' + cls for cls in class_names])
    _sample_images, _sample_labels = next(iter(dataloaders['train']))
    sample_pred_probas = model(_sample_images.to('cuda')).detach()
    sample_pred_logits = keras.ops.argmax(sample_pred_probas, axis=1)
    sample_pred_logits = sample_pred_logits.to('cpu').numpy()
    sample_pred_probas = sample_pred_probas.to('cpu').numpy()
    _sample_images = _sample_images.numpy()
    _sample_labels = _sample_labels.numpy()
    # We perform inference and detach the predicted probabilities from the Torch
    # computation graph with a tensor that does not require gradient computation.
    for _idx in tqdm(range(_sample_images.shape[0])):
        _image = _sample_images[_idx].transpose((1, 2, 0))
        _mean = np.array([0.485, 0.456, 0.406])
        _std = np.array([0.229, 0.224, 0.225])
        _image = _std * _image + _mean
        _image = np.clip(_image, 0, 1)
        table.add_data(wandb.Image(_image), class_names[int(_sample_labels[_idx])], class_names[int(sample_pred_logits[_idx])], *sample_pred_probas[_idx].tolist())
    wandb.log({'Evaluation-Table': table})
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
