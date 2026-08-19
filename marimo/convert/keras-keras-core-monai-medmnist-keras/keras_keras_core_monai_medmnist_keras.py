# /// script
# dependencies = ["monai-weekly", "namex", "wandb-addons @ git+https://github.com/soumik12345/wandb-addons"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/keras_core/monai_medmnist_keras.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
    # 🩺 Medical Image Classification Tutorial using MonAI and Keras

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/keras_core/monai_medmnist_keras.ipynb)

    This notebook demonstrates
    - an end-to-end training using [MonAI](https://github.com/Project-MONAI/MONAI) and [KerasCore](https://github.com/keras-team/keras-core).
    - how we can use the backend-agnostic Keras callbacks for [Weights & Biases](https://wandb.ai/site) to manage and track our experiment.

    Original Notebook: https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb
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
    - We install [monai](https://github.com/Project-MONAI/MONAI), a PyTorch-based, open-source framework for deep learning in healthcare imaging, part of the PyTorch Ecosystem.
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

    # install monai and wandb-addons
    # packages added via marimo's package management: git+https://github.com/soumik12345/wandb-addons !pip install -qq git+https://github.com/soumik12345/wandb-addons
    # packages added via marimo's package management: monai-weekly[pillow, tqdm] !pip install -q "monai-weekly[pillow, tqdm]"
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

    import shutil
    import tempfile
    import matplotlib.pyplot as plt
    import PIL
    import torch
    import numpy as np
    from sklearn.metrics import classification_report

    import keras_core as keras
    from keras_core.utils import TorchModuleWrapper

    from monai.apps import download_and_extract
    from monai.config import print_config
    from monai.data import decollate_batch, DataLoader
    from monai.metrics import ROCAUCMetric
    from monai.networks.nets import DenseNet121
    from monai.transforms import (
        Activations,
        EnsureChannelFirst,
        AsDiscrete,
        Compose,
        LoadImage,
        RandFlip,
        RandRotate,
        RandZoom,
        ScaleIntensity,
    )
    from monai.utils import set_determinism

    import wandb
    from wandb_addons.keras import WandbMetricsLogger, WandbModelCheckpoint

    return (
        Activations,
        AsDiscrete,
        Compose,
        DataLoader,
        DenseNet121,
        EnsureChannelFirst,
        LoadImage,
        PIL,
        RandFlip,
        RandRotate,
        RandZoom,
        ScaleIntensity,
        TorchModuleWrapper,
        WandbMetricsLogger,
        download_and_extract,
        keras,
        np,
        os,
        plt,
        tempfile,
        torch,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We initialize a [wandb run](https://docs.wandb.ai/guides/runs) and set the configs for the experiment.
    """)
    return


@app.cell
def _(wandb):
    wandb.init(project="keras-torch")

    config = wandb.config
    config.batch_size = 128
    config.num_epochs = 1
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup data directory

    You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.
    This allows you to save results and reuse downloads.
    If not specified a temporary directory will be used.
    """)
    return


@app.cell
def _(os, tempfile):
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    print(root_dir)
    return (root_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download dataset

    The MedNIST dataset was gathered from several sets from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Data+Usage+Policies+and+Restrictions),
    [the RSNA Bone Age Challenge](http://rsnachallenges.cloudapp.net/competitions/4),
    and [the NIH Chest X-ray dataset](https://cloud.google.com/healthcare/docs/resources/public-datasets/nih-chest).

    The dataset is kindly made available by [Dr. Bradley J. Erickson M.D., Ph.D.](https://www.mayo.edu/research/labs/radiology-informatics/overview) (Department of Radiology, Mayo Clinic)
    under the Creative Commons [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/).

    If you use the MedNIST dataset, please acknowledge the source.
    """)
    return


@app.cell
def _(download_and_extract, os, root_dir):
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"

    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)
    return (data_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read image filenames from the dataset folders

    First of all, check the dataset files and show some statistics.
    There are 6 folders in the dataset: Hand, AbdomenCT, CXR, ChestCT, BreastMRI, HeadCT,
    which should be used as the labels to train our classification model.
    """)
    return


@app.cell
def _(PIL, data_dir, os):
    class_names = sorted((x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))))
    num_class = len(class_names)
    image_files = [[os.path.join(data_dir, class_names[_i], x) for x in os.listdir(os.path.join(data_dir, class_names[_i]))] for _i in range(num_class)]
    num_each = [len(image_files[_i]) for _i in range(num_class)]
    image_files_list = []
    image_class = []
    for _i in range(num_class):
        image_files_list.extend(image_files[_i])
        image_class.extend([_i] * num_each[_i])
    num_total = len(image_class)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size
    print(f'Total image count: {num_total}')
    print(f'Image dimensions: {image_width} x {image_height}')
    print(f'Label names: {class_names}')
    print(f'Label counts: {num_each}')
    return class_names, image_class, image_files_list, num_class, num_total


@app.cell
def _(PIL, class_names, image_class, image_files_list, np, num_total, plt):
    plt.subplots(3, 3, figsize=(8, 8))
    for _i, k in enumerate(np.random.randint(num_total, size=9)):
        im = PIL.Image.open(image_files_list[k])
        arr = np.array(im)
        plt.subplot(3, 3, _i + 1)
        plt.xlabel(class_names[image_class[k]])
        plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare training, validation and test data lists

    Randomly select 10% of the dataset as validation and 10% as test.
    """)
    return


@app.cell
def _(image_class, image_files_list, np):
    val_frac = 0.1
    test_frac = 0.1
    length = len(image_files_list)
    indices = np.arange(length)
    np.random.shuffle(indices)
    test_split = int(test_frac * length)
    val_split = int(val_frac * length) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]
    train_x = [image_files_list[_i] for _i in train_indices]
    train_y = [image_class[_i] for _i in train_indices]
    val_x = [image_files_list[_i] for _i in val_indices]
    val_y = [image_class[_i] for _i in val_indices]
    test_x = [image_files_list[_i] for _i in test_indices]
    test_y = [image_class[_i] for _i in test_indices]
    print(f'Training count: {len(train_x)}, Validation count: {len(val_x)}, Test count: {len(test_x)}')
    return test_x, test_y, train_x, train_y, val_x, val_y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define MONAI transforms, Dataset and Dataloader to pre-process data
    """)
    return


@app.cell
def _(
    Activations,
    AsDiscrete,
    Compose,
    EnsureChannelFirst,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    np,
    num_class,
):
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )

    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])
    return train_transforms, val_transforms


@app.cell
def _(
    DataLoader,
    config,
    test_x,
    test_y,
    torch,
    train_transforms,
    train_x,
    train_y,
    val_transforms,
    val_x,
    val_y,
):
    class MedNISTDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]


    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)

    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=2)

    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, num_workers=2)
    return train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We typically define a model in PyTorch using [`torch.nn.Module`s](https://pytorch.org/docs/stable/notes/modules.html) which act as the building blocks of stateful computation. Even though Keras supports PyTorch as a backend, it does not mean that we can nest torch modules inside a [`keras_core.Model`](https://keras.io/keras_core/api/models/), because trainable variables inside a Keras Model is tracked exclusively via [Keras Layers](https://keras.io/keras_core/api/layers/).

    KerasCore provides us with a feature called `TorchModuleWrapper` which enables us to do exactly this. The `TorchModuleWrapper` is a Keras Layer that accepts a torch module and tracks its trainable variables, essentially converting the torch module into a Keras Layer. This enables us to put any torch modules inside a Keras Model and train them with a single `model.fit()`!

    The idea of the `TorchModuleWrapper` was proposed by Keras' creator [François Chollet](https://github.com/fchollet) on [this issue thread](https://github.com/keras-team/keras-core/issues/604).
    """)
    return


@app.cell
def _(DenseNet121, TorchModuleWrapper, keras, num_class, torch, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = keras.Input(shape=(1, 64, 64))
    outputs = TorchModuleWrapper(
        DenseNet121(
            spatial_dims=2, in_channels=1, out_channels=num_class
        )
    )(inputs)
    model = keras.Model(inputs, outputs)

    # model = MedMnistModel()
    model(next(iter(train_loader))[0].to(device)).shape
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:** It is actually possible to use torch modules inside a Keras Model without having to explicitly have them wrapped with the `TorchModuleWrapper` as evident by [this tweet](https://twitter.com/fchollet/status/1697381832164290754) from François Chollet. However, this doesn't seem to work at the point of time this example was created, as reported in [this issue](https://github.com/keras-team/keras-core/issues/834).
    """)
    return


@app.cell
def _(
    WandbMetricsLogger,
    config,
    keras,
    model,
    train_loader,
    val_loader,
    wandb,
):
    # Compile the model
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(1e-5),
        metrics=["accuracy"],
    )

    # Define the backend-agnostic WandB callbacks for KerasCore
    callbacks = [
        # Track experiment metrics
        WandbMetricsLogger(log_freq="batch")
    ]

    # Train the model by calling model.fit
    model.fit(
        train_loader,
        validation_data=val_loader,
        epochs=config.num_epochs,
        callbacks=callbacks,
    )

    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
