# /// script
# dependencies = ["pip", "restorers @ git+https://github.com/soumik12345/restorers.git", "setuptools"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/restorers/Train_MirNetv2_Restorers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{restorers-mirnetv2-train} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌈 Restorers + WandB 🪄🐝

    <!--- @wandbcode{restorers-mirnetv2-train} -->

    This notebook shows how to train a [MirNetv2](https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/zamir-2022-mirnetv2.pdf) model for low-light enhancement using [**restorers**](https://github.com/soumik12345/restorers) and [**wandb**](https://wandb.ai/site). For more details regarding usage of restorers, refer to the following report:

    [![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/ml-colabs/low-light-enhancement/reports/Lighting-up-Images-in-the-Deep-Learning-Era--VmlldzozNzE4Njkz)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: pip setuptools !pip install -q --upgrade pip setuptools
    # packages added via marimo's package management: git+https://github.com/soumik12345/restorers.git !pip install git+https://github.com/soumik12345/restorers.git
    return


@app.cell
def _():
    import wandb
    import tensorflow as tf
    from restorers.dataloader import LOLDataLoader

    return LOLDataLoader, tf, wandb


@app.cell
def _(LOLDataLoader, wandb):
    wandb.init(project="low-light-enhancement")

    # define dataloader for the LoL dataset
    data_loader = LOLDataLoader(
        # size of image crops on which we will train
        image_size=128,
        # bit depth of the images
        bit_depth=8,
        # fraction of images for validation
        val_split=0.2,
        # visualize the dataset on WandB or not
        visualize_on_wandb=True,
        # the wandb artifact address of the dataset,
        # this can be found from the `Usage` tab of
        # the aforemenioned weave panel
        dataset_artifact_address="ml-colabs/dataset/LoL:v0",
    )

    # call `get_datasets` on the `data_loader` to get
    # the TensorFlow datasets corresponding to the 
    # training and validation splits
    datasets = data_loader.get_datasets(batch_size=2)
    train_dataset, val_dataset = datasets
    return train_dataset, val_dataset


@app.cell
def _():
    # import MirNetv2 from restorers
    from restorers.model import MirNetv2


    # define the MirNetv2 model; this gives us a `tf.keras.Model`
    model = MirNetv2(
        # number of channels in the feature map
        channels=80,
        # number of multi-scale residual blocks
        channel_factor=1.5,
        # factor by which number of the number of output channels vary
        num_mrb_blocks=2,
        # number of groups in which the input is split along the
        # channel axis in the convolution layers.
        add_residual_connection=True,
    )
    return (model,)


@app.cell
def _(model, tf):
    from restorers.losses import CharbonnierLoss
    # import Peak Signal-to-Noise Ratio and Structural Similarity metrics,
    # implemented as part of restorers
    from restorers.metrics import PSNRMetric, SSIMMetric


    loss = CharbonnierLoss(
        # a small constant to avoid division by zero
        epsilon=1e-3,
        # type of reduction applied to the loss, it needs to be
        # explicitly specified in case of distributed training
        reduction=tf.keras.losses.Reduction.SUM,
    )


    optimizer = tf.keras.optimizers.experimental.AdamW(learning_rate=2e-4,)

    psnr_metric = PSNRMetric(max_val=1.0) # peak signal-to-noise ratio metric
    ssim_metric = SSIMMetric(max_val=1.0) # structural similarity metric

    model.compile(
        optimizer=optimizer, loss=loss, metrics=[psnr_metric, ssim_metric]
    )
    return


@app.cell
def _(model, train_dataset, val_dataset):
    # import the wandb callbacks for keras
    from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


    callbacks = [
        # define the metrics logger callback;
        # we set the `log_freq="batch"` explicitly
        # to the metrics are logged both batch-wise and epoch-wise
        WandbMetricsLogger(log_freq="batch"),
        # define the model checkpoint callback
        WandbModelCheckpoint(
            filepath="checkpoint",
            monitor="val_loss",
            save_best_only=False,
            save_weights_only=False,
            initial_value_threshold=None,
        )
    ]

    # call model.fit()
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=callbacks,
    )
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
