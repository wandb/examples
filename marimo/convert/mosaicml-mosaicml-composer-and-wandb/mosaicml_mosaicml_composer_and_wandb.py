# /// script
# dependencies = ["mosaicml", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{mosaicml} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <img src="https://raw.githubusercontent.com/mosaicml/composer/dev/docs/source/_static/images/header_dark.svg" width="400" alt="mosaicml" />

    <!--- @wandbcode{mosaicml} -->

    # Running fast with MosaicML Composer and Weight and Biases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [MosaicML Composer](https://docs.mosaicml.com) is a library for training neural networks better, faster, and cheaper. It contains many state-of-the-art methods for accelerating neural network training and improving generalization, along with an optional Trainer API that makes composing many different enhancements easy.

    Coupled with [Weights & Biases integration](https://docs.wandb.ai/guides/integrations/composer), you can quickly train and monitor models for full traceability and reproducibility with only 2 extra lines of code:

    ```python
    from composer import Trainer
    from composer.loggers import WandBLogger

    wandb_logger = WandBLogger(init_params=init_params)
    trainer = Trainer(..., logger=wandb_logger)
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    W&B integration with Composer can automatically:
    * log your configuration parameters
    * log your losses and metrics
    * log gradients and parameter distributions
    * log your model
    * keep track of your code
    * log your system metrics (GPU, CPU, memory, temperature, etc)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🛠️ Installation and set-up

    We need to install the following libraries:
    * [mosaicml-composer](https://docs.mosaicml.com/en/v0.5.0/getting_started/installation.html) to set up and train our models
    * [wandb](https://docs.wandb.ai/) to instrument our training
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb mosaicml !pip install -Uq wandb mosaicml
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use the Composer `Trainer` class with Weights and Biases 🏋️‍♀️

    W&B integration with MosaicML-Composer is built into the `Trainer` and can be configured to add extra functionalities through `WandBLogger`:

    * logging of Artifacts: Use `log_artifacts=True` to log model checkpoints as `wandb.Artifacts`. You can setup how often by passing an int value to `log_artifacts_every_n_batches` (default = 100)
    * you can also pass any parameter that you would pass to `wandb.init` in `init_params` as a dictionary. For example, you could pass `init_params = {"project":"try_mosaicml", "name":"benchmark", "entity":"user_name"}`.

    For more details refer to [Logger documentation](https://docs.mosaicml.com/en/latest/api_reference/composer.loggers.wandb_logger.html#composer.loggers.wandb_logger.WandBLogger) and [Wandb docs](https://docs.wandb.ai)
    """)
    return


@app.cell
def _():
    EPOCHS = 5
    BS = 32
    return BS, EPOCHS


@app.cell
def _():
    import wandb

    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    from composer import Callback, State, Logger, Trainer
    from composer.models import mnist_model
    from composer.loggers import WandBLogger
    from composer.callbacks import SpeedMonitor, LRMonitor
    from composer.algorithms import LabelSmoothing, CutMix, ChannelsLast

    return (
        Callback,
        ChannelsLast,
        CutMix,
        DataLoader,
        LRMonitor,
        LabelSmoothing,
        Logger,
        SpeedMonitor,
        State,
        Trainer,
        WandBLogger,
        datasets,
        mnist_model,
        transforms,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    let's grab a copy of MNIST from `torchvision`
    """)
    return


@app.cell
def _(DataLoader, datasets, transforms):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_dataloader = DataLoader(dataset, batch_size=128)
    return (train_dataloader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we can import a simple ConvNet model to try
    """)
    return


@app.cell
def _(mnist_model):
    model = mnist_model(num_classes=10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📊 Tracking the experiment
    > we define the `wandb.init` params here
    """)
    return


@app.cell
def _(BS, EPOCHS, WandBLogger):
    # config params to log
    config = {"epochs":EPOCHS,
              "batch_size":BS,
              "model_name":"MNIST_Classifier"}

    # these will get passed to wandb.init(**init_params)
    wandb_init_kwargs = {"config":config}

    # setup of the logger
    wandb_logger = WandBLogger(project="mnist-composer",
                               log_artifacts=True,
                               init_kwargs=wandb_init_kwargs)
    return (wandb_logger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we are able to tweak what are we logging using `Callbacks` into the `Trainer` class.
    """)
    return


@app.cell
def _(LRMonitor, SpeedMonitor):
    callbacks = [LRMonitor(),    # Logs the learning rate
                 SpeedMonitor(), # Logs the training throughput
                ]
    return (callbacks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we include callbacks that measure the model throughput (and the learning rate) and logs them to Weights & Biases. [Callbacks](https://docs.mosaicml.com/en/latest/trainer/callbacks.html) control what is being logged, whereas loggers specify where the information is being saved. For more information on loggers, see [Logging](https://docs.mosaicml.com/en/latest/trainer/logging.html).
    """)
    return


@app.cell
def _(
    ChannelsLast,
    CutMix,
    LabelSmoothing,
    Trainer,
    callbacks,
    mnist_model,
    train_dataloader,
    wandb_logger,
):
    trainer = Trainer(
        model=mnist_model(num_classes=10),
        train_dataloader=train_dataloader,
        max_duration="2ep",
        loggers=[wandb_logger],    # Pass your WandbLogger
        callbacks=callbacks,
        algorithms=[
            LabelSmoothing(smoothing=0.1),
            CutMix(alpha=1.0),
            ChannelsLast(),
            ]
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    once we are ready to train we call `fit`
    """)
    return


@app.cell
def _(trainer):
    trainer.fit()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We close the Trainer to properly finish all callbacks and loggers
    """)
    return


@app.cell
def _(trainer):
    trainer.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ⚙️ Advanced: Using callbacks to log sample predictions

    > Composer is extensible through its callback system.

    We create a custom callback to automatically log sample predictions during validation.
    """)
    return


@app.cell
def _(Callback, Logger, State, wandb):
    class LogPredictions(Callback):

        def __init__(self, num_samples=100):
            super().__init__()
            self.num_samples = num_samples
            self.data = []

        def batch_end(self, state: State, logger: Logger):
            """Compute predictions per batch and stores them on self.data"""
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data = self.data + data

        def epoch_end(self, state: State, logger: Logger):
            """Create a wandb.Table and logs it"""
            columns = ['image', 'ground truth', 'prediction']
            table = wandb.Table(columns=columns, data=self.data[:self.num_samples])
            wandb.log({'predictions_table': table}, step=int(state.timestamp.batch))

    return (LogPredictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    we add `LogPredictions` to the other callbacks
    """)
    return


@app.cell
def _(LogPredictions, callbacks):
    callbacks.append(LogPredictions())
    return


@app.cell
def _(trainer):
    trainer.close()
    return


@app.cell
def _(
    ChannelsLast,
    CutMix,
    LabelSmoothing,
    Trainer,
    callbacks,
    mnist_model,
    train_dataloader,
    wandb_logger,
):
    trainer_1 = Trainer(model=mnist_model(num_classes=10), train_dataloader=train_dataloader, max_duration='2ep', loggers=[wandb_logger], callbacks=callbacks, algorithms=[LabelSmoothing(smoothing=0.1), CutMix(alpha=1.0), ChannelsLast()])  # Pass your WandbLogger
    return (trainer_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Once we're ready to train, we just call the `fit` method.
    """)
    return


@app.cell
def _(trainer_1):
    trainer_1.fit()
    trainer_1.close()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can monitor losses, metrics, gradients, parameters and sample predictions as the model trains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![composer.png](https://i.imgur.com/VFZLOB3.png?1)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📚 Resources

    * We are excited to showcase this early support of [MosaicML-Composer](https://docs.mosaicml.com/en/latest/index.html) go ahead and try this new state of the art framework.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ❓ Questions about W&B

    If you have any questions about using W&B to track your model performance and predictions, please reach out to the [wandb community](https://community.wandb.ai).
    """)
    return


if __name__ == "__main__":
    app.run()
