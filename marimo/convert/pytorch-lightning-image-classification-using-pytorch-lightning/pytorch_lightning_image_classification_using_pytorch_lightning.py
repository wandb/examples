# /// script
# dependencies = ["lightning", "torchvision", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pytorch-lightning-image-classification-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{pytorch-lightning-image-classification-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image Classification using PyTorch Lightning ⚡️

    We will build an image classification pipeline using PyTorch Lightning. We will follow this [style guide](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html) to increase the readability and reproducibility of our code. A cool explanation of this available [here](https://wandb.ai/wandb/wandb-lightning/reports/Image-Classification-using-PyTorch-Lightning--VmlldzoyODk1NzY).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up PyTorch Lightning and W&B

    For this tutorial, we need PyTorch Lightning(ain't that obvious!) and Weights and Biases.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: lightning torchvision !pip install lightning torchvision -q
    # install weights and biases
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You're gonna need these imports.
    """)
    return


@app.cell
def _():
    import lightning.pytorch as pl
    # your favorite machine learning tracking tool
    from lightning.pytorch.loggers import WandbLogger

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import random_split, DataLoader

    from torchmetrics import Accuracy

    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    import wandb

    return (
        Accuracy,
        CIFAR10,
        DataLoader,
        F,
        WandbLogger,
        nn,
        pl,
        random_split,
        torch,
        transforms,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now you'll need to login to you wandb account.
    """)
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔧 DataModule - The Data Pipeline we Deserve

    DataModules are a way of decoupling data-related hooks from the LightningModule so you can develop dataset agnostic models.

    It organizes the data pipeline into one shareable and reusable class. A datamodule encapsulates the five steps involved in data processing in PyTorch:
    - Download / tokenize / process.
    - Clean and (maybe) save to disk.
    - Load inside Dataset.
    - Apply transforms (rotate, tokenize, etc…).
    - Wrap inside a DataLoader.

    Learn more about datamodules [here](https://lightning.ai/docs/pytorch/stable/data/datamodule.html). Let's build a datamodule for the Cifar-10 dataset.
    """)
    return


@app.cell
def _(CIFAR10, DataLoader, pl, random_split, transforms):
    class CIFAR10DataModule(pl.LightningDataModule):
        def __init__(self, batch_size, data_dir: str = './'):
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = batch_size

            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
            self.num_classes = 10
    
        def prepare_data(self):
            CIFAR10(self.data_dir, train=True, download=True)
            CIFAR10(self.data_dir, train=False, download=True)
    
        def setup(self, stage=None):
            # Assign train/val datasets for use in dataloaders
            if stage == 'fit' or stage is None:
                cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
                self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

            # Assign test dataset for use in dataloader(s)
            if stage == 'test' or stage is None:
                self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
    
        def train_dataloader(self):
            return DataLoader(self.cifar_train, batch_size=self.batch_size, shuffle=True)

        def val_dataloader(self):
            return DataLoader(self.cifar_val, batch_size=self.batch_size)

        def test_dataloader(self):
            return DataLoader(self.cifar_test, batch_size=self.batch_size)

    return (CIFAR10DataModule,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📱 Callbacks

    A callback is a self-contained program that can be reused across projects. PyTorch Lightning comes with few [built-in callbacks](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html#built-in-callbacks) which are regularly used.
    Learn more about callbacks in PyTorch Lightning [here](https://lightning.ai/docs/pytorch/latest/extensions/callbacks.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Built-in Callbacks

    In this tutorial, we will use [Early Stopping](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.EarlyStopping.html#lightning.callbacks.EarlyStopping) and [Model Checkpoint](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.ModelCheckpoint.html#pytorch_lightning.callbacks.ModelCheckpoint) built-in callbacks. They can be passed to the `Trainer`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Custom Callbacks
    If you are familiar with Custom Keras callback, the ability to do the same in your PyTorch pipeline is just a cherry on the cake.

    Since we are performing image classification, the ability to visualize the model's predictions on some samples of images can be helpful. This in the form of a callback can help debug the model at an early stage.
    """)
    return


@app.cell
def _(pl, torch, wandb):
    class ImagePredictionLogger(pl.callbacks.Callback):
        def __init__(self, val_samples, num_samples=32):
            super().__init__()
            self.num_samples = num_samples
            self.val_imgs, self.val_labels = val_samples
    
        def on_validation_epoch_end(self, trainer, pl_module):
            # Bring the tensors to CPU
            val_imgs = self.val_imgs.to(device=pl_module.device)
            val_labels = self.val_labels.to(device=pl_module.device)
            # Get model prediction
            logits = pl_module(val_imgs)
            preds = torch.argmax(logits, -1)
            # Log the images as wandb Image
            trainer.logger.experiment.log({
                "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                               for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                     preds[:self.num_samples], 
                                                     val_labels[:self.num_samples])]
                })

    return (ImagePredictionLogger,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎺 LightningModule - Define the System

    The LightningModule defines a system and not a model. Here a system groups all the research code into a single class to make it self-contained. `LightningModule` organizes your PyTorch code into 5 sections:
    - Computations (`__init__`).
    - Train loop (`training_step`)
    - Validation loop (`validation_step`)
    - Test loop (`test_step`)
    - Optimizers (`configure_optimizers`)

    One can thus build a dataset agnostic model that can be easily shared. Let's build a system for Cifar-10 classification.
    """)
    return


@app.cell
def _(Accuracy, F, nn, pl, torch):
    class LitModel(pl.LightningModule):
        def __init__(self, input_shape, num_classes, learning_rate=2e-4):
            super().__init__()
        
            # log hyperparameters
            self.save_hyperparameters()
            self.learning_rate = learning_rate
        
            self.conv1 = nn.Conv2d(3, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 32, 3, 1)
            self.conv3 = nn.Conv2d(32, 64, 3, 1)
            self.conv4 = nn.Conv2d(64, 64, 3, 1)

            self.pool1 = torch.nn.MaxPool2d(2)
            self.pool2 = torch.nn.MaxPool2d(2)
        
            n_sizes = self._get_conv_output(input_shape)

            self.fc1 = nn.Linear(n_sizes, 512)
            self.fc2 = nn.Linear(512, 128)
            self.fc3 = nn.Linear(128, num_classes)

            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        # returns the size of the output tensor going into Linear layer from the conv block.
        def _get_conv_output(self, shape):
            batch_size = 1
            input = torch.autograd.Variable(torch.rand(batch_size, *shape))

            output_feat = self._forward_features(input) 
            n_size = output_feat.data.view(batch_size, -1).size(1)
            return n_size
        
        # returns the feature tensor from the conv block
        def _forward_features(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool1(F.relu(self.conv2(x)))
            x = F.relu(self.conv3(x))
            x = self.pool2(F.relu(self.conv4(x)))
            return x
    
        # will be used during inference
        def forward(self, x):
           x = self._forward_features(x)
           x = x.view(x.size(0), -1)
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = F.log_softmax(self.fc3(x), dim=1)
       
           return x
    
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
        
            # training metrics
            preds = torch.argmax(logits, dim=1)
            acc = self.accuracy(preds, y)
            self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
            self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        
            return loss
    
        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)

            # validation metrics
            preds = torch.argmax(logits, dim=1)
            acc = self.accuracy(preds, y)
            self.log('val_loss', loss, prog_bar=True)
            self.log('val_acc', acc, prog_bar=True)
            return loss
    
        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
        
            # validation metrics
            preds = torch.argmax(logits, dim=1)
            acc = self.accuracy(preds, y)
            self.log('test_loss', loss, prog_bar=True)
            self.log('test_acc', acc, prog_bar=True)
            return loss
    
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

    return (LitModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🚋 Train and Evaluate

    Now that we have organized our data pipeline using `DataModule` and model architecture+training loop using `LightningModule`, the PyTorch Lightning `Trainer` automates everything else for us.

    The Trainer automates:
    - Epoch and batch iteration
    - Calling of `optimizer.step()`, `backward`, `zero_grad()`
    - Calling of `.eval()`, enabling/disabling grads
    - Saving and loading weights
    - Weights and Biases logging
    - Multi-GPU training support
    - TPU support
    - 16-bit training support
    """)
    return


@app.cell
def _(CIFAR10DataModule):
    dm = CIFAR10DataModule(batch_size=32)
    # To access the x_dataloader we need to call prepare_data and setup.
    dm.prepare_data()
    dm.setup()

    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape
    return dm, val_samples


@app.cell
def _(
    ImagePredictionLogger,
    LitModel,
    WandbLogger,
    dm,
    pl,
    val_samples,
    wandb,
):
    model = LitModel((3, 32, 32), dm.num_classes)

    # Initialize wandb logger
    wandb_logger = WandbLogger(project='wandb-lightning', job_type='train')

    # Initialize Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
    checkpoint_callback = pl.callbacks.ModelCheckpoint()

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=2,
                         logger=wandb_logger,
                         callbacks=[early_stop_callback,
                                    ImagePredictionLogger(val_samples),
                                    checkpoint_callback],
                         )

    # Train the model ⚡🚅⚡
    trainer.fit(model, dm)

    # Evaluate the model on the held-out test set ⚡⚡
    trainer.test(dataloaders=dm.test_dataloader())

    # Close wandb run
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Final Thoughts
    I come from the TensorFlow/Keras ecosystem and find PyTorch a bit overwhelming even though it's an elegant framework. Just my personal experience though. While exploring PyTorch Lightning, I realized that almost all of the reasons that kept me away from PyTorch is taken care of. Here's a quick summary of my excitement:
    - Then: Conventional PyTorch model definition used to be all over the place. With the model in some `model.py` script and the training loop in the `train.py `file. It was a lot of looking back and forth to understand the pipeline.
    - Now: The `LightningModule` acts as a system where the model is defined along with the `training_step`, `validation_step`, etc. Now it's modular and shareable.
    - Then: The best part about TensorFlow/Keras is the input data pipeline. Their dataset catalog is rich and growing. PyTorch's data pipeline used to be the biggest pain point. In normal PyTorch code, the data download/cleaning/preparation is usually scattered across many files.
    - Now: The DataModule organizes the data pipeline into one shareable and reusable class. It's simply a collection of a `train_dataloader`, `val_dataloader`(s), `test_dataloader`(s) along with the matching transforms and data processing/downloads steps required.
    - Then: With Keras, one can call `model.fit` to train the model and `model.predict` to run inference on. `model.evaluate` offered a good old simple evaluation on the test data. This is not the case with PyTorch. One will usually find separate `train.py` and `test.py` files.
    - Now: With the `LightningModule` in place, the `Trainer` automates everything. One needs to just call `trainer.fit` and `trainer.test` to train and evaluate the model.
    - Then: TensorFlow loves TPU, PyTorch...well!
    - Now: With PyTorch Lightning, it's so easy to train the same model with multiple GPUs and even on TPU. Wow!
    - Then: I am a big fan of Callbacks and prefer writing custom callbacks. Something as trivial as Early Stopping used to be a point of discussion with conventional PyTorch.
    - Now: With PyTorch Lightning using Early Stopping and Model Checkpointing is a piece of cake. I can even write custom callbacks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎨 Conclusion and Resources

    I hope you find this report helpful. I will encourage to play with the code and train an image classifier with a dataset of your choice.

    Here are some resources to learn more about PyTorch Lightning:
    - [Step-by-step walk-through](https://lightning.ai/docs/pytorch/latest/starter/introduction.html) - This is one of the official tutorials. Their documentation is really well written and I highly encourage it as a good learning resource.
    - [Use Pytorch Lightning with Weights & Biases](https://wandb.me/lightning) - This is a quick colab that you can run through to learn more about how to use W&B with PyTorch Lightning.
    """)
    return


if __name__ == "__main__":
    app.run()
