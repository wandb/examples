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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Transfer_Learning_Using_PyTorch_Lightning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pytorch-lightning-transfer-learning-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{pytorch-lightning-transfer-learning-colab} -->

    # Transfer Learning Using PyTorch Lightning ⚡️

    In this colab, we will extend the pipeline [here](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Image_Classification_using_PyTorch_Lightning.ipynb) to perform transfer learning with PyTorch Lightning.

    Transfer Learning is a technique where the knowledge learned while training a model for "task" A and can be used for "task" B. Here A and B can be the same deep learning tasks but on a different dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up PyTorch Lightning and W&B

    For this tutorial, we need PyTorch Lightning and Weights and Biases.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb lightning torchvision !pip install wandb lightning torchvision -qqq
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You're gonna need these imports.
    """)
    return


@app.cell
def _():
    import os

    import lightning.pytorch as pl
    # your favorite machine learning tracking tool
    from lightning.pytorch.loggers import WandbLogger

    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import random_split, DataLoader

    from torchmetrics import Accuracy

    from torchvision import transforms
    from torchvision.datasets import StanfordCars
    from torchvision.datasets.utils import download_url
    import torchvision.models as models


    import wandb

    return (
        Accuracy,
        DataLoader,
        StanfordCars,
        WandbLogger,
        models,
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
    ## The Dataset 💿

    We will be using the StanfordCars dataset to train our image classifier. It contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.
    """)
    return


@app.cell
def _(DataLoader, StanfordCars, pl, random_split, transforms):
    class StanfordCarsDataModule(pl.LightningDataModule):
        def __init__(self, batch_size, data_dir: str = './'):
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = batch_size

            # Augmentation policy for training set
            self.augmentation = transforms.Compose([
                  transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                  transforms.RandomRotation(degrees=15),
                  transforms.RandomHorizontalFlip(),
                  transforms.CenterCrop(size=224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
            # Preprocessing steps applied to validation and test set.
            self.transform = transforms.Compose([
                  transforms.Resize(size=256),
                  transforms.CenterCrop(size=224),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
            ])
        
            self.num_classes = 196

        def prepare_data(self):
            pass

        def setup(self, stage=None):
            # build dataset
            dataset = StanfordCars(root=self.data_dir, download=True, split="train")
            # split dataset
            self.train, self.val = random_split(dataset, [6500, 1644])

            self.test = StanfordCars(root=self.data_dir, download=True, split="test")
        
            self.test = random_split(self.test, [len(self.test)])[0]

            self.train.dataset.transform = self.augmentation
            self.val.dataset.transform = self.transform
            self.test.dataset.transform = self.transform
        
        def train_dataloader(self):
            return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=2)

        def val_dataloader(self):
            return DataLoader(self.val, batch_size=self.batch_size, num_workers=2)

        def test_dataloader(self):
            return DataLoader(self.test, batch_size=self.batch_size, num_workers=2)

    return (StanfordCarsDataModule,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LightingModule - Define the System

    Let us look at the model definition to see how transfer learning can be used with PyTorch Lightning.
    In the `LitModel` class, we can use the pre-trained model provided by Torchvision as a feature extractor for our classification model. Here we are using ResNet-18. A list of pre-trained models provided by PyTorch Lightning can be found here.
    - When `pretrained=True`, we use the pre-trained weights; otherwise, the weights are initialized randomly.
    - If `.eval()` is used, then the layers are frozen.
    - A single `Linear` layer is used as the output layer. We can have multiple layers stacked over the `feature_extractor`.

    Setting the `transfer` argument to `True` will enable transfer learning.
    """)
    return


@app.cell
def _(Accuracy, models, nn, pl, torch):
    class LitModel(pl.LightningModule):
        def __init__(self, input_shape, num_classes, learning_rate=2e-4, transfer=False):
            super().__init__()
        
            # log hyperparameters
            self.save_hyperparameters()
            self.learning_rate = learning_rate
            self.dim = input_shape
            self.num_classes = num_classes
        
            # transfer learning if pretrained=True
            self.feature_extractor = models.resnet18(pretrained=transfer)

            if transfer:
                # layers are frozen by using eval()
                self.feature_extractor.eval()
                # freeze params
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
        
            n_sizes = self._get_conv_output(input_shape)

            self.classifier = nn.Linear(n_sizes, num_classes)

            self.criterion = nn.CrossEntropyLoss()
            self.accuracy = Accuracy()
  
        # returns the size of the output tensor going into the Linear layer from the conv block.
        def _get_conv_output(self, shape):
            batch_size = 1
            tmp_input = torch.autograd.Variable(torch.rand(batch_size, *shape))

            output_feat = self._forward_features(tmp_input) 
            n_size = output_feat.data.view(batch_size, -1).size(1)
            return n_size
        
        # returns the feature tensor from the conv block
        def _forward_features(self, x):
            x = self.feature_extractor(x)
            return x
    
        # will be used during inference
        def forward(self, x):
           x = self._forward_features(x)
           x = x.view(x.size(0), -1)
           x = self.classifier(x)
       
           return x
    
        def training_step(self, batch):
            batch, gt = batch[0], batch[1]
            out = self.forward(batch)
            loss = self.criterion(out, gt)

            acc = self.accuracy(out, gt)

            self.log("train/loss", loss)
            self.log("train/acc", acc)

            return loss
    
        def validation_step(self, batch, batch_idx):
            batch, gt = batch[0], batch[1]
            out = self.forward(batch)
            loss = self.criterion(out, gt)

            self.log("val/loss", loss)

            acc = self.accuracy(out, gt)
            self.log("val/acc", acc)

            return loss
    
        def test_step(self, batch, batch_idx):
            batch, gt = batch[0], batch[1]
            out = self.forward(batch)
            loss = self.criterion(out, gt)
        
            return {"loss": loss, "outputs": out, "gt": gt}
    
        def test_epoch_end(self, outputs):
            loss = torch.stack([x['loss'] for x in outputs]).mean()
            output = torch.cat([x['outputs'] for x in outputs], dim=0)
        
            gts = torch.cat([x['gt'] for x in outputs], dim=0)
        
            self.log("test/loss", loss)
            acc = self.accuracy(output, gts)
            self.log("test/acc", acc)
        
            self.test_gts = gts
            self.test_output = output
    
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    return (LitModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train your Model 🏋️‍♂️
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To train the model, we instantiate the `StanfordCarsDataModule` and the `LitModel` along with the PyTorch Lightning Trainer. To the `Trainer`, we will pass the `WandbLogger` as the logger to use W&B to track the metrics during model training!
    """)
    return


@app.cell
def _(LitModel, StanfordCarsDataModule, WandbLogger, pl):
    dm = StanfordCarsDataModule(batch_size=32)
    model = LitModel((3, 300, 300), 196, transfer=True)
    trainer = pl.Trainer(logger=WandbLogger(project="TransferLearning"), max_epochs=10, accelerator="gpu")
    return dm, model, trainer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are good to go! Let's train our model!
    """)
    return


@app.cell
def _(dm, model, trainer):
    trainer.fit(model, dm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that the model is trained, let's see how it performs on the test set
    """)
    return


@app.cell
def _(dm, model, trainer):
    trainer.test(model, dm)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's close our W&B run, so we call `wandb.finish()`.
    """)
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The workspace generated to compare training the model from scratch vs using transfer learning is available [here](https://wandb.ai/manan-goel/StanfordCars). The conclusions that can be drawn from this are explained in detail in [this report](https://wandb.ai/wandb/wandb-lightning/reports/Transfer-Learning-Using-PyTorch-Lightning--VmlldzoyMzMxMzk4/edit).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    I will encourage you to play with the code and train an image classifier with a dataset of your choice from scratch and using transfer learning.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To learn more about transfer learning check out these resources:
    - [Gotchas of transfer learning for image classification](https://docs.google.com/presentation/d/1s29WOQoQvBD5KoPUzE5TPcavjqno8ZgnZaSljHGGHVU/edit?usp=sharing) by Sayak Paul.
    - [Transfer Learning with Keras and Deep Learning by PyImageSearch.](https://www.pyimagesearch.com/2019/05/20/transfer-learning-with-keras-and-deep-learning/)
    - [Transfer Learning - Machine Learning's Next Frontier](https://ruder.io/transfer-learning/) by Sebastian Ruder.
    """)
    return


if __name__ == "__main__":
    app.run()
