"""pytorch-lightning example with W&B logging.

Based on this colab:
https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31
"""

import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # not the best model...
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def prepare_data(self):
        # download MNIST data only once
        MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
        MNIST(os.getcwd(), train=False, download=True, transform=transforms.ToTensor())

    def forward(self, x):
        # called with self(x)
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor()), batch_size=32
        )

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor()), batch_size=32
        )

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            MNIST(os.getcwd(), train=True, download=False, transform=transforms.ToTensor()), batch_size=32
        )


if __name__ == "__main__":
    wandb_logger = WandbLogger()
    mnist_model = MNISTModel()
    trainer = pl.Trainer(gpus=0, max_epochs=2, logger=wandb_logger)
    trainer.fit(mnist_model)
    trainer.test(mnist_model)
