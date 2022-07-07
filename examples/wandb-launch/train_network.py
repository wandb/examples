import os
import json
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms


def get_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    return transform


def build_dataset(batch_size=100, train=True):
    # workaround to fetch MNIST data
    if not os.path.exists('./MNIST'):
        os.system('wget www.di.ens.fr/~lelarge/MNIST.tar.gz')
        os.system('tar -zxvf MNIST.tar.gz')
    dataset = datasets.MNIST(".", train=train, download=False,
        transform=get_transform())
    if batch_size is None:
        batch_size = dataset.data.shape[0]
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # fully-connected, single hidden layer
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))
    return network.to(DEVICE)
        

def build_optimizer(network, optimizer, learning_rate):
    optimizer = torch.optim.Adam(network.parameters(),
        lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()
        loss.backward()
        optimizer.step()
        wandb.log({"batch loss": loss.item()})
    return cumu_loss / len(loader)


def train(project='mnist_train', config=None):
    with wandb.init(project=project, config=config):
        config = wandb.config
        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)
        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})
    return network

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model on MNIST')
    parser.add_argument('config', help='location of the run config file')
    args = parser.parse_args()
    with open(args.config) as fp:
        run_config = json.load(fp)
    network = train(config=run_config)