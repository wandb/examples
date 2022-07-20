import os
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, transforms

import wandb


def get_transform():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    return transform


def build_dataset(data_dir='.', batch_size=100, train=True):
    dataset = datasets.MNIST(data_dir, train=train, download=False,
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
    return network
        

def build_optimizer(network, optimizer, learning_rate):
    optimizer = torch.optim.Adam(network.parameters(),
        lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer, device):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = nn.functional.nll_loss(network(data), target)
        cumu_loss += loss.item()
        loss.backward()
        optimizer.step()
        wandb.log({"batch loss": loss.item()})
    return cumu_loss / len(loader)


def download_data(run, tag='latest'):
    artifact = run.use_artifact(f'{run.entity}/{run.project}/MNIST:{tag}', type='dataset')
    artifact_dir = artifact.download()
    return artifact_dir


def train(run):
    config = run.config
    loader = build_dataset(config.data_dir, config.batch_size, train=True)
    network = build_network(config.fc_layer_size, config.dropout).to(config.device)
    optimizer = build_optimizer(network, config.optimizer, config.learning_rate)
    for epoch in range(config.epochs):
        avg_loss = train_epoch(network, loader, optimizer, config.device)
        run.log({"loss": avg_loss, "epoch": epoch})
    return network


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train a model on MNIST')
    parser.add_argument('-e', '--entity', help='wandb entity (usually your username)')
    parser.add_argument('-p', '--project', help='wandb project')
    parser.add_argument('-c', '--config', help='location of the run config file')
    args = parser.parse_args()
    if args.config is None:
        args.config = 'run_config.yaml'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with wandb.init(entity=args.entity, project=args.project, config=args.config) as run:
        data_dir = download_data(run, tag='latest')
        run.config.update({
            'data_dir': data_dir,
            'device': DEVICE
        })
        network = train(run)
        torch.save(network.state_dict(), './model.pt')
        model_artifact = wandb.Artifact('model', type='model')
        model_artifact.add_file('./model.pt')
        run.log_artifact(model_artifact)
