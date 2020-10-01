import argparse
import os

from ray import tune
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test, train
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.integration.wandb import WandbLogger
import torch
import torch.optim as optim
import wandb

'''
Make sure that os.environ['WANDB_API_KEY'] is set. Ray tune requires you to manually pass API key
or API Key file path if it cannot find WANDB_API_KEY environment variable
'''
assert os.environ.get('WANDB_API_KEY'), "os.environ['WANDB_API_KEY'] is not set"


@wandb_mixin
def train_mnist(config):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_data_loaders()

    model = ConvNet()
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

    for _i in range(10):
        train(model, optimizer, train_loader, device=device)
        acc = test(model, test_loader, device=device)

        # When using WandbLogger, the metrics reported to tune are also logged in the W&B dashboard
        tune.report(mean_accuracy=acc)

        # @wandb_mixin enables logging custom metric using wandb.log()
        error_rate = 100 * (1 - acc)
        wandb.log({"error_rate": error_rate})


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--project", type=str, help="name of the wandb project", default="ray-example")
args = parser.parse_args()

analysis = tune.run(
    train_mnist,
    loggers=[WandbLogger],  # WandbLogger logs experiment configurations and metrics reported via tune.report() to W&B Dashboard
    resources_per_trial={'gpu': 1},
    config={
        # wandb dict accepts all arguments that can be passed in wandb.init()
        "wandb": {"project": args.project},
        # Hyperparameters
        "lr": tune.grid_search([0.0001, 0.001, 0.1]),
        "momentum": tune.grid_search([0.9, 0.99])
    })
