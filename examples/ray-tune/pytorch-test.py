import wandb

import torch
import torch.optim as optim

from ray import tune
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test, train
from ray.tune.integration.wandb import wandb_mixin
from ray.air.callbacks.wandb import WandbLoggerCallback

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

        # When using WandbLoggerCallback, the metrics reported to tune are also logged in the W&B dashboard
        print("DEBUG: tune.report run_id={} mean_accuracy={}".format(wandb.run.id, acc))
        tune.report(mean_accuracy=acc)

        # @wandb_mixin enables logging custom metric using wandb.log()
        error_rate = 100 * (1 - acc)
        wandb.log({"error_rate": error_rate})

if __name__=="__main__":

    # Log into wandb account
    wandb.login()
    analysis = tune.run(
        train_mnist,
        callbacks=[WandbLoggerCallback(
            project="ray-example", # you can pass your project name as an arg via callback or via config as well
            log_config=True)],  # WandbLoggerCallback logs experiment configurations and metrics reported via tune.report() to W&B Dashboard
        resources_per_trial={'gpu': 1},
        config={
            # wandb dict accepts all arguments that can be passed in wandb.init()
            "wandb": {"project": "ray-example"},
            # Hyperparameters
            "lr": tune.grid_search([0.0001, 0.001, 0.1]),
            "momentum": tune.grid_search([0.9, 0.99])
        })