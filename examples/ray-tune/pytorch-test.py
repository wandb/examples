import random

from ray import tune
import torch
import torch.optim as optim
from ray.tune.examples.mnist_pytorch import ConvNet, get_data_loaders, test, train
from ray.tune.integration.wandb import wandb_mixin
from ray.tune.integration.wandb import WandbLogger
import wandb


torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


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


analysis = tune.run(
    train_mnist,
    loggers=[WandbLogger],  # WandbLogger logs experiment configurations and metrics reported via tune.report() to W&B Dashboard

    resources_per_trial={'gpu': 1},
    config={
        # wandb dict accepts all arguments that can be passed in wandb.init()
        "wandb": {'project':'ray-example'},

        # Hyperparameters
        "lr": tune.grid_search([0.0001, 0.001, 0.1]),
        "momentum": tune.grid_search([0.9, 0.99])
    })

print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))
