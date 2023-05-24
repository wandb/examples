import math, argparse
from types import SimpleNamespace

import timm
import wandb
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from utils import save_model

PROJECT = 'fashion-launch'
ENTITY = "capecape"

defaults = SimpleNamespace(
    image_size=32,
    batch_size=128,
    learning_rate=1e-3,
    epochs=1,
    num_workers=0,
    model_name='resnest14d',
    num_classes=10,
    in_chans=1,
    device = "cuda:0" if torch.cuda.is_available() else "cpu",
    link_model=False,
)

def train(config):
    train_tfms = T.Compose([
        T.Resize(config.image_size, antialias=True), 
        T.RandomHorizontalFlip(),
        T.ToTensor()])

    train_ds = FashionMNIST(".", train=True, download=True, transform=train_tfms)

    train_dl = DataLoader(
        train_ds, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers)

    model = timm.create_model(
        config.model_name, 
        pretrained=False, 
        num_classes=config.num_classes, 
        in_chans=config.in_chans)

    model = model.to(device=config.device)

    run = wandb.init(project=PROJECT, entity=ENTITY, tags=["train"], config=config)

    # Log code wo W&B for launch
    run.log_code(".", include_fn=lambda path: path.endswith(".py"))

    # Copy your config 
    config = wandb.config

    # Get the data
    n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)

    # Make the loss and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr=config.learning_rate, 
                                                    total_steps=len(train_dl)*config.epochs)
    # Training
    example_ct = 0
    step_ct = 0
    for epoch in tqdm(range(config.epochs)):
        model.train()
        correct = 0.
        for step, (images, labels) in enumerate(tqdm(train_dl, leave=False)):
            images, labels = images.to(config.device), labels.to(config.device)

            outputs = model(images)
            train_loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            
            example_ct += len(images)
            
            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            train_accuracy = correct / example_ct

            metrics = {"train/train_loss": train_loss, 
                        "train/epoch": (step + 1 + (n_steps_per_epoch * epoch)) / n_steps_per_epoch, 
                        "train/example_ct": example_ct,
                        "train/accuracy": train_accuracy}
            
            if step + 1 < n_steps_per_epoch:
                # 🐝 Log train metrics to wandb 
                wandb.log(metrics)
                
            step_ct += 1

        wandb.log(metrics)
        print(f"{epoch} - Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.2f}")

    # Save trained model to disk and to W&B Artifacts
    save_model(model, model_name=config.model_name, 
               metadata=dict(config), link=config.link_model)
    run.finish()

def parse_args(default_cfg):
    "Override default hyper-parameters using argparse"
    parser = argparse.ArgumentParser(description="Process hyper-parameters")
    parser.add_argument("--batch_size", type=int, default=defaults.batch_size)
    parser.add_argument("--image_size", type=int, default=defaults.image_size)
    parser.add_argument("--learning_rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--num_workers", type=int, default=defaults.num_workers)
    parser.add_argument("--model_name", type=str, default=defaults.model_name)
    parser.add_argument("--num_classes", type=int, default=defaults.num_classes)
    parser.add_argument("--in_chans", type=int, default=defaults.in_chans)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--link_model", action="store_true", default=defaults.link_model)
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(default_cfg, k, v)

if __name__ == "__main__":
    parse_args(defaults)
    train(defaults)