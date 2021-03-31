# Usage:
# python -m torch.distributed.launch \
# --nproc_per_node 2 \
# --nnodes 1 \
# --node_rank 0 \
# log-all.py \
# --epochs 10 \
# --batch 512 \
# --entity <ENTITY> \
# --project <PROJECT>

# IMPORTS
import wandb
import argparse
import numpy as np
from datetime import datetime

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms


def parse_args():
    """
    Parse arguments given to the scrip.

    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser()
    # Used for `distribution.launch`
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch",
        default=32,
        type=int,
        metavar="N",
        help="number of data samples in one batch",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="wandb entity",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="wandb project",
    )
    args = parser.parse_args()
    return args


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(args, run=None):
    """
    Train method for the model.

    Args:
        args: The parsed argument object
        run: The wandb run object
    """
    # Check to see if local_rank is 0
    args.is_master = args.local_rank == 0

    # set the device
    args.device = torch.device(args.local_rank)

    # initialize PyTorch distributed using environment variables
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(args.local_rank)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(42)

    # initialize model
    model = ConvNet()
    # send your model to GPU
    model = model.to(args.device)

    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )

    # watch gradients only for rank 0
    if run:
        run.watch(model)

    # Data loading code
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler,
    )

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        batch_loss = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())

            if (i + 1) % 100 == 0 and args.is_master:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, 10, i + 1, total_step, loss.item()
                    )
                )
            if run:
                run.log({"batch_loss": loss.item()})

        if run:
            run.log({"epoch": epoch, "loss": np.mean(batch_loss)})

    print("Training complete in: " + str(datetime.now() - start))


if __name__ == "__main__":
    # get args
    args = parse_args()
    # Initialize run
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",
    )
    # Train model with DDP
    train(args, run)
