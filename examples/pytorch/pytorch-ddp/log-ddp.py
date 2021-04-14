# Usage:
# python -m torch.distributed.launch \
# --nproc_per_node <NUM_GPUS> \
# --nnodes 1 \
# --node_rank 0 \
# log-ddp.py \
# --log_all \
# --epochs 10 \
# --batch 512 \
# --entity <ENTITY> \
# --project <PROJECT>

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
import wandb

import utils


def train(args, run=None):
    """
    Train method for the model.

    Args:
        args: The parsed argument object
        run: If logging, the wandb run object, otherwise None
    """
    # Check to see if local_rank is 0
    is_master = args.local_rank == 0
    do_log = run is not None

    # set the device
    total_devices = torch.cuda.device_count()
    device = torch.device(args.local_rank % total_devices)

    # initialize PyTorch distributed using environment variables
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(device)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(42)

    # initialize model -- no changes from normal training
    model = utils.ConvNet()
    # send your model to GPU
    model = model.to(device)

    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    # watch gradients only for rank 0
    if is_master:
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

    total_step = len(train_loader)
    for epoch in range(args.epochs):
        batch_loss = []
        for ii, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = float(loss)
            batch_loss.append(loss)

            if (ii + 1) % 100 == 0 and is_master:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, 10, ii + 1, total_step, loss
                    )
                )
            if do_log:
                run.log({"batch_loss": loss})

        if do_log:
            run.log({"epoch": epoch, "loss": np.mean(batch_loss)})


def setup_run(args):
    if args.log_all:
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group="DDP",
        )
    else:
        if args.local_rank == 0:
            run = wandb.init(
                entity=args.entity,
                project=args.project,
            )
        else:
            run = None

    return run


if __name__ == "__main__":
    # get args
    args = utils.parse_args()

    # wandb.init a run if logging, otherwise return None
    run = setup_run(args)

    train(args, run)
