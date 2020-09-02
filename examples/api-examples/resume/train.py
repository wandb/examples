#!/usr/bin/env python
"""An example of resuming a previously started run.

This example takes 100 seconds to run by default, to show that system
metrics, which are logged every 30s, are also resumed.

Example commands:
WANDB_RUN_ID=myrun1 python train.py

# and then to resume it:
WANDB_RESUME=allow WANDB_RUN_ID=myrun1 python train.py --start_epoch=10
"""

import argparse
import random
import time

import wandb


parser = argparse.ArgumentParser()
parser.description = 'Train an example model'
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.01)

def loss_curve(step):
    result = 10 / (step + 1)
    noise = (random.random() - 0.5) * 0.5 * result
    return result + noise

def accuracy(loss):
    return (100 - loss) / 100.0

def main():
    args = parser.parse_args()
    wandb.init()
    wandb.config.update(args)

    for i in range(args.start_epoch, args.start_epoch + args.num_epochs):
        print('Step', i)
        loss = loss_curve(i)
        acc = accuracy(loss)

        wandb.log({'loss': loss, 'acc': acc})

        time.sleep(10)


if __name__ == '__main__':
    main()
