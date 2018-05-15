#!/usr/bin/env python
"""validation process."""

import argparse
import random
import time

import wandb


parser = argparse.ArgumentParser()
parser.description = 'Train an example model'
parser.add_argument('--group_id', type=str, default=None)
parser.add_argument('--worker_index', type=int, default=0)
parser.add_argument('--epoch', type=int, default=0)

# Just a made up validation parameter
parser.add_argument('--phase_shift', type=int, default=14)


def loss_curve(step):
    result = 10 / (step + 1)
    noise = (random.random() - 0.5) * 0.5 * result
    return result + noise


def accuracy(loss):
    return (100 - loss) / 100.0


def main():
    args = parser.parse_args()
    if args.group_id == None:
        print('Please pass --group_id')
        return
    wandb.init()
    wandb.config.update(args)
    wandb.config.job_type = 'validation'

    print('Validation Epoch', args.epoch)
    loss = loss_curve(args.epoch)
    acc = accuracy(loss)
    # Same key for accuracy as train.py, so we can put them on the same plot.
    # In a future version we will allow showing different keys within a group on the same
    # plot.
    wandb.log({'acc': acc - 0.05, 'epoch': args.epoch})


if __name__ == '__main__':
    main()
