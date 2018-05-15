#!/usr/bin/env python
"""trainer process"""

import argparse
import random
import time

import wandb


parser = argparse.ArgumentParser()
parser.description = 'Train an example model'
parser.add_argument('--group_id', type=str, default=None)
parser.add_argument('--worker_index', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.9)
parser.add_argument('--momentum', type=float, default=0.8)


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
    wandb.config.job_type = 'training'

    print('Training worker #%s' % args.worker_index)

    for i in range(50):
        print('Training Epoch', i)
        loss = loss_curve(i)
        acc = accuracy(loss)

        # training metrics, but don't commit the step.
        # Same key for accuracy as validate.py, so we can put them on the same plot.
        # In a future version we will allow showing different keys within a group on the same
        # plot.
        wandb.log({'loss': loss, 'acc': acc, 'epoch': i})

        time.sleep(0.25)


if __name__ == '__main__':
    main()
