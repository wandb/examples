#!/usr/bin/env python
"""eval process."""

import argparse
import random
import time

import wandb


parser = argparse.ArgumentParser()
parser.description = 'Train an example model'
parser.add_argument('--group_id', type=str, default=None)
parser.add_argument('--epoch', type=int, default=0)

# Just a made up eval parameter
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
    wandb.init(group=args.group_id, job_type = 'eval')
    wandb.config.update({'phase_shift': args.phase_shift})
    wandb.config.job_type = 'eval'

    print('Eval Epoch', args.epoch)
    loss = loss_curve(args.epoch)
    acc = accuracy(loss)

    # Keys with a '<section>/' prefix will be separated into different
    # plot sections. Since train and eval both log a 'loss' key, train
    # and eval loss results will show up on the same plot by default.
    # But they each log their own prefixed 'acc', which will get different
    # plots in different sections.
    wandb.log({'epoch': args.epoch, 'loss': loss, 'eval/acc': acc - 0.05})


if __name__ == '__main__':
    main()
