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
    wandb.init(group=args.group_id, job_type='train')
    wandb.config.update(args)
    wandb.config.job_type = 'training'

    print('Training worker #%s' % args.worker_index)

    for i in range(50):
        print('Training Epoch', i)
        loss = loss_curve(i)
        acc = accuracy(loss)

        # Keys with a '<section>/' prefix will be separated into different
        # plot sections. Since train and eval both log a 'loss' key, train
        # and eval loss results will show up on the same plot by default.
        # But they each log their own prefixed 'acc', which will get different
        # plots in different sections.
        wandb.log({'epoch': i, 'loss': loss, 'train/acc': acc})

        time.sleep(0.25)


if __name__ == '__main__':
    main()
