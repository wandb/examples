#!/usr/bin/env python
"""An example of sending validation metrics every N time steps.

This example doesn't do any actual machine learning
"""

import argparse
import random
import time

import wandb


parser = argparse.ArgumentParser()
parser.description = 'Train an example model'
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--validate_every', type=int, default=10)
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

    for i in range(args.max_epochs):
        print('Step', i)
        loss = loss_curve(i)
        acc = accuracy(loss)

        # training metrics, but don't commit the step.
        wandb.log({'loss': loss, 'acc': acc}, commit=False)

        # validation metrics, which could be reported in another part of the code
        if i % args.validate_every == 0:
            wandb.log({'val_acc': acc - 0.05}, commit=False)

        # commit the current step's values
        wandb.log()

        time.sleep(0.25)


if __name__ == '__main__':
    main()
