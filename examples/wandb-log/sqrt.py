#!/usr/bin/env python
"""Calculates sqrt of argument using Babylonian algorithm
and tracks the answer with wandb.

Example command:
python sqrt.py 2
"""

import sys

import wandb

wandb.init(project="sqrt" + sys.argv[1])

wandb.config.steps = 10

x = float(sys.argv[1])

approx_sqrt = x / 2.0

for i in range(wandb.config.steps):
    approx_sqrt = (x / approx_sqrt + approx_sqrt) / 2
    wandb.log({"Approx. Sqrt" + sys.argv[1]: approx_sqrt})
