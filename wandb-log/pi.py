#!/usr/bin/env python
"""Approximates pi with the Gregory-Leibniz series
terms up to argument and tracks the answer with wandb.

Example command:
python pi.py
"""

import sys

import wandb

wandb.init(project="pi")

wandb.config.steps = sys.argv[1]

approx_pi = 0.0

for i in range(wandb.config.steps):
    approx_pi += -((i % 2) * 2 - 1) * 4.0 / (2 * i + 1)
    wandb.log({"Approx. Pi": approx_pi})
