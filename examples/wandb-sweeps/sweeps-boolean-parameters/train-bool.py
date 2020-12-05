import argparse

import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", default=None, type=str)
parser.add_argument("--flag1", default=False, type=lambda s: s.lower() == 'true')
parser.add_argument("--flag2", default=True, type=lambda s: s.lower() == 'true')

args = parser.parse_args()

run = wandb.init(config=args)
run.log(dict(metric=1.0))
