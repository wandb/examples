#!/usr/bin/env python
import wandb
from train_lib import train

if __name__ == '__main__':
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'layers': {
                'values': [32, 64, 96, 128, 256]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train)
