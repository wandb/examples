# Example Keras CNN on Fashion MNIST

## Setup
- pip install --upgrade -r requirements_gpu.txt  # or requirements.txt if using a CPU only
- wandb login

## Execute script
- python train.py

## Create Sweep
If you want to explore hyperparameters you can run a sweep over a range of values.

Edit the example sweep configuration files to adjust parameter bounds.

Create the sweep:
- wandb sweep sweep-bayes-hyperband.yaml

If using Ray/Tune programmatic sweep definition, create the sweep yaml file then create the sweep:
- python sweep-tune-hyperopt.py
- wandb sweep sweep-tune-hyperopt.yaml

## Start Sweep
- wandb agent SWEEP_ID  # SWEEP_ID is from the create sweep command above

## Sweep Results
- [Bayesian HyperBand Sweep](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf)
- [Tune Hyperopt Sweep](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e)
