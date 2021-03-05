# Example Keras CNN on Fashion MNIST

This example trains a classifier on black-and-white images of clothing
with a small CNN using the Keras framework with the TensorFlow backend.

## Setup
- `pip install --upgrade -r requirements_gpu.txt  # or requirements.txt if using a CPU only`
- `wandb login`

## Execute script
- `python train.py`

## Execute with Resumable Runs
Stuff happens: programs crash, nodes get pre-empted, lightning strikes the data center.
With the `resume` feature, you can execute fault-tolerant experiments
and keep going even after things go belly up.

To execute a Run that can be continued even if it fails,
use the command
- `python train.py --resume`

Try interrupting the Run once it has finished an epoch,
then re-executing the command.

## Create Sweep
If you want to explore hyperparameters you can run a [Sweep](https://docs.wandb.com/sweeps) over a range of values.

Edit the example sweep configuration files to adjust parameter bounds.

Create the sweep:
- `wandb sweep sweep-bayes-hyperband.yaml`

If using Ray/Tune programmatic sweep definition, create the [Sweep yaml file](https://docs.wandb.com/sweeps/configuration) then create the Sweep:
- `python sweep-tune-hyperopt.py`
- `wandb sweep sweep-tune-hyperopt.yaml`

## Start Sweep
- `wandb agent SWEEP_ID  # SWEEP_ID is returned by the wandb sweep command above`

## Sweep Results
- [Bayesian HyperBand Sweep](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf)
- [Tune Hyperopt Sweep](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e)
