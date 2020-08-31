# Simpsons classification

*Classification of Simpsons characters*

## Introduction

This is a simple demo for classifying Simpsons characters with fast.ai and optimizing the neural network by monitoring and comparing runs with Weights & Biases.

Hyper-parameters are defined pseudo-randomly and every run is automatically logged onto [Weighs & Biases](https://www.wandb.com/) for easier analysis/interpretation of results and how to optimize the architecture.

You can also run [sweeps](https://docs.wandb.com/sweeps/) to optimize automatically hyper-parameters.

## Usage

1. Install dependencies through `requirements.txt`, `Pipfile` or manually (Pytorch, Fast.ai & Wandb)
2. Log in or sign up for an account -> `wandb login`
3. Run `python train.py`
4. Visualize and compare your runs through generated link

   ![alt text](imgs/results.png)

## Sweeps

1. Run `wandb sweep sweep.yaml`
2. Run `wandb agent <sweep_id>` where `<sweep_id>` is given by previous command.
3. Visualize and compare the sweep runs. See [my sweep](https://app.wandb.ai/borisd13/simpsons-fastai/sweeps/erraqo0l?workspace=user-borisd13).

## Results

After running the script a few times, you will be able to compare quickly a large combination of hyperparameters. As an example, you can refer to [my runs](https://app.wandb.ai/borisd13/simpsons-fastai/reports?view=borisd13%2Fsimpsons).

![alt text](imgs/graphs.png)

Feel free to modify the script and define your own hyperparameters.
