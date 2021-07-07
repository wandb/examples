# K-Fold Cross Validation with W&B Sweeps

Here's a quick example of running k-fold cross validation with a Weights & Biases sweep. In this example, we simulate the model training with just a random number generator producing accuracy values. You can set the `num_folds` variable to change how many folds it runs through.

To run this sweep:
```
wandb sweep sweep-cross-validation.yaml
wandb agent SWEEP_ID
```


Example Sweep URL:
https://app.wandb.ai/jeffr/examples-sweeps-cross-validation/sweeps/vp0fsvku

Example Sweep Group URL:
https://app.wandb.ai/jeffr/examples-sweeps-cross-validation/groups/vp0fsvku

##### Troubleshooting

Please refer to the [Sweeps Quickstart Guide](https://docs.wandb.ai/guides/sweeps/quickstart) for detailed instructions.

- If you get permissions error, make sure you run `wandb init`.
