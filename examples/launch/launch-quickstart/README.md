# Weights & Biases Launch Quickstart

In order to follow this tutorial, you will need a machine with `docker` and `wandb`
installed. You will also want to make sure you have authenticated your `wandb`
client with `wandb login` or by settings the `WANDB_API_KEY` environment variable.

The `train.py` script in this directory will train a simple neural network to
classify handwritten digits from the MNIST dataset. It will also log training 
metrics and evaluation results to Weights & Biases.

From within this directory (`examples/examples/launch/launch-quickstart`) run

```
docker build . -t launch-quickstart
```

to build the training script into an image named `launch-quickstart`. We can 
use the `wandb launch` command to run a container from that image with our
`wandb` credentials automatically passed in from our host environment.

Running `wandb launch -d launch-quickstart -p mnist` will start the job and send
metrics to a new project called `mnist` in your Weights & Biases account.

To learn how you use the same tool to launch jobs into Kubernetes or Amazon 
SageMaker, check out more of [our docs](https://docs.wandb.ai/guides/launch)!