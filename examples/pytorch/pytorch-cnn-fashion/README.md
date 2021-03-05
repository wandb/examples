# Hyperparameter Optimization in 5 minutes
It’s easy to start optimizing your models with hyperparameter sweeps. I’ll walk through the steps in this repo with a working example — you can open my [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion). I'm using the code from this example repo, which trains a PyTorch convolutional neural network to classify images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## 1. Create a project
Run your first baseline run manually to check that W&B logging is working properly. You'll download this simple example model, train it for a few minutes, and see the example appear in the web dashboard.

- Clone this repository with `git clone https://github.com/wandb/examples.git`.
- Navigate to this folder.
- Run an individual run manually `python train.py`.

[View an example project page →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

## 2. Create a sweep
From your project page, open the Sweep tab in the sidebar and click "Create Sweep".
![](https://i.imgur.com/q3o0EGT.png)

The auto-generated config guesses values to sweep over based on the runs you've done already. Edit the config to specify what ranges of hyperparameters you want to try. When you launch the sweep, it starts a new process on our hosted W&B sweep server. This centralized service coordinates the agents— your machines that are running the training jobs.
![](https://i.imgur.com/gucKbHO.png)

## 3. Launch agents
Next, launch an agent locally. You can launch dozens of agents on different machines in parallel if you want to distribute the work and finish the sweep more quickly. The agent will print out the set of parameters it’s trying next.
![](https://i.imgur.com/6pWCOym.png)

That’s it! Now you're running a sweep. Here’s what the dashboard looks like as my example sweep gets started.
[View an example project page →](https://app.wandb.ai/carey/pytorch-cnn-fashion)

![](https://i.imgur.com/gK42OOB.png)

