# Run your first sweep from an existing W&B Project

If you have an existing W&B project, it’s easy to start optimizing your models with hyperparameter sweeps. I’ll work through the steps with a working example— you can open my [W&B Dashboard](https://app.wandb.ai/carey/pytorch-cnn-fashion). I'm using the code from this example repo, which trains a PyTorch convolutional neural network to classify images from the [Fashion MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).


1. I open my project page. Here are a couple of runs I’ve done already.
![](https://i.imgur.com/aS019gx.png)

2. I open the sweep tab and click “Create sweep” in the upper right corner.
![](https://i.imgur.com/q3o0EGT.png)

3. These steps take me through running my first sweep. To make sure I have the latest version of wandb I run pip install --upgrade wandb first.
![](https://i.imgur.com/CWTNBiV.png)

4. The auto-generated config guesses values to sweep over based on the runs I’ve done already. In the “Parameters” tab, I remove channels_one and channels_two from my sweep config. I don’t want to sweep over those hyperparameters. Once I’m happy with the ranges of parameters to sweep over, I download the file.
![](https://i.imgur.com/gqcTBct.png)

5. I move the generated config file to my training script repo.
![](https://i.imgur.com/2F8ADud.png)

6. I run wandb sweep sweep.yaml to start a sweep on the W&B server. This is a centralized service sends out the next set of hyperparameters to agents that I run on my own machines.
![](https://i.imgur.com/t4nJ6yO.png)

7. Next I launch an agent locally. I can launch dozens of agents in parallel if I want to distribute the work and finish the sweep more quickly. The agent will print out the set of parameters it’s trying next.
![](https://i.imgur.com/GUGn4Oo.png)


That’s it! Now I’m running a sweep. Here’s what my dashboard looks like as the sweep begins to explore the space of hyperparameters.
![](https://i.imgur.com/gK42OOB.png)

