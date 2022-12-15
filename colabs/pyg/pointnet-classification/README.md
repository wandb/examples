# 🔥🔥 Point Cloud Classification using PyTorch Geometric and Weights & Biases 🪄🐝

This example demonstrates an implementation of the [PointeNet++](https://arxiv.org/pdf/1706.02413.pdf) architecture implemented using PyTorch Geometric and experiment tracked and visualized using [Weights & Biases](https://wandb.ai/site).

[![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](http://wandb.me/pointnet2-classification)

## Notebooks

|Task|Notebook|
|---|---|
|Exploring Datasets|[![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-modelnet-eda)|
|Comparing Sampling Strategies|[![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-sampling)|
|Train PointNet++|[![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-pointnet2-train)|
|Tune Hyperparameters using Weights & Biases Sweep|[![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/pyg-pointnet2-sweep)|

## Running Sweep using WandB CLI

A Hyperparameter Sweep can be run using the aforementioned notebook, however, it has its limitations. Since a notebook runs multiple experiments under a single process, it might result the GPU to run out of memeory after a few runs. In that case, it is advisable to run the sweep using the WandB CLI.

1. First, we define a [YAML file](./sweep_config.yaml) that stores the sweep configurations. You can refer to [this guide](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) to know more on how to define sweep configurations.

2. Next, we create a script [`train.py`](./train.py) that runs our training and validation loop instrumented with Weights & Biases logging.

3. We then use the Weights & Biases CLI to initialize a sweep using `wandb sweep --project <project-name> sweep_config.yaml`. This would initialize a **Sweep Controller** that enables Weights & Biases to manage sweeps on the cloud (standard), locally (local) across one or more machines. After a run completes, the sweep controller will issue a new set of instructions describing a new run to execute.

4. Now we start a the Sweep on one or more **agents** on one or more machines using `wandb agent sweep_id`, where we use the sweep id of the sweep controller. The Sweep agent performs the runs by picking up the instructions issued by the sweep controller. 