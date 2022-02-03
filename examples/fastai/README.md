# Fastai Integration

We can propose very simple `wandb` integration in `fastai` with the `WandbCallback`. A good starting point is this [colab](http://wandb.me/fastai_demo) and the [docs](https://docs.wandb.ai/guides/integrations/fastai)

Use Fastai + W&B to:

- Log and compare runs and hyperparameters
- Keep track of code, models and datasets
- Automatically log prediction samples to visualize during training
- Make custom graphs and reports with data from your runs
- Launch and scale hyperparameter search on your own compute, orchestrated by W&B
- Collaborate in a transparent way, with traceability and reproducibility

## Install
You will need `fastai` and the `wandb` python packages. We really recommend `conda` for fastai installs as it will grab the latest fastai and pytorch and install cleanly in your conda environment:

```bash
conda install -c fastchan fastai
pip install wandb
```

Some examples require extra installation steps.

## Quick Start

In any fastai training pipeline, you can start uisng wandb straight away with only this minimal code:

```python
import wandb
from fastai.callback.wandb import *

# start logging a wandb run
wandb.init(project='my_project')

# To log only during one training phase
learn.fit(..., cbs=WandbCallback())

# To log continuously for all training phases
learn = learner(..., cbs=WandbCallback())
```

## [basics](basics)

A bunch of scripts for simple `fastai` core tasks with `wandb` logging enabled.

## [aws-segmentation](aws-segmentation) 

A more complex example,  we retrieve data from an S3 bucket and then train a segmentation model on top of it.

## [fastai-v1](fasta-v1-examples)

Examples using the old `fastai` v1 library. They are not maintained anymore.