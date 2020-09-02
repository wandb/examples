#!/bin/bash

python -m wandb.kubeflow.arena submit tf --name=exp1 \
    --wandb-project=distributed-mnist \
    --gpus=1 \
    --workers=2 \
    --workerImage=ufoym/deepo:keras \
    --syncMode=git \
    --syncSource=https://github.com/wandb/examples.git \
    --ps=1 \
    --psImage=ufoym/deepo:keras-cpu \
    --tensorboard \
    "pip install wandb[kubeflow] pillow && python code/examples/tf-distributed-mnist/train.py"
