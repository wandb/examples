'''
dataset_generator.py

This simple script represents a workload that generates and registers
a dataset for a particular model use case.

Author: Tim Sweeney
@wandb
'''

import wandb
import util
import argparse

project             = "model_registry_example"
model_use_case_id   = "mnist"
job_type            = "dataset_builder"

# First, we launch a run which registers this workload with W&B.
parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=int, default=100, help='number of training examples')
run = wandb.init(project=project, job_type=job_type, config=parser.parse_args())

# Next, we generate the raw data. For simplicity, we're using MNIST.
(x_train, y_train), (x_eval, y_eval) = util.generate_raw_data(run.config.train_size)

# Finally, we publish this dataset to W&B. The utility method generates a W&B Artifact 
# that contains both training and evaluation data, which can be visualized in the W&B UI.
util.publish_dataset_to_wb(x_train, y_train, x_eval, y_eval, model_use_case_id)