import sagemaker
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
from sagemaker.pytorch import PyTorch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import wandb
import argparse

parser = argparse.ArgumentParser(description='Run a sagemaker parameter sweep')
parser.add_argument('--max-jobs', type=int, default=3,
                    help='Maximum jobs')
parser.add_argument('--max-parallel-jobs', type=int, default=3,
                    help='Maximum parallel jobs')
parser.add_argument('--upload-data', action="store_true",
                    help='Upload the dataset (should be run the first time using this script)')


args = parser.parse_args()

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/pytorch-cifar10'

# If you are running this outside of a sagemaker notebook, you must set SAGEMAKER_ROLE
role = os.getenv('SAGEMAKER_ROLE') or sagemaker.get_execution_role()
wandb.sagemaker_auth(path="source")

if args.upload_data:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        'data', download=True, transform=transform)
    print("Uploading data...")
    inputs = sagemaker_session.upload_data(
        path='data', bucket=bucket, key_prefix=prefix)
else:
    inputs = "s3://{}/{}".format(bucket, prefix)

print("Using inputs: ", inputs)

estimator = PyTorch(entry_point="cifar10.py",
                    source_dir=os.getcwd() + "/source",
                    role=role,
                    framework_version='1.0.0.dev',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 50,
                        'momentum': 0.9
                    })

hyperparameter_ranges = {
    'lr': ContinuousParameter(0.0001, 0.01),
    'hidden_nodes': IntegerParameter(20, 100),
    'batch_size': CategoricalParameter([128, 256, 512]),
    'conv1_channels': CategoricalParameter([32, 64, 128]),
    'conv2_channels': CategoricalParameter([64, 128, 256, 512]),
}

objective_metric_name = 'average test accuracy'
objective_type = 'Maximize'
metric_definitions = [{'Name': 'average test accuracy',
                       'Regex': 'Test Accuracy: ([0-9\\.]+)'}]

tuner = HyperparameterTuner(estimator,
                            objective_metric_name,
                            hyperparameter_ranges,
                            metric_definitions,
                            max_jobs=args.max_jobs,
                            max_parallel_jobs=args.max_parallel_jobs,
                            objective_type=objective_type)

tuner.fit({'training': inputs})
