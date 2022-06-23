# PyTorch Cifar10 SageMaker Sweep

This example uses [AWS SageMaker](https://aws.amazon.com/sagemaker/) to launch a hyperparameter sweep. `train.py` uses the AWS SageMaker api to launch a sweep.

If you run this script outside of a SageMaker notebook instance, you'll need to download the aws cli and run aws configure and then set the **SAGEMAKER_ROLE** environment variable to an AWS role name that has access to SageMaker, you can find your roles [here](https://console.aws.amazon.com/iam/home?#/roles).

To authenticate with W&B, we call `wandb.sagemaker_auth(path="source")` in `train.py`. `wandb.sagemaker_auth(path="source")` will look for W&B credentials in the current environment and generate a `secrets.env` which will be passed to Sagemaker. Calling `wandb login MY_API_KEY` before running the script on the machine you're running from will ensure credentials get passed. Similarly you can use the `WANDB_API_KEY` environment variable which needs to be set before usage of `wandb` in the script.
