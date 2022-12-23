# ProtoBERT
Our goal here is to show how to train a protein language model using Google TPU, Jax/Flax and Weights & Biases. The code is based on [JAX/Flax Community Week](https://github.com/huggingface/transformers/blob/main/examples/research_projects/jax-projects/README.md), make sure to give it a star!

This repo accompanies our W&B report: [Training a Protein Language Model on a TPU](https://wandb.ai/darek/protobert/reports/Training-a-Protein-Language-Model-on-a-TPU--VmlldzoyNzg2NTgz).

## Instructions
1. We'll be using a TPU-VM in our project. To get access to a TPU-VM, follow [this guide](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).
2. Next, you'll need to install all the necessary libraries. You can use `environment_setup.sh` script for this. 
3. Login to W&B: `wandb login`
4. Let's define the model architecture we will use in our training. Watch out specifically for the vocabulary size that should match the tokenizer: `python create_config.py`
5. Now you should be able to train a protein language model. Specify the training parameters and use the `run.py` script. Example list of hyperparameters we have used is documented in `commands.txt`.
6. If you'd like to experiment with the tokenizer, you can use the `create_tokenizer.ipynb` notebook. Again, make sure the vocab size is aligned between your model and tokenizer!


