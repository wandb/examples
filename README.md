# Examples

## Getting started

1. Install wandb

```
pip install wandb
wandb login
```

2. Clone this repository

```
git clone https://github.com/wandb/examples
```

## Example projects

Example deep learning projects that use wandb's features.

### keras-cnn-fashion

Trains a fashion mnist classifier with a small CNN using the keras framework with the tensorflow backend.  Uses a simple integration with WandbKerasCallback.

```
cd examples/keras-cnn-fashion
wandb init
wandb run train.py
```

### keras-gan-mnist

Trains a GAN on mnist data using a CNN in the keras framework with the tensorflow backend.  This shows a more complicated integration with wandb using a custom callback on the generator model and the discriminator model.

```
cd examples/keras-gan-mnist
wandb init
wandb run train.py
```

### tf-cnn-fashion

Trains a fashion mnist classifier with a small CNN using the tensorflow framework.

```
cd examples/tf-cnn-fashion
wandb init
wandb run train.py
```

### pytorch-cnn-fashion

Trains a fashion mnist classifier with a small CNN using the pytorch framework.

```
cd examples/pytorch-cnn-fashion
wandb init
wandb run train.py
```
