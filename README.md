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

Trains a fashion mnist classifier with a small CNN using the keras framework with a tensorflow backend.

```
cd examples/keras-cnn-fashion
wandb run train.py
```

### tf-cnn-fashion

Trains a fashion mnist classifier with a small CNN using the tensorflow framework.

```
cd examples/tf-cnn-fashion
wandb run train.py
```
