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

### keras-cnn-nature

#### train_small_cnn.py

Trains a small CNN on images of plants and animals using Keras. Highly configurable through command line flags: run with ``-h`` to see all the options. 
The ``data_tools`` directory contains a helper script to generate more manageable training datasets from the full 186GB iNaturalist 2017 dataset. A 12K subset of the data can be downloaded by [clicking this link](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). For more context on this example, see this [blog post](https://www.wandb.com/blog/better-paths-through-idea-space) and this W&B [report](https://app.wandb.ai/stacey/keras_finetune/reports?view=stacey%2FiNaturalist), which explores various settings and hyperparameters. 

```
cd examples/keras-cnn-nature
wandb init
python train_small_cnn.py
```

#### finetune_experiments.py
 
 Enables two kinds of finetuning experiments:
 * loading various pretrained base CNNs (Xception,ResNet, InceptionResNetV2, InceptionV3), pretraining for some epochs, freezing some of the layers of the resulting network, then continuing to finetune the rest of the layers 
 * loading a small CNN, pretraining on general labels (in this case, predicting one of 5 biological classes) for a certain number of epochs, then finetuning on specific labels (predicting one of 25 biological species)

 Highly configurable with commandline flags: run with ``-h`` to see all the options. 
```
cd examples/keras-cnn-nature
wandb init
python finetune_experiments.py
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
