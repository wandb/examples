# Examples

These are simple examples that show how to use wandb to track experiments with different frameworks.  They should be easy to use.

If you have any questions, please don't hesitate to ask in our [Slack community](http://bit.ly/wandb-forum).

## Getting started

1. Install wandb

```
pip install wandb
```

2. Clone this repository

```
git clone https://github.com/wandb/examples
```

3. Create a free account (optional)

```
wandb login
```

## Example projects

Example deep learning projects that use wandb's features.

### keras-cnn-fashion

Trains a fashion mnist classifier with a small CNN using the keras framework with the tensorflow backend.  Uses a simple integration with WandbKerasCallback.

```
cd examples/keras-cnn-fashion
python train.py
```

### keras-cnn-nature

#### train_small_cnn.py

Trains a small CNN on images of plants and animals using Keras. Highly configurable through command line flags: run with ``-h`` to see all the options. 
The ``data_tools`` directory contains a helper script to generate more manageable training datasets from the full 186GB iNaturalist 2017 dataset. A 12K subset of the data can be downloaded by [clicking this link](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). For more context on this example, see this [blog post](https://www.wandb.com/blog/better-paths-through-idea-space) and this W&B [report](https://app.wandb.ai/stacey/keras_finetune/reports?view=stacey%2FiNaturalist), which explores various settings and hyperparameters. 

```
cd examples/keras-cnn-nature
python train_small_cnn.py
```

#### finetune_experiments.py
 
 Enables two kinds of finetuning experiments:
 * loading various pretrained base CNNs (Xception,ResNet, InceptionResNetV2, InceptionV3), pretraining for some epochs, freezing some of the layers of the resulting network, then continuing to finetune the rest of the layers 
 * loading a small CNN, pretraining on general labels (in this case, predicting one of 5 biological classes) for a certain number of epochs, then finetuning on specific labels (predicting one of 25 biological species)

 Highly configurable with commandline flags: run with ``-h`` to see all the options. 
```
cd examples/keras-cnn-nature
python finetune_experiments.py
``` 

### keras-gan-mnist

Trains a GAN on mnist data using a CNN in the keras framework with the tensorflow backend.  This shows a more complicated integration with wandb using a custom callback on the generator model and the discriminator model.

```
cd examples/keras-gan-mnist
python train.py
```

### tf-cnn-fashion

Trains a fashion mnist classifier with a small CNN using the tensorflow framework.

```
cd examples/tf-cnn-fashion
python train.py
```

### pytorch-cnn-fashion

Trains a fashion mnist classifier with a small CNN using the pytorch framework.

```
cd examples/pytorch-cnn-fashion
python train.py
```
### fastai-food101
Trains a 121 layer DenseNet on the [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) using the 1cycle learning rate policy, mixed precision training, mixup data augmentation, and progressive resizing.
```
cd fastai/food-101
pip install -r requirements.txt
python train.py
```

### fastai-unet-segmentation

Trains a semantic segmentation on a dataset from the game "witness"
```
cd fastai/unet-segmentation
pip install -r requirements.txt
python train.py
```

### scikit-iris

Trains an SVM on the Iris dataset using scikit-learn

```
cd examples/scikit-iris
python train.py
```

### xgboost-dermatology

Trains a gradient boosted forest on the dermatology dataset

```
cd examples/xgboost-dermatology
python train.py
```

### numpy-boston

Trains a perceptron on the Boston real estate dataset using numpy

```
cd examples/numpy-boston
python train.py
```
