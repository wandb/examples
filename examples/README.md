<img src="https://i.imgur.com/gb6B4ig.png" width="400" alt="Weights & Biases" />

<div><img /></div>

<img src="https://i.imgur.com/uEtWSEb.png" width="650" alt="Weights & Biases" />

# 👩🏽‍🏫 Examples

We've created some simple examples that show how to use `wandb` to track experiments with different frameworks.

They're designed to be easy to use and regularly checked for errors.

If you run into any trouble, let us know by opening up [an issue](https://github.com/wandb/examples/issues)!

The examples are primarily organized by framework.

## 🚀 Getting started

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

## 🔥 [pytorch](https://github.com/wandb/examples/tree/master/examples/pytorch)

### 👋 [pytorch-intro](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-intro)

Demonstrates how to integrate PyTorch with Weights & Biases in a Jupyter notebook.

```
cd examples/examples/pytorch/pytorch-intro
jupyter notebook intro.ipynb
```

Also check out the [colab version](https://tiny.cc/wb-pytorch-colab).

### 👡 [pytorch-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-fashion)

Trains a classifier on black-and-white images of clothing items
from the Fashion MNIST dataset with a small CNN using the PyTorch  framework.
Includes instructions for how to use the [Sweeps](https://docs.wandb.com/sweeps)
tool to run hyperparameter optimization.

```
cd examples/examples/pytorch/pytorch-cnn-fashion
python main.py
```

### 9️⃣ [pytorch-cnn-mnist](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cnn-mnist)

Trains a classifier on hand-written digits from the MNIST dataset
with a small CNN using the PyTorch framework.

```
cd examples/examples/pytorch/pytorch-cnn-fashion
python train.py
```

### 🌿 [pytorch-cifar10-sagemaker](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-cifar10-sagemaker)

Uses the [AWS SageMaker](https://aws.amazon.com/sagemaker/)
API to launch a hyperparameter optimization [Sweep](https://docs.wandb.com/sweeps).

See the README for instructions.

## 🥕 [keras](https://github.com/wandb/examples/tree/master/examples/keras)

### 👋 [keras-tensorflow2](https://github.com/wandb/examples/tree/master/examples/keras/keras-tensorflow2)

This tutorial is adapts Francois Chollet's [brilliant introduction to TensorFlow 2.0](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO)
to include W&B callbacks.

```
cd examples/examples/keras/keras-cnn-fashion
jupyter notebook "TensorFlow_2_0_+_Keras_Crash_Course_+_W&B.ipynb"
```

### 👢 [keras-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)

Trains a fashion-mnist classifier with a small CNN using the keras framework with the tensorflow backend.  Uses a simple integration with `WandbKerasCallback`.
Includes instructions for how to use the [Sweeps](https://docs.wandb.com/sweeps)
tool to run hyperparameter optimization.

```
cd examples/examples/keras/keras-cnn-fashion
python train.py
```

### 🏞️ [keras-cnn-nature](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-nature)

Multiple examples which train and finetune CNNs in Keras with the goal of exploring curriculum learning
on the [iNaturalist 2017 dataset](https://github.com/visipedia/inat_comp/tree/master/2017)
of animal and plant images plus phylogenetic annotations.

See the [README](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-nature)
for more details.


### ✍️ [keras-gan-mnist](https://github.com/wandb/examples/tree/master/examples/keras/keras-gan-mnist)

Trains a GAN on mnist data using a CNN in the keras framework with the tensorflow backend.  This shows a more complicated integration with wandb using a custom callback on the generator model and the discriminator model.

```
cd examples/examples/keras/keras-gan-mnist
python train.py
```

## 🌊 [tensorflow](https://github.com/wandb/examples/tree/master/examples/tensorflow)

### 👠 [tf-cnn-fashion](https://github.com/wandb/examples/tree/master/examples/tensorflow/tf-cnn-fashion)

Trains a classifier on black-and-white images from the Fashion MNIST dataset
with a small CNN using the TensorFlow framework.

```
cd examples/examples/tensorflow/tf-cnn-fashion
python train.py
```

## 💨 [fastai](https://github.com/wandb/examples/tree/master/examples/fastai)

### ↪️ [fastai-unet-segmentation](https://github.com/wandb/examples/tree/master/examples/fastai/fastai-unet-segmentation)

Trains a semantic segmentation model on a dataset from the video game [The Witness](https://en.wikipedia.org/wiki/The_Witness_(2016_video_game))
```
cd examples/examples/fastai/fastai-unet-segmentation
pip install -r requirements.txt
python train.py
```

### 🍜 [fastai-food101](https://github.com/wandb/examples/tree/master/examples/fastai/fastai-food101)
Trains a 121 layer DenseNet on the [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) using the 1cycle learning rate policy, mixed precision training, mixup data augmentation, and progressive resizing.

```
cd examples/examples/fastai/fastai-food101
pip install -r requirements.txt
python train.py
```

## 🌳 [boosting-algorithms](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms)

### 🦶 [xgboost-dermatology](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms/xgboost-dermatology)

Trains an ensemble of decision trees on the [Dermatology Dataset](https://archive.ics.uci.edu/ml/datasets/dermatology)
with [XGBoost](https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/).

```
cd examples/examples/boosting-algorithms/xgboost-dermatology
python train.py
```

### 📈 [lightgbm-regression](https://github.com/wandb/examples/tree/master/examples/boosting-algorithms/lightgbm-regression)

Trains an ensemble of decision trees on
[Microsoft's regression example](https://github.com/microsoft/LightGBM/tree/master/examples/regression)
with [LightGBM](https://lightgbm.readthedocs.io/en/latest/).

## 👩‍🔬 [scikit](https://github.com/wandb/examples/tree/master/examples/scikit)

### 🌼 [scikit-iris](https://github.com/wandb/examples/tree/master/examples/scikit/scikit-iris)

Trains an SVM on the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris)
using scikit-learn and logs results to W&B.

```
cd examples/examples/scikit/scikit-iris
python train.py
```

### 🏡 [scikit-regression](https://github.com/wandb/examples/tree/master/examples/scikit/scikit-regression)

Trains a ridge regression model on the [Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html)
using scikit-learn and logs the results to W&B.

```
cd examples/examples/scikit/scikit-regression
python train.py
```
