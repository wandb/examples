# Finetuning CNNs in Keras

These examples train and finetune CNNs in Keras with the goal of exploring curriculum learning: the effect of training data order (from simple to more complex, or general to more specific material) on the performance or accuracy of the network. The dataset is iNaturalist 2017, which contains >675K photos of >5K different species of life. Species are also annotated with higher-level taxonomy like biological class, phylum, and/or kingdom.

## Data acquisition and tools

The full iNaturalist 2017 dataset, [available here](https://github.com/visipedia/inat_comp/tree/master/2017), is 186GB. You can download a more manageable random 12K subset by [clicking this link](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). You can also download a small dataset (10GB) for curriculum learning (one copy partitioned into 5 biological classes and another into 25 species) by [clicking this link](https://storage.googleapis.com/wandb_datasets/curr_learn_data.zip). The ``data_tools`` directory contains a helper script to generate other subsets for training via symlinks.

## Training a small CNN

This example trains a small CNN on images of plants and animals using Keras. Highly configurable through command line flags: run with ``-h`` to see all the options. For more context on this example, see this [blog post](https://www.wandb.com/blog/better-paths-through-idea-space) and this W&B [report](https://app.wandb.ai/stacey/keras_finetune/reports?view=stacey%2FiNaturalist), which explores various settings and hyperparameters. 

```
cd examples/keras-cnn-nature
wandb init
python train_small_cnn.py
```

## Finetuning experiments
 
 Enables two kinds of finetuning experiments:
 * loading various pretrained base CNNs in Keras (Xception, ResNet, InceptionResNetV2, InceptionV3), pretraining for some epochs, freezing some of the layers of the resulting network, then continuing to finetune the rest of the layers 
 * loading a small CNN, pretraining on general labels (in this case, predicting one of 5 biological classes) for a certain number of epochs, then finetuning on specific labels (predicting one of 25 biological species). Note that you'll likely need to modify the absolute paths for loading the curriculum learning data on L205:208 of ``finetune_experiments.py``.

For more context on this example, see this W&B [report](https://app.wandb.ai/stacey/curr_learn/reports?view=stacey%2Fkeras_nature_explore).

Highly configurable with commandline flags: run with ``-h`` to see all the options. 
```
cd examples/keras-cnn-nature
wandb init
python finetune_experiments.py
``` 
