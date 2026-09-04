# /// script
# dependencies = ["tensorflow", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Plot_Precision_Recall_Curves_with_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot Precision-Recall Curves with W&B

    How to log [Precision-Recall curves](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve) with [Vega](https://vega.github.io/vega/docs/) in [Weights & Biases](https://www.wandb.com).

    ## Method: wandb.plot.pr_curve()

    - More info and customization details: [Plot Precision Recall Curves](https://wandb.ai/wandb/plots/reports/Plot-Precision-Recall-Curves--VmlldzoyNjk1ODY)
    - More examples in this W&B project: [Custom Charts](https://app.wandb.ai/demo-team/custom-charts).

    These are simple cases to explain the basics—you can build much more sophisticated custom charts with our powerful new query editor.

    This Colab explores a transfer learning problem: finetuning InceptionV3 with ImageNet weights to identify 10 types of living things (birds, plants, insects, etc) from 10K photos from [iNaturalist 2017](https://github.com/visipedia/inat_comp).

    ![roc_and_pr](https://i.imgur.com/CqGXSzj.png)

    Note: Hyperparameters like number of epochs and training dataset size are set to minimum values here for demo efficiency. On the full training data, the model should get to the low 80s in validation accuracy within an epoch or so.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: Download data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note: **this stage might take a few minutes (~3.6GB of data)**. If you end up needing to rerun this cell, comment out the first capture line (change ```%%capture``` to ```#%%capture``` ) so you can respond to the prompt about re-downloading the dataset (and see the progress bar).

    Download sample data: 10,000 training images and 2,000 validation images from the [iNaturalist dataset](https://github.com/visipedia/inat_comp), evenly distributed across 10 classes of living things like birds, insects, plants, and mammals (names given in Latin—so Aves, Insecta, Plantae, etc :). We will fine-tune a convolutional neural network already trained on ImageNet on this task: given a photo of a living thing, correctly classify it into one of the 10 classes.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !curl -SL https://storage.googleapis.com/wandb_datasets/nature_12K.zip > nature_12K.zip
    # !unzip nature_12K.zip
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Install dependencies

    Install tensorflow and wandb; log in to wandb.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: tensorflow !pip install tensorflow -qqq
    # packages added via marimo's package management: wandb !pip install wandb -qqq
    return


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Training code

    Feel free to try different values for "NUM_TRAIN" and "NUM_EPOCHS" below so you can see a variety of PR curves (generally better ones with more training examples/longer training time)
    """)
    return


@app.cell
def _(wandb):
    # this determines the name of your wandb project, where all your
    # runs will be loggeed
    PROJECT_NAME = "custom_pr_curve"

    # EXPERIMENT CONFIG
    #---------------------------
    # try changing the number of training examples
    # to generate a range of different PR curves
    NUM_TRAIN = 100 # try 500, 1000, 2000, or max 10000
    NUM_EPOCHS = 1 # try 3, 5, or as many as you like

    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    from wandb.keras import WandbCallback

    # local paths to data
    train_data = "inaturalist_12K/train"
    val_data = "inaturalist_12K/val"

    # experiment configuration saved to W&B
    config_defaults = {
      # number of images used to train--set low for demo training speed
      # you can set this up to 10000 for the full dataset
      # GOOD CONFIG TO TRY: 100, 500, 1000, 2000
      "num_train" : NUM_TRAIN, # up to 10000,
      # number of images used to validate--set low for demo training speed
      # you can set this up to 2000 for the full dataset
      "num_val" : 500, #2000,
      "num_classes" : 10,
      "fc_size" : 1024,

      # inceptionV3 settings
      "img_width" : 299,
      "img_height": 299,
      "batch_size" : 32,

      # number of epochs--set low for demo training speed
      # you can set this up to 5, 10, or more for better results
      # GOOD CONFIG TO TRY: 3, 5, 10
      "pretrain_epochs" : NUM_EPOCHS, #5,
      # number of validation data batches to use when computing metrics
      # at the end of each epoch
      "num_log_batches": 15
    }

    def build_model(fc_size, num_classes):
      """Load InceptionV3 with ImageNet weights, freeze it,
      and attach a finetuning top for this classification task"""
      # load InceptionV3 as base
      base = InceptionV3(weights="imagenet", include_top="False")
      # freeze base layers
      for layer in base.layers:
        layer.trainable = False
      x = base.get_layer('mixed10').output 

      # attach a fine-tuning layer
      x = GlobalAveragePooling2D()(x)
      x = Dense(fc_size, activation='relu')(x)
      guesses = Dense(num_classes, activation='softmax')(x)

      model = Model(inputs=base.input, outputs=guesses)
      model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
      return model

    def pretrain():
      """ Main training loop. This is called pretrain because it freezes
      the InceptionV3 layers of the model and only trains the new top layers
      on the new data.   subsequent training phase would unfreeze all the layers
      and finetune the whole model on the new data""" 
      # track this experiment with wandb: all runs will be sent
      # to the given project name
      wandb.init(project=PROJECT_NAME, config=config_defaults)
      cfg = wandb.config

      # create train and validation data generators
      train_datagen = ImageDataGenerator(
          rescale=1. / 255,
          shear_range=0.2,
          zoom_range=0.2,
          horizontal_flip=True)
      val_datagen = ImageDataGenerator(rescale=1. / 255)

      train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(cfg.img_width, cfg.img_height),
        batch_size=cfg.batch_size,
        class_mode='categorical')

      val_generator = val_datagen.flow_from_directory(
        val_data,
        target_size=(cfg.img_width, cfg.img_height),
        batch_size=cfg.batch_size,
        class_mode='categorical')

      # instantiate model and callbacks
      model = build_model(cfg.fc_size, cfg.num_classes)
      callbacks = [WandbCallback(), PRMetrics(val_generator, num_log_batches=15)]

      # train!
      model.fit(
        train_generator,
        steps_per_epoch = cfg.num_train // cfg.batch_size,
        epochs=cfg.pretrain_epochs,
        validation_data=val_generator,
        callbacks = callbacks,
        validation_steps=cfg.num_val // cfg.batch_size)

      wandb.run.finish()
  
    class PRMetrics(Callback):
      """ Custom callback to compute per-class PR & ROC curves
      at the end of each training epoch"""
      def __init__(self, generator=None, num_log_batches=1):
        self.generator = generator
        self.num_batches = num_log_batches
        # store full names of classes
        self.class_names = { v: k for k, v in generator.class_indices.items() }
        self.flat_class_names = [k for k, v in generator.class_indices.items()]

      def on_epoch_end(self, epoch, logs={}):
        # collect validation data and ground truth labels from generator
        val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
        val_data, val_labels = np.vstack(val_data), np.vstack(val_labels)

        # use the trained model to generate predictions for the given number
        # of validation data batches (num_batches)
        val_predictions = self.model.predict(val_data)
        ground_truth_class_ids = val_labels.argmax(axis=1)

        # Log precision-recall curve
        # the key "pr_curve" is the id of the plot--do not change
        # this if you want subsequent runs to show up on the same plot
        wandb.log({"pr_curve" : wandb.plot.pr_curve(ground_truth_class_ids, 
                                                    val_predictions,
                                                    labels=self.flat_class_names)})

    return (pretrain,)


@app.cell
def _(pretrain):
    # run this cell to launch your experiment!
    # charts will show up in your run page under the heading "Custom Charts",
    # which you may need to click on to expand
    pretrain()
    return


if __name__ == "__main__":
    app.run()
