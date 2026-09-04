# /// script
# dependencies = ["wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Basic_Artifacts_with_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{artifacts-basics} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B Artifacts Quickstart

    <!--- @wandbcode{artifacts-basics} -->

    This tutorial shows how to get started with W&B Artifacts very quickly. I finetune a convnet in Keras to identify 10 types of living things in photos: plants, animals, insects, etc.
    [Check out the companion report on W&B](https://wandb.ai/wandb/arttest/reports/Artifacts-Quickstart--VmlldzozNTAzMDM)

    In this example we're using Google Colab as a convenient hosted environment, but you can run your own training scripts from anywhere and visualize metrics with W&B's experiment tracking tool.

    ## Sign up or login

    [Sign up or login](https://wandb.ai/login) to W&B to see and interact with your experiments in the browser.

    ### Note on Artifacts storage space and deletion

    Running this colab end-to-end will create at least 7GB of artifacts in your wandb account (more if you try different experiments, increase the number of epochs or examples, etc). If you'd like to free up this space later, you can
    * delete the whole project (top right menu at wandb.ai / USERNAME / PROJECT_NAME /overview), or
    * delete individual artifacts (hover on the three vertical dots to the right of the artifact name in the sidebar), or
    * delete specific artifact versions through the storage explorer at wandb.ai / storage / USERNAME /PROJECT_NAME.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download sample data: Nature photos

    Note: **this stage might take a few minutes (~3.6GB of data)**. If you end up needing to rerun this cell, comment out the first capture line (change ```%%capture``` to ```#%%capture``` ) so you can respond to the prompt about re-downloading the dataset (and see the progress bar).

    Download subsampled data: 10,000 training images and 2,000 validation images from the [iNaturalist dataset](https://github.com/visipedia/inat_comp), evenly distributed across 10 classes of living things like birds, insects, plants, and mammals (names given in Latin—so Aves, Insecta, Plantae, etc :). We will fine-tune a convolutional neural network already trained on ImageNet on this task: given a photo of a living thing, correctly classify it into one of the 10 classes.
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
    # Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Start out by installing the experiment tracking library and setting up your free W&B account:

    *   **pip install wandb** – Install the W&B library
    *   **import wandb** – Import the wandb library
    *   **wandb login** – Login to your W&B account so you can log all your metrics in one place
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qq
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _():
    import os
    from random import shuffle

    # source directory for all raw data (technically a subset of only 10K images)
    SRC = "inaturalist_12K/train"

    # number of images per class label
    # The total number of images is
    # 10 classes * 1000 images = 10,000 images in SRC
    NUM_IMAGES = 1000 # per class label, set this lower for faster results/fewer files
    PROJECT_NAME = "artifacts_demo"
    PREFIX = "inat" # convenient for tracking local data
    return NUM_IMAGES, PREFIX, PROJECT_NAME, SRC, os, shuffle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 1: Upload raw data
    """)
    return


@app.cell
def _(NUM_IMAGES, PREFIX, PROJECT_NAME, SRC, os, shuffle, wandb):
    _RAW_DATA_AT = '_'.join([PREFIX, 'raw_data_10K'])
    _run = wandb.init(project=PROJECT_NAME, job_type='upload')
    raw_data_at = wandb.Artifact(_RAW_DATA_AT, type='raw_data')
    # create an artifact for all the raw data
    _labels = os.listdir(SRC)
    for _l in _labels:
    # SRC_DIR contains 10 folders, one for each of 10 class labels
    # each folder contains images of the corresponding class
        _imgs_per_label = os.path.join(SRC, _l)
        if os.path.isdir(_imgs_per_label):
            _imgs = os.listdir(_imgs_per_label)
            shuffle(_imgs)
            img_file_ids = _imgs[:NUM_IMAGES]
            for f in img_file_ids:  # randomize the order
                file_path = os.path.join(SRC, _l, f)
                raw_data_at.add_file(file_path, name=_l + '/' + f)
    _run.log_artifact(raw_data_at)
    # save artifact to W&B
    _run.finish()  # add file to artifact by full path
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2: Split raw data to prepare for training
    """)
    return


@app.cell
def _(PREFIX, PROJECT_NAME, os, shuffle, wandb):
    _RAW_DATA_AT = 'inat_raw_data_10K'
    _run = wandb.init(project=PROJECT_NAME, job_type='data_split')
    data_at = _run.use_artifact(_RAW_DATA_AT + ':latest')
    # find the most recent ("latest") version of the full raw data
    # you can of course pass around programmatic aliases and not string literals
    data_dir = data_at.download()
    # download it locally (for illustration purposes/across hardware; you can
    # also sync/version artifacts by reference)
    DATA_SPLITS = {'train': 800, 'val': 100, 'test': 100}
    ats = {}
    # create balanced train, val, test splits
    # each count is the number of images per label
    for split, count in DATA_SPLITS.items():
        ats[split] = wandb.Artifact('_'.join([PREFIX, split, 'data', str(count * 10)]), '_'.join([split, 'data']))
    _labels = os.listdir(data_dir)
    # wrap artifacts in dictionary for convenience
    for _l in _labels:
        if _l.startswith('.'):
            continue
        _imgs_per_label = os.listdir(os.path.join(data_dir, _l))
        shuffle(_imgs_per_label)
        start_id = 0
        for split, count in DATA_SPLITS.items():  # skip non-label file
            split_imgs = _imgs_per_label[start_id:start_id + count]
            for img_file in split_imgs:
                full_path = os.path.join(data_dir, _l, img_file)
                ats[split].add_file(full_path, name=os.path.join(_l, img_file))
            start_id += count
    for split, artifact in ats.items():  # take a subset
        _run.log_artifact(artifact)
    # save all three artifacts to W&B
    # note: yes, in this example, we are cheating and have labels for the "test" data ;)
    _run.finish()  # add file to artifact by full path  # note: pass the label to the name parameter to retain it in  # the data structure 
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 3: Train with artifacts and save model
    """)
    return


@app.cell
def _(PREFIX, PROJECT_NAME, os, wandb):
    # EXPERIMENT CONFIG
    #---------------------------
    # number of training and validation examples
    # set these lower for fewer files/faster results
    # if you set these higher, make sure the total count is less than or equal to
    # the number of files uploaded for that split in the train/val data artifact
    NUM_TRAIN = 800  # try 500, 1000, 2000, or max 10000
    NUM_VAL = 100
    NUM_EPOCHS = 1  # set low for demo purposes; try 3, 5, or as many as you like
    MODEL_NAME = 'iv3_trained'
    # model name
    # if you want to train a sufficiently different model, give this a new name
    # to start a new lineage for the model, instead of just incrementing the
    # version of the old model
    INIT_MODEL_DIR = 'init_model_keras_iv3.keras'
    FINAL_MODEL_DIR = 'trained_keras_model_iv3.keras'
    # folder in which to save initial, untrained model
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve
    # folder in which to save the final, trained model
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
    config_defaults = {'num_train': NUM_TRAIN, 'num_val': NUM_VAL, 'num_classes': 10, 'fc_size': 1024, 'img_width': 299, 'img_height': 299, 'batch_size': 32, 'epochs': NUM_EPOCHS}

    def finetune_inception_model(fc_size, num_classes):
        """Load InceptionV3 with ImageNet weights, freeze it,
      and attach a finetuning top for this classification task"""
        base = InceptionV3(weights='imagenet', include_top='False')
        for layer in base.layers:
            layer.trainable = False
        x = base.get_layer('mixed10').output
        x = GlobalAveragePooling2D()(x)
        x = Dense(fc_size, activation='relu')(x)
        guesses = Dense(num_classes, activation='softmax')(x)
    # experiment configuration saved to W&B
        model = Model(inputs=base.input, outputs=guesses)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train():
        """ Main training loop. This is called pretrain because it freezes
      the InceptionV3 layers of the model and only trains the new top layers  # inceptionV3 settings
      on the new data.   subsequent training phase would unfreeze all the layers
      and finetune the whole model on the new data"""
        _run = wandb.init(project=PROJECT_NAME, job_type='train', config=config_defaults)
        cfg = wandb.config
        train_at = os.path.join(PROJECT_NAME, PREFIX + '_train_data_8000') + ':latest'
        val_at = os.path.join(PROJECT_NAME, PREFIX + '_val_data_1000') + ':latest'
        train_data = _run.use_artifact(train_at, type='train_data')
        train_dir = train_data.download()
        val_data = _run.use_artifact(val_at, type='val_data')
        val_dir = val_data.download()
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)  # load InceptionV3 as base
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(train_dir, target_size=(cfg.img_width, cfg.img_height), batch_size=cfg.batch_size, class_mode='categorical')  # freeze base layers
        val_generator = val_datagen.flow_from_directory(val_dir, target_size=(cfg.img_width, cfg.img_height), batch_size=cfg.batch_size, class_mode='categorical')
        model = finetune_inception_model(cfg.fc_size, cfg.num_classes)
        model_artifact = wandb.Artifact('iv3', type='model', description='unmodified inception v3', metadata=dict(cfg))
        model.save(INIT_MODEL_DIR)
        model_artifact.add_file(INIT_MODEL_DIR)  # attach a fine-tuning layer
        _run.log_artifact(model_artifact)
        callbacks = [WandbMetricsLogger(), WandbModelCheckpoint('checkpoint.keras')]
        model.fit(train_generator, steps_per_epoch=cfg.num_train // cfg.batch_size, epochs=cfg.epochs, validation_data=val_generator, callbacks=callbacks, validation_steps=cfg.num_val // cfg.batch_size)
        trained_model_artifact = wandb.Artifact(MODEL_NAME, type='model', description='trained inception v3', metadata=dict(cfg))
        model.save(FINAL_MODEL_DIR)
        trained_model_artifact.add_file(FINAL_MODEL_DIR)
        _run.log_artifact(trained_model_artifact)
        _run.finish()  # track this experiment with wandb: all runs will be sent  # to the given project name  # artifact names  # create train and validation data generators  # instantiate model and callbacks  # log model  # train!  # save trained model as artifact

    return MODEL_NAME, np, train


@app.cell
def _(train):
    train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 4: Load model for inference
    """)
    return


@app.cell
def _(MODEL_NAME, PROJECT_NAME, np, os, wandb):
    from tensorflow.keras.preprocessing import image
    from tensorflow import keras
    import pathlib
    _run = wandb.init(project=PROJECT_NAME, job_type='inference')
    model_at = _run.use_artifact(MODEL_NAME + ':latest')
    model_dir = pathlib.Path(model_at.download())
    print('model: ', model_dir)
    model_path = model_dir / 'trained_keras_model_iv3.keras'
    model = keras.models.load_model(model_path, compile=False)
    test_data_at = _run.use_artifact('inat_test_data_1000:latest')
    test_dir = test_data_at.download()
    _imgs = []
    class_labels = os.listdir(test_dir)
    for _l in class_labels:
        if _l.startswith('.'):
            continue
        imgs_per_class = os.listdir(os.path.join(test_dir, _l))
        for img in imgs_per_class:
            img_path = os.path.join(test_dir, _l, img)
            img = image.load_img(img_path, target_size=(299, 299))
            img = image.img_to_array(img)
            img = np.expand_dims(img / 255.0, axis=0)
            _imgs.append(img)
    preds = {}
    _imgs = np.vstack(_imgs)
    classes = model.predict(_imgs, batch_size=32)
    for c in classes:
        class_id = np.argmax(c)
        preds[class_id] = preds.get(class_id, 0) + 1
    print(preds)
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # More about Weights & Biases
    We're always free for academics and open source projects. Here are some more resources:

    1. [Documentation](http://docs.wandb.com) - Python docs
    2. [Gallery](https://app.wandb.ai/gallery) - example reports in W&B
    3. [Articles](https://www.wandb.com/articles) - blog posts and tutorials
    4. [Community](wandb.me/slack) - join our Slack community forum
    """)
    return


if __name__ == "__main__":
    app.run()
