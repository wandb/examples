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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{artifacts_quickstart} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{artifacts_quickstart} -->

    # W&B Artifacts Quickstart

    This tutorial shows how to get started with **W&B Artifacts** very quickly. I finetune a convnet in Keras to identify 10 types of living things in photos: plants, animals, insects, etc.

    * [follow along in a W&B Report](https://wandb.ai/wandb/arttest/reports/Artifacts-Quickstart--VmlldzozNTAzMDM)
    * [see the Artifacts API and documentation](https://docs.wandb.com/artifacts/api)

    This demo will generate an experiment workflow like the following:

    ![artifact DAG](https://i.imgur.com/Kn69ir1.png)

    In this example we're using Google Colab as a convenient hosted environment, but you can run your own training scripts from anywhere and visualize metrics with W&B's experiment tracking tool.

    ## Sign up or login

    [Sign up or login](https://wandb.ai/login) to W&B to see and interact with your experiments in the browser.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download sample data: Choose 1 of 3 sizes

    Choose one of the three dataset size options below to run the rest of the demo. With fewer images, you'll run through the demo much faster and use less storage space. With more images, you'll get more realistic model training and more interesting results and examples to explore.

    Note: **for the largest dataset, this stage might take a few minutes**. If you end up needing to rerun a cell, comment out the first capture line (change ```%%capture``` to ```#%%capture``` ) so you can respond to the prompt about re-downloading the dataset (and see the progress bar).

    Each zipped directory contains randomly sampled images from the [iNaturalist dataset](https://github.com/visipedia/inat_comp), evenly distributed across 10 classes of living things like birds, insects, plants, and mammals (names given in Latin—so Aves, Insecta, Plantae, etc :).
    """)
    return


@app.cell
def _():
    # set SIZE to "TINY", "MEDIUM", or "LARGE"
    # to select one of these three datasets
    # TINY dataset: 100 images, 30MB
    # MEDIUM dataset: 1000 images, 312MB
    # LARGE datast: 12,000 images, 3.6GB

    SIZE = "TINY"
    return (SIZE,)


@app.cell
def _(SIZE):
    if SIZE == "TINY":
      src_url = "https://storage.googleapis.com/wandb_datasets/nature_100.zip"
      src_zip = "nature_100.zip"
      DATA_SRC = "nature_100"
      IMAGES_PER_LABEL = 10
      BALANCED_SPLITS = {"train" : 8, "val" : 1, "test": 1}
    elif SIZE == "MEDIUM":
      src_url = "https://storage.googleapis.com/wandb_datasets/nature_1K.zip"
      src_zip = "nature_1K.zip"
      DATA_SRC = "nature_1K"
      IMAGES_PER_LABEL = 100
      BALANCED_SPLITS = {"train" : 80, "val" : 10, "test": 10}
    elif SIZE == "LARGE":
      src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
      src_zip = "nature_12K.zip"
      DATA_SRC = "inaturalist_12K/train" # (technically a subset of only 10K images)
      IMAGES_PER_LABEL = 1000
      BALANCED_SPLITS = {"train" : 800, "val" : 100, "test": 100}
    return BALANCED_SPLITS, DATA_SRC, IMAGES_PER_LABEL


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !curl -SL $src_url > $src_zip
    # !unzip $src_zip
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 0: Setup
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
def _(DATA_SRC, IMAGES_PER_LABEL):
    import os
    from random import shuffle

    # source directory for all raw data
    SRC = DATA_SRC
    # number of images per class label
    # the total number of images is 10X this (10 classes)
    TOTAL_IMAGES = IMAGES_PER_LABEL * 10
    PROJECT_NAME = "artifacts_demo"
    PREFIX = "inat" # convenient for tracking local data
    return PREFIX, PROJECT_NAME, SRC, TOTAL_IMAGES, os, shuffle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 1: Upload raw data
    """)
    return


@app.cell
def _(
    IMAGES_PER_LABEL,
    PREFIX,
    PROJECT_NAME,
    SRC,
    TOTAL_IMAGES,
    os,
    shuffle,
    wandb,
):
    RAW_DATA_AT = '_'.join([PREFIX, 'raw_data', str(TOTAL_IMAGES)])
    _run = wandb.init(project=PROJECT_NAME, job_type='upload')
    raw_data_at = wandb.Artifact(RAW_DATA_AT, type='raw_data')
    # create an artifact for all the raw data
    _labels = os.listdir(SRC)
    for _l in _labels:
    # SRC_DIR contains 10 folders, one for each of 10 class labels
    # each folder contains images of the corresponding class
        _imgs_per_label = os.path.join(SRC, _l)
        if os.path.isdir(_imgs_per_label):
            _imgs = os.listdir(_imgs_per_label)
            shuffle(_imgs)
            img_file_ids = _imgs[:IMAGES_PER_LABEL]
            for f in img_file_ids:  # randomize the order
                file_path = os.path.join(SRC, _l, f)
                raw_data_at.add_file(file_path, name=_l + '/' + f)
    _run.log_artifact(raw_data_at)
    # save artifact to W&B
    _run.finish()  # add file to artifact by full path
    return (RAW_DATA_AT,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2: Prepare a data split
    """)
    return


@app.cell
def _(BALANCED_SPLITS, PREFIX, PROJECT_NAME, RAW_DATA_AT, os, shuffle, wandb):
    _run = wandb.init(project=PROJECT_NAME, job_type='data_split')
    data_at = _run.use_artifact(RAW_DATA_AT + ':latest')
    # find the most recent ("latest") version of the full raw data
    # you can of course pass around programmatic aliases and not string literals
    data_dir = data_at.download()
    # download it locally (for illustration purposes/across hardware; you can
    # also sync/version artifacts by reference)
    DATA_SPLITS = BALANCED_SPLITS
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
def _(BALANCED_SPLITS, PREFIX, PROJECT_NAME, os, wandb):
    NUM_TRAIN = BALANCED_SPLITS['train'] * 10
    NUM_VAL = BALANCED_SPLITS['val'] * 10
    NUM_EPOCHS = 1
    MODEL_NAME = 'iv3_trained'
    INIT_MODEL_DIR = 'init_model_keras_iv3.keras'
    FINAL_MODEL_DIR = 'trained_keras_model_iv3.keras'
    import numpy as np
    from sklearn.metrics import precision_recall_curve, roc_curve
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
    config_defaults = {'num_train': NUM_TRAIN, 'num_val': NUM_VAL, 'epochs': NUM_EPOCHS, 'num_classes': 10, 'fc_size': 1024, 'img_width': 299, 'img_height': 299, 'batch_size': 32}

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
        train_at = os.path.join(PROJECT_NAME, PREFIX + '_train_data_' + str(NUM_TRAIN)) + ':latest'
        val_at = os.path.join(PROJECT_NAME, PREFIX + '_val_data_' + str(NUM_VAL)) + ':latest'
        train_data = _run.use_artifact(train_at, type='train_data')
        train_dir = train_data.download()
        val_data = _run.use_artifact(val_at, type='val_data')
        val_dir = val_data.download()
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(train_dir, target_size=(cfg.img_width, cfg.img_height), batch_size=cfg.batch_size, class_mode='categorical')
        val_generator = val_datagen.flow_from_directory(val_dir, target_size=(cfg.img_width, cfg.img_height), batch_size=cfg.batch_size, class_mode='categorical')
        model = finetune_inception_model(cfg.fc_size, cfg.num_classes)
        model_artifact = wandb.Artifact('iv3', type='model', description='unmodified inception v3', metadata=dict(cfg))
        model.save(INIT_MODEL_DIR)
        model_artifact.add_file(INIT_MODEL_DIR)
        _run.log_artifact(model_artifact)
        callbacks = [WandbMetricsLogger(), WandbModelCheckpoint('checkpoint.keras')]
        model.fit(train_generator, steps_per_epoch=cfg.num_train // cfg.batch_size, epochs=cfg.epochs, validation_data=val_generator, callbacks=callbacks, validation_steps=cfg.num_val // cfg.batch_size)
        trained_model_artifact = wandb.Artifact(MODEL_NAME, type='model', description='trained inception v3', metadata=dict(cfg))
        model.save(FINAL_MODEL_DIR)
        trained_model_artifact.add_file(FINAL_MODEL_DIR)
        _run.log_artifact(trained_model_artifact)
        _run.finish()

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
def _(BALANCED_SPLITS, MODEL_NAME, PREFIX, PROJECT_NAME, np, os, wandb):
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image
    import pathlib
    _run = wandb.init(project=PROJECT_NAME, job_type='inference')
    model_at = _run.use_artifact(MODEL_NAME + ':latest')
    artifact_dir = pathlib.Path(model_at.download())
    print('artifact directory:', artifact_dir)
    model_path = artifact_dir / 'trained_keras_model_iv3.keras'
    model = keras.models.load_model(model_path, compile=False)
    print('loaded model from', model_path)
    TEST_DATA_AT = PREFIX + '_test_data_' + str(BALANCED_SPLITS['test'] * 10) + ':latest'
    test_data_at = _run.use_artifact(TEST_DATA_AT)
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
        if class_id in preds:
            preds[class_id] += 1
        else:
            preds[class_id] = 1
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
