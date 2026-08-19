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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/Image_Classification_with_Tables.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{tables_mendeleev} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{tables_mendeleev} -->

    # Image Classification with W&B Tables

    This is a walkthrough of [Tables for visualization](https://docs.wandb.ai/guides/data-vis/tables) and [Artifacts for versioning](https://docs.wandb.com/artifacts) deep learning models in Weights & Biases. As an example, I finetune a convnet in Keras on photos from  [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017) to identify 10 classes of living things (plants, insects, birds, etc).

    <img src="https://i.imgur.com/PK4VA6u.png"
    alt="Table comparison example"/>

    ## [Explore more examples in this W&B Report](https://wandb.ai/stacey/mendeleev/reports/DSViz-for-Image-Classification--VmlldzozNjE3NjA)

    ## Sign up or login

    [Sign up or login](https://wandb.ai/login) to W&B to see and interact with your experiments in the browser.

    In this example we're using Google Colab as a convenient hosted environment, but you can run your own training scripts from anywhere and visualize metrics with W&B's experiment tracking tool.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download sample data: Choose 1 of 4 sizes

    Choose one of the three dataset size options below to run the rest of the demo. With fewer images, you'll run through the demo much faster and use less storage space. With more images, you'll get more realistic model training and more interesting results and examples to explore.

    Note: **for the largest dataset, this stage might take a few minutes**. If you end up needing to rerun a cell, comment out the first capture line (change ```%%capture``` to ```#%%capture``` ) so you can respond to the prompt about re-downloading the dataset (and see the progress bar).

    Each zipped directory contains randomly sampled images from the [iNaturalist dataset](https://github.com/visipedia/inat_comp), evenly distributed across 10 classes of living things like birds, insects, plants, and mammals (names given in Latin—so Aves, Insecta, Plantae, etc :).
    """)
    return


@app.cell
def _():
    # set SIZE to "TINY", "SMALL", "MEDIUM", or "LARGE"
    # to select one of these three datasets
    # TINY dataset: 100 images, 30MB
    # SMALL dataset: 1000 images, 312MB
    # MEDIUM dataset: 5000 images, 1.5GB
    # LARGE dataset: 12,000 images, 3.6GB

    SIZE = "SMALL"
    return (SIZE,)


@app.cell
def _(SIZE):
    if SIZE == "TINY":
      src_url = "https://storage.googleapis.com/wandb_datasets/nature_100.zip"
      src_zip = "nature_100.zip"
      DATA_SRC = "nature_100"
      IMAGES_PER_LABEL = 10
      BALANCED_SPLITS = {"train" : 8, "val" : 1, "test": 1}
    elif SIZE == "SMALL":
      src_url = "https://storage.googleapis.com/wandb_datasets/nature_1K.zip"
      src_zip = "nature_1K.zip"
      DATA_SRC = "nature_1K"
      IMAGES_PER_LABEL = 100
      BALANCED_SPLITS = {"train" : 80, "val" : 10, "test": 10}
    elif SIZE == "MEDIUM":
      src_url = "https://storage.googleapis.com/wandb_datasets/nature_12K.zip"
      src_zip = "nature_12K.zip"
      DATA_SRC = "inaturalist_12K/train" # (technically a subset of only 10K images)
      IMAGES_PER_LABEL = 500
      BALANCED_SPLITS = {"train" : 400, "val" : 50, "test": 50}
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
    # packages added via marimo's package management: wandb !pip install wandb -qU
    return


@app.cell
def _():
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
    import numpy as np

    # source directory for all raw data
    SRC = DATA_SRC
    PREFIX = "inat" # convenient for tracking local data
    PROJECT_NAME = "nature_photos"

    # number of images per class label
    # the total number of images is 10X this (10 classes)
    TOTAL_IMAGES = IMAGES_PER_LABEL * 10
    return PREFIX, PROJECT_NAME, SRC, TOTAL_IMAGES, np, os, shuffle


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
    # if this is a substantially new dataset, give it a new name
    # this will create a whole new placeholder (Artifact) for this dataset
    # instead of just incrementing a version of the old dataset
    RAW_DATA_AT = '_'.join([PREFIX, 'raw_data', str(TOTAL_IMAGES)])
    _run = wandb.init(project=PROJECT_NAME, job_type='upload')
    # create an artifact for all the raw data
    raw_data_at = wandb.Artifact(RAW_DATA_AT, type='raw_data')
    _labels = os.listdir(SRC)
    # SRC_DIR contains 10 folders, one for each of 10 class labels
    # each folder contains images of the corresponding class
    for _l in _labels:
        _imgs_per_label = os.path.join(SRC, _l)
        if os.path.isdir(_imgs_per_label):
            _imgs = [i for i in os.listdir(_imgs_per_label) if not i.startswith('.DS')]
            shuffle(_imgs)  # filter out "DS_Store"
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
    ![img](https://i.imgur.com/EjVjKuL.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2: Split raw data to prepare for training
    """)
    return


@app.cell
def _(
    BALANCED_SPLITS,
    PREFIX,
    PROJECT_NAME,
    RAW_DATA_AT,
    SIZE,
    TOTAL_IMAGES,
    os,
    shuffle,
    wandb,
):
    # if this is a substantially different dataset, give it a new name
    # this will create a whole new placeholder (Artifact) for this split
    # instead of just incrementing a version of the old data split
    SPLIT_DATA_AT = '_'.join([PREFIX, '80-10-10', str(TOTAL_IMAGES)])
    _run = wandb.init(project=PROJECT_NAME, job_type='data_split')
    SPLIT_COUNTS = BALANCED_SPLITS
    # create balanced train, val, test splits
    # each count is the number of images per label
    data_at = _run.use_artifact(RAW_DATA_AT + ':latest')
    data_dir = data_at.download()
    # find the most recent ("latest") version of the full raw data
    # you can of course pass around programmatic aliases and not string literals
    # note: RAW_DATA_AT is defined in the previous cell—if you're running
    # just this step, you may need to hardcode it
    data_split_at = wandb.Artifact(SPLIT_DATA_AT, type='balanced_data')
    # download it locally (for illustration purposes/across hardware; you can
    # also sync/version artifacts by reference)
    preview_dt = wandb.Table(columns=['id', 'image', 'label', 'split'])
    _labels = os.listdir(data_dir)
    for _l in _labels:
        if _l.startswith('.'):
    # create a table with columns we want to track/compare
            continue
        _imgs_per_label = os.listdir(os.path.join(data_dir, _l))
        shuffle(_imgs_per_label)
        start_id = 0
        for split, count in SPLIT_COUNTS.items():  # skip non-label file
            split_imgs = _imgs_per_label[start_id:start_id + count]
            for img_file in split_imgs:
                f_id = img_file.split('.')[0]
                full_path = os.path.join(data_dir, _l, img_file)
                data_split_at.add_file(full_path, name=os.path.join(split, _l, img_file))
                if SIZE == 'LARGE':  # take a subset
                    continue
                if split != 'test':
                    preview_dt.add_data(f_id, wandb.Image(full_path), _l, split)
                else:
                    preview_dt.add_data(f_id, wandb.Image(full_path), 'unknown', split)  # add file to artifact by full path
            start_id += count  # note: pass the label to the name parameter to retain it in
    data_split_at.add(preview_dt, 'data_split')  # the data structure 
    _run.log_artifact(data_split_at)
    # log artifact to W&B
    _run.finish()  # add a preview of the image  # skip for the largest dataset for efficiency  # pretend we have unlabeled test data  # (replace "unknown" with l if you'd like to keep the labels :)
    return data_split_at, preview_dt


@app.cell
def _(data_split_at, preview_dt):
    # NOTE: if this Colab is running out of RAM, try running this cell
    del data_split_at
    del preview_dt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 3: Train with artifacts and save model
    """)
    return


@app.cell
def _(BALANCED_SPLITS, PREFIX, PROJECT_NAME, TOTAL_IMAGES, np, os, wandb):
    # EXPERIMENT CONFIG
    #------------------------
    # Core globals to modify
    NUM_EPOCHS = 1  # set low for demo purposes, try 3, or 5, or as many as you like
    _RUN_NAME = ''
    VAL_TABLE_NAME = 'predictions'
    # optional globals to modify
    # set to a custom name to help keep your experiments organized
    NUM_TRAIN = BALANCED_SPLITS['train'] * 10
    # change this if you'd like start a new set of comparable Tables
    # (only Tables logged to the same key can be compared)
    NUM_VAL = BALANCED_SPLITS['val'] * 10
    NUM_LOG_BATCHES = 16
    # hyperparams set low for demo/training speed
    # if you set these higher, be mindful of how many items are in
    # the dataset artifacts you chose by setting the SIZE at the top
    TRAIN_DATA_AT = PREFIX + '_80-10-10_' + str(TOTAL_IMAGES)
    _MODEL_NAME = 'iv3_finetuned'
    SAVE_MODEL_DIR = 'finetune_iv3_keras'
    # enforced max for this is ceil(NUM_VAL/batch_size)
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.callbacks import Callback
    # ARTIFACTS CONFIG
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    # training data artifact to load
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    # model name
    # if you want to train a sufficiently different model, give this a new name
    # to start a new lineage for the model, instead of just incrementing the
    # version of the old model
    from wandb.keras import WandbCallback
    CFG = {'num_train': NUM_TRAIN, 'num_val': NUM_VAL, 'num_classes': 10, 'fc_size': 1024, 'epochs': NUM_EPOCHS, 'batch_size': 32, 'img_width': 299, 'img_height': 299}
    # folder in which to save the final, trained model
    max_log_batches = int(np.ceil(float(CFG['num_val']) / float(CFG['batch_size'])))
    CFG['num_log_batches'] = min(max_log_batches, NUM_LOG_BATCHES)

    def finetune_inception_model(fc_size, num_classes):
        """Load InceptionV3 with ImageNet weights, freeze it,
      and attach a finetuning top for this classification task"""
        base = InceptionV3(weights='imagenet', include_top='False')
        for layer in base.layers:
            layer.trainable = False
        x = base.get_layer('mixed10').output
        x = GlobalAveragePooling2D()(x)
        x = Dense(fc_size, activation='relu')(x)
    # experiment configuration saved to W&B
        guesses = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=guesses)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train():
        """ Main training loop which freezes the InceptionV3 layers of the model
      and only trains the new top layers on the new data. A subsequent training
      phase might unfreeze all the layers and finetune the whole model on the new data"""  # inceptionV3 settings
        _run = wandb.init(project=PROJECT_NAME, name=_RUN_NAME, job_type='train', config=CFG)
        cfg = wandb.config
        data_at = TRAIN_DATA_AT + ':latest'
        data = _run.use_artifact(data_at, type='balanced_data')
    # number of validation data batches to log/use when computing metrics
    # at the end of each epoch
        data_dir = data.download()
    # change this min to max to log ALL the available images to a Table
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        train_datagen = ImageDataGenerator(rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_generator = train_datagen.flow_from_directory(train_dir, target_size=(cfg.img_width, cfg.img_height), batch_size=cfg.batch_size, class_mode='categorical')
        val_generator = val_datagen.flow_from_directory(val_dir, target_size=(cfg.img_width, cfg.img_height), batch_size=cfg.batch_size, class_mode='categorical', shuffle=False)  # load InceptionV3 as base
        model = finetune_inception_model(cfg.fc_size, cfg.num_classes)
        callbacks = [WandbCallback(), ValLog(val_generator, cfg.num_log_batches)]  # freeze base layers
        model.fit(train_generator, steps_per_epoch=cfg.num_train // cfg.batch_size, epochs=cfg.epochs, validation_data=val_generator, callbacks=callbacks, validation_steps=cfg.num_val // cfg.batch_size)
        trained_model_artifact = wandb.Artifact(_MODEL_NAME, type='model', description='finetuned inception v3', metadata=dict(cfg))
        model.save(SAVE_MODEL_DIR)
        trained_model_artifact.add_dir(SAVE_MODEL_DIR)
        _run.log_artifact(trained_model_artifact)  # attach a fine-tuning layer
        _run.finish()

    class ValLog(Callback):
        """ Custom callback to log validation images
      at the end of each training epoch"""

        def __init__(self, generator=None, num_log_batches=1):
            self.generator = generator
            self.num_batches = num_log_batches
            self.flat_class_names = [k for k, v in generator.class_indices.items()]

        def on_epoch_end(self, epoch, logs={}):
            val_data, val_labels = zip(*(self.generator[i] for i in range(self.num_batches)))
            val_data, val_labels = (np.vstack(val_data), np.vstack(val_labels))
            val_preds = self.model.predict(val_data)
            true_ids = val_labels.argmax(axis=1)  # locate and download training and validation data
            max_preds = val_preds.argmax(axis=1)
            columns = ['id', 'image', 'guess', 'truth']
            for a in self.flat_class_names:
                columns.append('score_' + a)
            predictions_table = wandb.Table(columns=columns)
            for filepath, img, top_guess, scores, truth in zip(self.generator.filenames, val_data, max_preds, val_preds, true_ids):
                img_id = filepath.split('/')[-1].split('.')[0]  # create train and validation data generators
                row = [img_id, wandb.Image(img), self.flat_class_names[top_guess], self.flat_class_names[truth]]
                for s in scores.tolist():
                    row.append(np.round(s, 4))
                predictions_table.add_data(*row)
            wandb.run.log({VAL_TABLE_NAME: predictions_table})  # instantiate model and callbacks  # train!  # save trained model as artifact  # store full names of classes  # collect validation data and ground truth labels from generator  # use the trained model to generate predictions for the given number  # of validation data batches (num_batches)  # log validation predictions alongside the run  # log image, predicted and actual labels, and all scores

    return (train,)


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
def _(PREFIX, PROJECT_NAME, TOTAL_IMAGES, np, os, wandb):
    _RUN_NAME = ''
    TEST_TABLE_NAME = 'test_results'
    from tensorflow import keras
    from tensorflow.keras.preprocessing import image
    _MODEL_NAME = 'iv3_finetuned'
    TEST_DATA_AT = '_'.join([PREFIX, '80-10-10', str(TOTAL_IMAGES)])
    _run = wandb.init(project=PROJECT_NAME, job_type='inference', name=_RUN_NAME)
    model_at = _run.use_artifact(_MODEL_NAME + ':latest')
    model_dir = model_at.download()
    print('model: ', model_dir)
    model = keras.models.load_model(model_dir)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    test_data_at = _run.use_artifact(TEST_DATA_AT + ':latest')
    test_dir = test_data_at.download()
    test_dir += '/test/'
    class_names = ['Animalia', 'Amphibia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']
    _imgs = []
    filenames = []
    class_labels = os.listdir(test_dir)
    truth = []
    for _l in class_labels:
        if _l.startswith('.'):
            continue
        imgs_per_class = os.listdir(os.path.join(test_dir, _l))
        for img in imgs_per_class:
            filenames.append(img.split('.')[0])
            truth.append(_l)
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
    columns = ['id', 'image', 'guess', 'truth']
    for a in class_names:
        columns.append('score_' + a)
    test_dt = wandb.Table(columns=columns)
    for img_id, i, t, c in zip(filenames, _imgs, truth, classes):
        guess = class_names[np.argmax(c)]
        row = [img_id, wandb.Image(i), guess, t]
        for c_i in c.tolist():
            row.append(np.round(c_i, 4))
        test_dt.add_data(*row)
    _run.log({TEST_TABLE_NAME: test_dt})
    print('Quick distribution of predicted classes: ')
    print(preds)
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # More about Weights & Biases
    We're always free for academics and open source projects. Email carey@wandb.com with any questions or feature suggestions. Here are some more resources:

    1. [Documentation](http://docs.wandb.com) - Python docs
    2. [Gallery](https://app.wandb.ai/gallery) - example reports in W&B
    3. [Articles](https://www.wandb.com/articles) - blog posts and tutorials
    4. [Community](wandb.me/slack) - join our Slack community forum
    """)
    return


if __name__ == "__main__":
    app.run()
