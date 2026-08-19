# /// script
# dependencies = ["lightning", "torchvision", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Wandb_End_to_End_with_PyTorch_Lightning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{pytorch-lightning-e2e} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{pytorch-lightning-colab} -->

    # W&B Tutorial with Pytorch Lightning
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🛠️ Install `wandb` and `pytorch-lightning`
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: lightning wandb torchvision !pip install -q lightning wandb torchvision
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Login to W&B either through Python or CLI
    If you are using the public W&B cloud, you don't need to specify the `WANDB_HOST`.

    You can set environment variables `WANDB_API_KEY` and `WANDB_HOST` and pass them in as:
    ```
    import os
    import wandb

    wandb.login(host=os.getenv("WANDB_HOST"), key=os.getenv("WANDB_API_KEY"))
    ```
    You can also login via the CLI with:
    ```
    wandb login --host <YOUR_WANDB_HOST> <YOUR API KEY>
    ```
    """)
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
    ## ⚱ Logging the Raw Training Data as an Artifact
    """)
    return


@app.cell
def _():
    #@title Enter your W&B project and entity

    # FORM VARIABLES
    PROJECT_NAME = "pytorch-lightning-e2e" #@param {type:"string"}
    ENTITY = "wandb"#@param {type:"string"}

    # set SIZE to "TINY", "SMALL", "MEDIUM", or "LARGE"
    # to select one of these three datasets
    # TINY dataset: 100 images, 30MB
    # SMALL dataset: 1000 images, 312MB
    # MEDIUM dataset: 5000 images, 1.5GB
    # LARGE dataset: 12,000 images, 3.6GB

    SIZE = "TINY"

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
    return ENTITY, PROJECT_NAME


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !curl -SL $src_url > $src_zip
    # !unzip $src_zip
    return


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return


@app.cell
def _(ENTITY, PROJECT_NAME, wandb):
    import pandas as pd
    import os
    with wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='log_datasets') as run:
        img_paths = []
        for root, dirs, files in os.walk('nature_100', topdown=False):
            for name in files:
                img_path = os.path.join(root, name)
                label = img_path.split('/')[1]
                img_paths.append([img_path, label])
        index_df = pd.DataFrame(columns=['image_path', 'label'], data=img_paths)
        index_df.to_csv('index.csv', index=False)
        train_art = wandb.Artifact(name='Nature_100', type='raw_images', description='nature image dataset with 10 classes, 10 images per class')
        train_art.add_dir('nature_100')
        train_art.add_file('index.csv')
        wandb.log_artifact(train_art)  # Also adding a csv indicating the labels of each image
    return os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Using Artifacts in Pytorch Lightning `DataModule`'s and Pytorch `Dataset`'s
    - Makes it easy to interopt your DataLoaders with new versions of datasets
    - Just indicate the `name:alias` as an argument to your `Dataset` or `DataModule`
    """)
    return


@app.cell
def _(os, pd):
    from torchvision import transforms
    import lightning.pytorch as pl
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from skimage import io, transform
    from torchvision import utils, models
    import math

    class NatureDataset(Dataset):

        def __init__(self, wandb_run, artifact_name_alias='Nature_100:latest', local_target_dir='Nature_100:latest', transform=None):
            self.local_target_dir = local_target_dir
            self.transform = transform
            art = wandb_run.use_artifact(artifact_name_alias)
            path_at = art.download(root=self.local_target_dir)
            self.ref_df = pd.read_csv(os.path.join(self.local_target_dir, 'index.csv'))
            self.class_names = self.ref_df.iloc[:, 1].unique().tolist()
            self.idx_to_class = {k: v for k, v in enumerate(self.class_names)}  # Pull down the artifact locally to load it into memory
            self.class_to_idx = {v: k for k, v in enumerate(self.class_names)}

        def __len__(self):
            return len(self.ref_df)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()
            img_path = self.ref_df.iloc[idx, 0]
            image = io.imread(img_path)
            label = self.ref_df.iloc[idx, 1]
            label = torch.tensor(self.class_to_idx[label], dtype=torch.long)
            if self.transform:
                image = self.transform(image)
            return (image, label)

    class NatureDatasetModule(pl.LightningDataModule):

        def __init__(self, wandb_run, artifact_name_alias: str='Nature_100:latest', local_target_dir: str='Nature_100:latest', batch_size: int=16, input_size: int=224, seed: int=42):
            super().__init__()
            self.wandb_run = wandb_run
            self.artifact_name_alias = artifact_name_alias
            self.local_target_dir = local_target_dir
            self.batch_size = batch_size
            self.input_size = input_size
            self.seed = seed

        def setup(self, stage=None):
            self.nature_dataset = NatureDataset(wandb_run=self.wandb_run, artifact_name_alias=self.artifact_name_alias, local_target_dir=self.local_target_dir, transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(self.input_size), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
            nature_length = len(self.nature_dataset)
            train_size = math.floor(0.8 * nature_length)
            val_size = math.floor(0.2 * nature_length)
            self.nature_train, self.nature_val = random_split(self.nature_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))
            return self

        def train_dataloader(self):
            return DataLoader(self.nature_train, batch_size=self.batch_size)

        def val_dataloader(self):
            return DataLoader(self.nature_val, batch_size=self.batch_size)

        def predict_dataloader(self):
            pass

        def teardown(self, stage: str):
            pass

    return NatureDatasetModule, models, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ##How Logging in your Pytorch `LightningModule`works:
    When you train the model using `Trainer`, ensure you have a `WandbLogger` instantiated and passed in as a `logger`.

    ```
    wandb_logger = WandbLogger(project="my_project", entity="machine-learning")
    trainer = Trainer(logger=wandb_logger)
    ```

    You can always use `wandb.log` as normal throughout the module. When the `WandbLogger` is used, `self.log` will also log metrics to W&B.
    - To access the current run from within the `LightningModule`, you can access `Trainer.logger.experiment`, which is a `wandb.Run` object
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Some helper functions
    """)
    return


@app.cell
def _(models, torch):
    # Some helper functions
    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in _model.parameters():
                param.requires_grad = False

    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
        model_ft = None
        input_size = 0  # Initialize these variables which will be set in this if statement. Each of these
        if model_name == 'resnet':  #   variables is model specific.
            ' Resnet18\n        '
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
            input_size = 224
        elif model_name == 'squeezenet':
            ' Squeezenet\n        '
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = torch.nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224
        elif model_name == 'densenet':
            ' Densenet\n        '
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = torch.nn.Linear(num_ftrs, num_classes)
            input_size = 224
        else:
            print('Invalid model name, exiting...')
            exit()
        return (model_ft, input_size)

    return (initialize_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Writing the `LightningModule`
    """)
    return


@app.cell
def _(initialize_model, torch, wandb):
    from torch.nn import Linear, CrossEntropyLoss, functional as F
    from torch.optim import Adam
    from torchmetrics.functional import accuracy
    from lightning.pytorch import LightningModule

    class NatureLitModule(LightningModule):

        def __init__(self, model_name, num_classes=10, feature_extract=True, lr=0.01):
            """method used to define our model parameters"""
            super().__init__()
            self.model_name = model_name
            self.num_classes = num_classes
            self.feature_extract = feature_extract
            self.model, self.input_size = initialize_model(model_name=self.model_name, num_classes=self.num_classes, feature_extract=True)
            self.loss = CrossEntropyLoss()
            self.lr = lr
            self.save_hyperparameters()
            wandb.watch(self.model)

        def forward(self, x):
            """method used for inference input -> output"""
            x = self.model(x)
            return x
      # loss
        def training_step(self, batch, batch_idx):
            """needs to return a loss from a single batch"""
            preds, y, loss, acc = self._get_preds_loss_accuracy(batch)  # optimizer parameters
            self.log('train/loss', loss)
            self.log('train/accuracy', acc)
            return loss  # save hyper-parameters to self.hparams (auto-logged by W&B)

        def validation_step(self, batch, batch_idx):
            """used for logging metrics"""  # Record the gradients of all the layers
            preds, y, loss, acc = self._get_preds_loss_accuracy(batch)
            self.log('validation/loss', loss)
            self.log('validation/accuracy', acc)
            return (preds, y)

        def test_step(self, batch, batch_idx):
            """used for logging metrics"""
            preds, y, loss, acc = self._get_preds_loss_accuracy(batch)
            self.log('test/loss', loss)
            self.log('test/accuracy', acc)

        def configure_optimizers(self):
            """defines model optimizer"""  # Log loss and metric
            return Adam(self.parameters(), lr=self.lr)

        def _get_preds_loss_accuracy(self, batch):
            """convenience function since train/valid/test steps are similar"""
            x, y = batch
            logits = self(x)
            preds = torch.argmax(logits, dim=1)
            loss = self.loss(logits, y)
            acc = accuracy(preds, y, task='multiclass', num_classes=10)
            return (preds, y, loss, acc)  # Log loss and metric  # Let's return preds to use it in a custom callback  # Log loss and metric

    return (NatureLitModule,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Instrument Callbacks to log additional things at certain points in your code
    """)
    return


@app.cell
def _(pd, wandb):
    from lightning.pytorch.callbacks import Callback

    class LogPredictionsCallback(Callback):

        def __init__(self):
            super().__init__()

        def on_validation_epoch_start(self, trainer, pl_module):
            self.batch_dfs = []
            self.image_list = []
            self.val_table = wandb.Table(columns=['image', 'ground_truth', 'prediction'])

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            """Called when the validation batch ends."""
            x, y = batch
            preds, y = outputs
            self.batch_dfs.append(pd.DataFrame({'Ground Truth': y.cpu().numpy(), 'Predictions': preds.cpu().numpy()}))
            x = x.cpu().numpy().transpose(0, 2, 3, 1)
            for x_i, y_i, y_pred in list(zip(x, y, preds)):  # Append validation predictions and ground truth to log in confusion matrix
                self.image_list.append(wandb.Image(x_i, caption=f'Ground Truth: {y_i} - Prediction: {y_pred}'))
                self.val_table.add_data(wandb.Image(x_i), y_i, y_pred)

        def on_validation_epoch_end(self, trainer, pl_module):
            class_names = _trainer.datamodule.nature_dataset.class_names  # Add wandb.Image to a table to log at the end of validation
            val_df = pd.concat(self.batch_dfs)
            wandb.log({'validation_table': self.val_table, 'images_over_time': self.image_list, 'validation_conf_matrix': wandb.plot.confusion_matrix(y_true=val_df['Ground Truth'].tolist(), preds=val_df['Predictions'].tolist(), class_names=class_names)}, step=_trainer.global_step)
            del self.batch_dfs
            del self.val_table  # Collect statistics for whole validation set and log

    return (LogPredictionsCallback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏋️‍ Main Training Loop
    """)
    return


@app.cell
def _(
    ENTITY,
    LogPredictionsCallback,
    NatureDatasetModule,
    NatureLitModule,
    PROJECT_NAME,
    wandb,
):
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch import Trainer
    wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='training', config={'model_name': 'squeezenet', 'batch_size': 16})
    _wandb_logger = WandbLogger(log_model='all', checkpoint_name=f'nature-{wandb.run.id}')
    _log_predictions_callback = LogPredictionsCallback()
    _checkpoint_callback = ModelCheckpoint(every_n_epochs=1)
    _model = NatureLitModule(model_name=wandb.config['model_name'])
    _nature_module = NatureDatasetModule(wandb_run=_wandb_logger.experiment, artifact_name_alias='Nature_100:latest', local_target_dir='Nature_100:latest', batch_size=wandb.config['batch_size'], input_size=_model.input_size)
    _nature_module.setup()
    _trainer = Trainer(logger=_wandb_logger, callbacks=[_log_predictions_callback, _checkpoint_callback], max_epochs=5, log_every_n_steps=5)
    _trainer.fit(_model, datamodule=_nature_module)
    wandb.finish()  # Access hyperparameters downstream to instantiate models/datasets  # W&B integration
    return ModelCheckpoint, Trainer, WandbLogger


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Syncing with W&B Offline
    If for some reason, network communication is lost during the course of training, you can always sync progress with `wandb sync`

    The W&B sdk caches all logged data in a local directory `wandb` and when you call `wandb sync`, this syncs the your local state with the web app.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Retrieve a model checkpoint artifact and resume training
    - Artifacts make it easy to track state of your training remotely and then resume training from a checkpoint
    """)
    return


@app.cell
def _():
    #@title Enter which checkpoint you want to resume training from:

    # FORM VARIABLES
    ARTIFACT_NAME_ALIAS = "nature-oyxk79m1:v4" #@param {type:"string"}
    return (ARTIFACT_NAME_ALIAS,)


@app.cell
def _(
    ARTIFACT_NAME_ALIAS,
    ENTITY,
    LogPredictionsCallback,
    ModelCheckpoint,
    NatureDatasetModule,
    NatureLitModule,
    PROJECT_NAME,
    Trainer,
    WandbLogger,
    wandb,
):
    wandb.init(project=PROJECT_NAME, entity=ENTITY, job_type='resume_training')
    model_chkpt_art = wandb.use_artifact(f'{ENTITY}/{PROJECT_NAME}/{ARTIFACT_NAME_ALIAS}')
    model_chkpt_art.download()
    logging_run = model_chkpt_art.logged_by()
    # Retrieve model checkpoint artifact and restore previous hyperparameters
    wandb.config = logging_run.config
    artifact_name = ARTIFACT_NAME_ALIAS.split(':')[0]  # Can change download directory by adding `root`, defaults to "./artifacts"
    _wandb_logger = WandbLogger(log_model='all', checkpoint_name=artifact_name)
    _log_predictions_callback = LogPredictionsCallback()
    _checkpoint_callback = ModelCheckpoint(every_n_epochs=1)
    # Can create a new artifact name or continue logging to the old one
    _model = NatureLitModule.load_from_checkpoint(f'./artifacts/{ARTIFACT_NAME_ALIAS}/model.ckpt')
    _nature_module = NatureDatasetModule(wandb_run=_wandb_logger.experiment, artifact_name_alias='Nature_100:latest', local_target_dir='Nature_100:latest', batch_size=wandb.config['batch_size'], input_size=_model.input_size)
    _nature_module.setup()
    _trainer = Trainer(logger=_wandb_logger, callbacks=[_log_predictions_callback, _checkpoint_callback], max_epochs=10, log_every_n_steps=5)
    _trainer.fit(_model, datamodule=_nature_module)
    wandb.finish()  # W&B integration
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Registry
    After logging a bunch of checkpoints across multiple runs during experimentation, now comes time to hand-off the best checkpoint to the next stage of the workflow (e.g. testing, deployment).

    The model registry offers a centralized place to house the best checkpoints for all your model tasks. Any `model` artifact you log can be "linked" to a Registered Model. Here are the steps to start using the model registry for more organized model management:
    1. Access your team's model registry by going the team page and selecting `Model Registry`
    ![model registry](https://drive.google.com/uc?export=view&id=1ZtJwBsFWPTm4Sg5w8vHhRpvDSeQPwsKw)

    2. Create a new Registered Model.
    ![model registry](https://drive.google.com/uc?export=view&id=1RuayTZHNE0LJCxt1t0l6-2zjwiV4aDXe)

    3. Go to the artifacts tab of the project that holds all your model checkpoints
    ![model registry](https://drive.google.com/uc?export=view&id=1r_jlhhtcU3as8VwQ-4oAntd8YtTwElFB)

    4. Click "Link to Registry" for the model artifact version you want. (Alternatively you can [link a model via api](https://docs.wandb.ai/guides/models) with `wandb.run.link_artifact`)

    **A note on linking:** The process of linking a model checkpoint is akin to "bookmarking" it. Each time you link a new model artifact to a Registered Model, this increments the version of the Registered Model. This helps delineate the model development side of the workflow from the model deployment/consumption side. The globally understood version/alias of a model should be unpolluted from all the experimental versions being generated in R&D and thus the versioning of a Registered Model increments according to new "bookmarked" models as opposed to model checkpoint logging.

    ### Create a Centralized Hub for all your models
    - Add a model card, tags, slack notifactions to your Registered Model
    - Change aliases to reflect when models move through different phases
    - Embed the model registry in reports for model documentation and regression reports. See this report as an [example](https://api.wandb.ai/links/wandb-smle/r82bj9at)
    ![model registry](https://drive.google.com/uc?export=view&id=1lKPgaw-Ak4WK_91aBMcLvUMJL6pDQpgO)
    """)
    return


if __name__ == "__main__":
    app.run()
