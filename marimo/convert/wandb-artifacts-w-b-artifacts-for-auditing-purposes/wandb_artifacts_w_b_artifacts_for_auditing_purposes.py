# /// script
# dependencies = ["awscli", "boto3", "six", "timm", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/W&B_artifacts_for_auditing_purposes.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{audit-artifacts-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{audit-artifacts-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [Weights & Biases](https://wandb.ai/site) makes running collaborative machine learning projects a breeze. You can focus on what you're trying to experiment with, and W&B will take on the burden of keeping track of everything. If you want to review a loss plot, download the latest model for production, or just see which configurations produced a certain model, W&B is your friend. There's also a bunch of features to help you and your team collaborate, like having a shared dashboard and sharing interactive reports.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # How Weights and Biases can help you with Audits and Regulatory Guidelines

    This notebook accompanies and implements a
    [blog post](http://wandb.me/audit-artifacts-report)
    on using W&B Artifacts to help teams in regulation-heavy industries share their Machine Learning models with clients.

    Run the cells below to train an image classifier and upload the model checkpoints as W&B Artifacts. Then you can reliably know which models you've given to your clients and happily share this information with any regulators.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Please make sure that you set CUDA device before running the following colab. This can be done by changing `Runtime Type` to use GPU hardware accelerator.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: awscli six !pip install awscli --ignore-installed six
    # packages added via marimo's package management: timm wandb boto3 !pip install timm wandb boto3
    return


@app.cell
def _(subprocess):
    # install packages and prepare dataset
    #! wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz -q
    subprocess.call(['wget', 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz', '-q'])
    #! tar -xf imagenette2-160.tgz
    subprocess.call(['tar', '-xf', 'imagenette2-160.tgz'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ✍️ Login to W&B
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


@app.cell
def _():
    import timm
    import boto3
    import torch
    import operator
    import os
    import logging
    import warnings
    import tempfile
    import torchvision
    import torch.nn as nn
    from tqdm import tqdm
    from torchvision import transforms
    from timm.utils.log import setup_default_logging

    return (
        boto3,
        logging,
        nn,
        operator,
        os,
        setup_default_logging,
        tempfile,
        timm,
        torch,
        torchvision,
        tqdm,
        transforms,
        warnings,
    )


@app.cell
def _(logging):
    _logger = logging.getLogger('TrainEval')
    return


@app.cell
def _(transforms):
    Config = dict(
        PROJECT='artifacts',
        DATA_DIR="./imagenette2-160",
        TRAIN_DATA_DIR="./imagenette2-160/train",
        TEST_DATA_DIR="./imagenette2-160/val",
        DEVICE="cuda",
        MODEL="efficientnet_b3",
        PRETRAINED=False,
        LR=3e-4,
        EPOCHS=3,
        IMG_SIZE=160,
        FILENAME='checkpoint-1.pth.tar',
        ALIAS='v0',
        BS=96,
        TRAIN_AUG=transforms.Compose(
            [
                transforms.RandomCrop(160),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        TEST_AUG=transforms.Compose(
            [
                transforms.CenterCrop(160),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        ),
        NUM_CHECKPOINTS=2,
        BUCKET='test-bucket-wandb'
    )
    return (Config,)


@app.cell
def _(torch):
    assert torch.cuda.is_available()
    DEVICE = torch.device('cuda')
    return (DEVICE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏋️‍♀️ Model Training and Evaluation
    """)
    return


@app.cell
def _(DEVICE, nn, tqdm):
    def train_fn(model, train_data_loader, optimizer, epoch):
        model.train()
        fin_loss = 0.0
        tk = tqdm(train_data_loader, desc="Epoch" + " [TRAIN] " + str(epoch + 1))

        for t, data in enumerate(tk):
            data[0] = data[0].to(DEVICE)
            data[1] = data[1].to(DEVICE)

            optimizer.zero_grad()
            out = model(data[0])
            loss = nn.CrossEntropyLoss()(out, data[1])
            loss.backward()
            optimizer.step()

            fin_loss += loss.item()
            tk.set_postfix(
                {
                    "loss": "%.6f" % float(fin_loss / (t + 1)),
                    "LR": optimizer.param_groups[0]["lr"],
                }
            )
        return fin_loss / len(train_data_loader), optimizer.param_groups[0]["lr"]

    return (train_fn,)


@app.cell
def _(DEVICE, nn, torch, tqdm):
    def eval_fn(model, eval_data_loader, epoch):
        model.eval()
        fin_loss = 0.0
        tk = tqdm(eval_data_loader, desc="Epoch" + " [VALID] " + str(epoch + 1))

        with torch.no_grad():
            for t, data in enumerate(tk):
                data[0] = data[0].to(DEVICE)
                data[1] = data[1].to(DEVICE)
                out = model(data[0])
                loss = nn.CrossEntropyLoss()(out, data[1])
                fin_loss += loss.item()
                tk.set_postfix({"loss": "%.6f" % float(fin_loss / (t + 1))})
            return fin_loss / len(eval_data_loader)

    return (eval_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🏁 Checkpoint Saver

    Track top-n training checkpoints and maintain recovery checkpoints on specified intervals.
    Hacked together by / Copyright 2020 Ross Wightman

    This script has been adapted from `pytorch-image-models` checkpoint saver script
    written by Ross Wightman.
    This script adds Weights and Biases artifact integration on top.
    (https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/checkpoint_saver.py)
    """)
    return


@app.cell
def _(Config, operator, os, torch, wandb):
    class CheckpointSaver:
        def __init__(
                self,
                model,
                optimizer,
                config=None,
                checkpoint_prefix='checkpoint',
                checkpoint_dir='',
                decreasing=False,
                max_history=2,
                wandb_run=None):

            # wandb run
            self.wandb_run = wandb_run if wandb_run is not None else wandb.init(job_type='model-artifact')

            # objects to save state_dicts of
            self.model = model
            self.optimizer = optimizer
            self.config = config

            # state
            self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
            self.best_epoch = None
            self.best_metric = None
            self.curr_recovery_file = ''
            self.last_recovery_file = ''

            # config
            self.checkpoint_dir = checkpoint_dir
            self.save_prefix = checkpoint_prefix
            self.extension = '.pth.tar'
            self.decreasing = decreasing  # a lower metric is better if True
            self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
            self.max_history = max_history
            assert self.max_history >= 1

        def save_checkpoint(self, epoch, metric=None):
            assert epoch >= 0
            tmp_save_path = os.path.join(self.checkpoint_dir, 'tmp' + self.extension)
            last_save_path = os.path.join(self.checkpoint_dir, 'last' + self.extension)
            self._save(tmp_save_path, epoch, metric)
            if os.path.exists(last_save_path):
                os.unlink(last_save_path)  # required for Windows support.
            os.rename(tmp_save_path, last_save_path)
            worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
            if (len(self.checkpoint_files) < self.max_history
                    or metric is None or self.cmp(metric, worst_file[1])):
                if len(self.checkpoint_files) >= self.max_history:
                    self._cleanup_checkpoints(1)
                filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
                save_path = os.path.join(self.checkpoint_dir, filename)
                os.link(last_save_path, save_path)
                self.log_artifact(filename, save_path)
                self.checkpoint_files.append((save_path, metric))
                self.checkpoint_files = sorted(
                    self.checkpoint_files, key=lambda x: x[1],
                    reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

                checkpoints_str = "Current checkpoints:\n"
                for c in self.checkpoint_files:
                    checkpoints_str += ' {}\n'.format(c)
                _logger.info(checkpoints_str)

                if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                    self.best_epoch = epoch
                    self.best_metric = metric
                    best_save_path = os.path.join(self.checkpoint_dir, 'model_best' + self.extension)
                    if os.path.exists(best_save_path):
                        os.unlink(best_save_path)
                    os.link(last_save_path, best_save_path)

            return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

        def _save(self, save_path, epoch, metric=None):
            save_state = {
                'epoch': epoch,
                'arch': type(self.model).__name__.lower(),
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            if metric is not None:
                save_state['metric'] = metric
            torch.save(save_state, save_path)

        def _cleanup_checkpoints(self, trim=0):
            trim = min(len(self.checkpoint_files), trim)
            delete_index = self.max_history - trim
            if delete_index < 0 or len(self.checkpoint_files) <= delete_index:
                return
            to_delete = self.checkpoint_files[delete_index:]
            for d in to_delete:
                try:
                    _logger.debug("Cleaning checkpoint: {}".format(d))
                    # Optionally, only keep top N artifacts in W&B.
                    # self.delete_artifact(os.path.basename(d[0]))
                    os.remove(d[0])
                except Exception as e:
                    _logger.error("Exception '{}' while deleting checkpoint".format(e))
            self.checkpoint_files = self.checkpoint_files[:delete_index]

        def log_artifact(self, filename, save_path):
            try: 
                artifact = wandb.Artifact(filename, type='model')
                artifact.add_file(save_path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                _logger.error("Exception '{}' while logging wandb artifact".format(e))

        def delete_artifact(self, filename, alias='v0'):
            api = wandb.Api()
            artifact = api.artifact(f'{Config["PROJECT"]}/{filename}:{alias}')
            try: 
                artifact.delete(delete_aliases=True)
            except Exception as e:
                _logger.error("Exception '{}' while deleting wandb artifact {}".format(e, filename))

    return (CheckpointSaver,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 💡Bring it all together!
    """)
    return


@app.cell
def _(
    CheckpointSaver,
    Config,
    eval_fn,
    timm,
    torch,
    torchvision,
    train_fn,
    wandb,
):
    def main(wandb_run=None):
        # train and eval datasets
        train_dataset = torchvision.datasets.ImageFolder(
            Config["TRAIN_DATA_DIR"], transform=Config["TRAIN_AUG"]
        )
        eval_dataset = torchvision.datasets.ImageFolder(
            Config["TEST_DATA_DIR"], transform=Config["TEST_AUG"]
        )

        # train and eval dataloaders
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=Config["BS"],
            shuffle=True,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=Config["BS"],
        )

        # model
        model = timm.create_model(Config["MODEL"], pretrained=Config["PRETRAINED"])
        model = model.cuda()

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=Config["LR"])

        # setup checkpoint saver
        saver = CheckpointSaver(model=model, optimizer=optimizer, config=Config, decreasing=True, 
                            wandb_run=wandb_run, max_history=Config['NUM_CHECKPOINTS'])

        for epoch in range(Config["EPOCHS"]):
            avg_loss_train, lr = train_fn(
                model, train_dataloader, optimizer, epoch
            )
            avg_loss_eval = eval_fn(model, eval_dataloader, epoch)
            wandb.run.log({
                "epoch": epoch, 
                "learning rate": lr, 
                "train loss": avg_loss_train, 
                "evaluation loss": avg_loss_eval
                })
            saver.save_checkpoint(epoch, metric=avg_loss_eval)

    return (main,)


@app.cell
def _(setup_default_logging):
    setup_default_logging()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train Model and Log Artifacts to W&B
    """)
    return


@app.cell
def _(Config, main, wandb):
    run = wandb.init(project=Config['PROJECT'], config=Config)
    wandb.config = Config
    main(wandb_run=run)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upload Artifact to S3
    """)
    return


@app.cell
def _(subprocess):
    # Setup AWSCLI https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html
    #! aws configure
    subprocess.call(['aws', 'configure'])
    return


@app.cell
def _(boto3, wandb):
    s3 = boto3.client('s3')
    api = wandb.Api()
    return api, s3


@app.cell
def _(Config, api, boto3, os, s3, tempfile, warnings):
    def upload_artifact_to_s3(config):
        artifact = api.artifact(f"{config['PROJECT']}/{config['FILENAME']}:{config['ALIAS']}")
        digest = artifact.digest

        with tempfile.TemporaryDirectory() as tmpdir:
            path = artifact.download(tmpdir)
            fname = os.listdir(path)[0]
            fpath = path + '/' + fname

            _logger.info(f"Downloaded artifact {fname} to {fpath} locally.")

            try: 
                metadata = s3.head_object(Bucket=Config['BUCKET'], Key=fname)['Metadata']
            except: 
                warnings.warn(f"""File {fname} does not already exist in Bucket {Config['BUCKET']} on AWS.\
                Cleaning up AWS bucket for any existing files, and uploading new \
                artifact.""")
                bucket = boto3.resource('s3').Bucket(Config['BUCKET'])
                bucket.objects.all().delete()
                metadata = {'digest': -1}
        
            # upload files to S3 if digests are different 
            if metadata['digest']!=digest:
                s3.upload_file(fpath, Config['BUCKET'], fname, ExtraArgs={"Metadata": {"digest": digest}})
            else: 
                _logger.info(f"File {fname} already exists in Bucket {Config['BUCKET']} on AWS with same digest. Nothing to do.")

    return (upload_artifact_to_s3,)


@app.cell
def _(Config, upload_artifact_to_s3):
    upload_artifact_to_s3(config=Config)
    return


if __name__ == "__main__":
    app.run()
