from fastai.vision import *
import wandb
from wandb.fastai import WandbCallback
import pathlib
import requests
import tarfile
import random
from functools import partial

# Define default hyper-parameters
model = models.resnet34     # we want to find the best resnet34 config
hyperparameter_defaults = dict(
    img_size = 64,
    batch_size = 16,
    epochs = 20,
    encoder = model.__name__,
    pretrained = False,  # use pre-trained model and train only last layers
    dropout = 0.5,
    one_cycle = False,  # "1cycle" policy -> https://arxiv.org/abs/1803.09820
    learning_rate = 1e-3
)

# Initialize W&B project
wandb.init(config=hyperparameter_defaults)
config = wandb.config

# Download data
PATH_DATA = pathlib.Path('data/simpsons')
if not (PATH_DATA).exists():
    PATH_DATAFILE = pathlib.Path('simpsons.tar.gz')
    URL_DATA = 'https://storage.googleapis.com/wandb-production.appspot.com/mlclass/simpsons.tar.gz'
    r = requests.get(URL_DATA)
    PATH_DATAFILE.open("wb").write(r.content)
    with tarfile.open(PATH_DATAFILE) as archive:
        archive.extractall('data')
    PATH_DATAFILE.unlink()

# Load data
data = (ImageList.from_folder(PATH_DATA)
                 .split_by_folder(train='train', valid='test')
                 .label_from_folder()
                 .transform(get_transforms(), size=config.img_size)
                 .databunch(bs=config.batch_size).normalize())

# Create model
learn = cnn_learner(data,
                    model,
                    pretrained=config.pretrained,
                    ps=config.dropout,
                    metrics=accuracy,
                    callback_fns=partial(WandbCallback, input_type='images'))  # Log training in W&B

# Train
if config.one_cycle:
    learn.fit(config.epochs, lr=config.learning_rate)
else:
    learn.fit_one_cycle(config.epochs, max_lr=config.learning_rate)
