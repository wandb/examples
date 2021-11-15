from fastai.tabular.all import *
import wandb
from fastai.callback.wandb import *

# create dataloaders
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
dls = TabularDataLoaders.from_csv(path/'adult.csv', path=path, y_names="salary",
    cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race'],
    cont_names = ['age', 'fnlwgt', 'education-num'],
    procs = [Categorify, FillMissing, Normalize])

# start a run
wandb.init(project='fastai-tabular')

# create a learner and train
learn = tabular_learner(dls, metrics=accuracy, cbs=[WandbCallback()])
learn.fit(2)