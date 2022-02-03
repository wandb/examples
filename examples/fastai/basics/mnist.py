import wandb
from fastai.vision.all import *
from fastai.callback.wandb import *

# build MNIST dataloaders
path = untar_data(URLs.MNIST_SAMPLE)
items = get_image_files(path)
tds = Datasets(items, [PILImageBW.create, [parent_label, Categorize()]], splits=GrandparentSplitter()(items))
dls = tds.dataloaders(bs=32, after_item=[ToTensor(), IntToFloatTensor()])

# start a run
wandb.init(project='fastai-mnist')

# create a learner with gradient accumulation
learn = cnn_learner(dls, resnet18, loss_func=CrossEntropyLossFlat(), cbs=[WandbCallback(), GradientAccumulation(50)])

# train
learn.fit(2)
