from fastai import *
from fastai.vision import *

import wandb
from wandb.fastai import WandbCallback
wandb.init(project="fastai-food-101", entity="wandb-examples")

path = untar_data("https://s3.amazonaws.com/fast-ai-imageclas/food-101")

tfms = get_transforms()
data = ImageDataBunch.from_folder(path, train='images', valid_pct = 0.25, bs = 128, ds_tfms = tfms, size = 224)

learn = cnn_learner(data, models.densenet121, metrics=[accuracy, top_k_accuracy], callback_fns=partial(WandbCallback, input_type='images')).mixup().to_fp16()

learn.fit_one_cycle(3)

learn.unfreeze()
learn.fit_one_cycle(10)

learn.save('stage-1')

data = data = ImageDataBunch.from_folder(path, train='images', valid_pct = 0.25, ds_tfms = tfms, size = 512)

learn.load('stage-1')
learn.unfreeze()
learn.fit_one_cycle(10)
