from fastai.vision.all import *
import wandb
from fastai.callback.wandb import *

# load camvid dataset
path = untar_data(URLs.CAMVID_TINY)
codes = np.loadtxt(path/'codes.txt', dtype=str)
fnames = get_image_files(path/"images")

# get label from an input file
def label_func(fn): return path/"labels"/f"{fn.stem}_P{fn.suffix}"

# create dataloaders
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = fnames, label_func = label_func, codes = codes
)

# start a run
wandb.init(project='fastai-segmentation')

# create a learner and log dataset & model
learn = unet_learner(dls, resnet18, cbs=[WandbCallback(log_model=True, log_dataset=True), SaveModelCallback()])

# train
learn.fine_tune(3)