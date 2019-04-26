#!/usr/bin/env python3

from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from PIL import Image
import json
from urllib.parse import urlparse
import os
import os.path
import wandb
import pandas
import random
import numpy
from wandb.fastai import WandbCallback
from fastai.callbacks import LearnerCallback
import boto3

# Seed the randomizer so training/validation set split is consistent
numpy.random.seed(789032)

run = wandb.init(project='witness-puzzle-finder')

encoder = models.resnet18

wandb.config.batch_size = 6
wandb.config.img_size = (360, 240)
wandb.config.encoder = encoder.__name__
wandb.config.learning_rate = 5e-3
wandb.config.weight_decay = 1e-2
wandb.config.num_epochs = 20

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class AwsGroundTruthFetcher():
    """Read an AWS Ground Truth manifest file, and download all the source and
    result files referenced within it to a local staging directory.
    """
    def __init__(self, manifest_s3_url, job_name, working_dir='/tmp/working'):
        # Job name is used in the AWS manifest to prefix key data
        self.job_name = job_name
        # Should be in form s3://bucket/path
        self.manifest_url = urlparse(manifest_s3_url)
        self.working_dir = working_dir
        self.s3 = boto3.client('s3')

    def sync_down(self, s3_url, local_subpath=None):
        "Download s3 file to local file system."
        if not local_subpath:
            # Drop leading slash
            local_subpath = s3_url.path[1:]
        local_path = os.path.join(self.working_dir, local_subpath)
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        if not os.path.exists(local_path):
            print('-  downloading {}'.format(s3_url.path[1:]))
            self.s3.download_file(s3_url.netloc, s3_url.path[1:], local_path)
        return local_path

    def get_manifest_lines(self):
        "Get manifest and return filtered lines."
        manifest_path = os.path.join(self.working_dir, 'manifest.jsonl')
        manifest_path = self.sync_down(self.manifest_url)
        manifest_lines = open(manifest_path).read().split('\n')
        manifest_lines = filter(bool, manifest_lines)
        manifest_lines = map(lambda l: json.loads(l), manifest_lines)
        manifest_lines = filter(lambda l: self.filter_manifest_line(l),
            manifest_lines)
        return manifest_lines

    def filter_manifest_line(self, line):
        "Return whether a manifest line should be included."
        source_ref = line['source-ref']
        job_metadata_key = '{}-ref-metadata'.format(self.job_name)
        job_metadata = line[job_metadata_key]
        if 'failure-reason' in job_metadata:
            print('!  skipping failed item {}: {}'.format(source_ref,
                job_metadata['failure-reason']))
            return False
        if source_ref.endswith('.webp'):
            print('!  skipping webp item {}'.format(source_ref))
            return False
        return True

    def fetch(self):
        "Fetch all items from the manifest and return local paths."
        manifest_lines = self.get_manifest_lines()
        for line in manifest_lines:
            source_url = urlparse(line['source-ref'])
            source_path = self.sync_down(source_url)
            result_key = '{}-ref'.format(self.job_name)
            result_ref = line[result_key]
            result_url = urlparse(result_ref)
            result_path = self.sync_down(result_url)
            yield { 'source': source_path, 'result': result_path }

# Fetch all files from S3.
results_path = 's3://wandb-ss-witness/raw-images/ss-witness'
manifest_path = '{}/manifests/output/output.manifest'.format(results_path)
manifest_fetcher = AwsGroundTruthFetcher(manifest_path, 'ss-witness')
manifest_items = list(manifest_fetcher.fetch())

# codes and image colors from the mask file.
mask_mappings = [
    ('BACKGROUND', 0),
    ('puzzle': 112)
]

# Create a pixel mapping of raw mask values (0-255) to class indices (0-c).
mask_mapping = [0] * 256
for i, (code, val) in enumerate(mask_mappings):
    mask_mapping[val] = i

def witness_mask_xform(img):
    "Convert an image from a raw mask to an class index mask."
    return img.point(mask_mapping)

class WitnessSegmentationLabelList(SegmentationLabelList):
    "Label list that transforms mask values."
    def open(self, fn):
        "Open mask and trasform to class indices."
        return open_mask(fn, after_open=witness_mask_xform)

class WitnessSegmentationItemList(SegmentationItemList):
    "Item List with proper label class."
    _label_cls = WitnessSegmentationLabelList
    _square_show_res = False

# Create training and validation set
y_path_lookup = { i['source']: i['result'] for i in manifest_items }
get_y_fn = lambda x: y_path_lookup[str(x)]
codes = [i[0] for i in mask_mappings]

item_paths = [pathlib.Path(i['source']) for i in manifest_items]
item_list = WitnessSegmentationItemList(item_paths)
item_lists = item_list.split_by_rand_pct(0.2)
dataset = (item_lists
    .label_from_func(get_y_fn, classes=codes)
    .transform(get_transforms(), size=wandb.config.img_size, tfm_y=True))

# Set up data transforms
databunch = (dataset
    .databunch(bs=wandb.config.batch_size, num_workers=0)
    .normalize(imagenet_stats))

def acc(input, target):
    "Accuracy."
    target = target.squeeze(1)
    return (input.argmax(dim=1)==target).float().mean()

def total_flagged(input, target):
    "Total pixels flagged as a puzzle."
    return (input == 1).sum()

metrics=[acc, total_flagged]

def fastaiim2np(im):
    "Convert FastAI image to numpy array."
    thumb_ratio = 0.5  # Scale images down so they can all be sent to W&B.
    thumb_size = (
        int(wandb.config.img_size[0] * thumb_ratio),
        int(wandb.config.img_size[1] * thumb_ratio)
    )
    return (PIL.Image
        .fromarray(image2np(im.data*255).astype(np.uint8))
        .resize(thumb_size, Image.ANTIALIAS))

def gather_image_groups():
    "Convert dataset into image group for sending to W&B."
    for i in range(len(dataset.train)):
        image = dataset.train.x[i]
        label = dataset.train.y[i]
        pred_class, pred_labels, pred_probs = learn.predict(image)
        yield {
            'image': wandb.Image(fastaiim2np(image)),
            'label': wandb.Image(fastaiim2np(label)),
            'prediction': wandb.Image(fastaiim2np(pred_class))
        }

class LogImagesCallback(LearnerCallback):
    "Log images at end of every epoch."
    def on_epoch_end(self, epoch, **kwargs):
        # Log training set with predictions
        images = list(gather_image_groups())
        wandb.log({
            'images': [i['image'] for i in images],
            'labels': [i['label'] for i in images],
            'prediction': [i['prediction'] for i in images]
        })

# Create the learner.
learn = unet_learner(
    databunch,
    arch=encoder,
    metrics=metrics,
    callback_fns=[WandbCallback, LogImagesCallback],
    wd=wandb.config.weight_decay)

# Learn!
learn.fit_one_cycle(
    wandb.config.num_epochs,
    slice(wandb.config.learning_rate),
    pct_start=0.9)

learn.save('stage-1')
