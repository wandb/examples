#!/usr/bin/env python3

from PIL import Image
import os
import shutil

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cur_path = os.path.dirname(__file__)

RAW_DATA_DIR = 'data/all'

paths = [
    'data/train/images',
    'data/train/labels',
    'data/valid/images',
    'data/valid/labels'
]

for path in paths:
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(os.path.join(cur_path, path), exist_ok=True)

orig_size = (1280, 720)
final_size = (512, 288)

valid_percent = 16
valid_mod = int(100 / valid_percent)

for i in range(311):
    source_path = '{}/images/{:03d}.jpg'.format(RAW_DATA_DIR, i)
    label_path = '{}/labels/{:03d}.png'.format(RAW_DATA_DIR, i)
    set_name = 'valid' if i % valid_mod == 0 else 'train'
    sources_dest = os.path.join(cur_path, 'data/{}/images'.format(set_name))
    labels_dest = os.path.join(cur_path, 'data/{}/labels'.format(set_name))
    source = (Image
        .open(source_path)
        .convert('RGB')
        .resize(final_size))
    label = (Image
        .open(label_path)
        .convert('L')
        .resize(final_size, Image.NEAREST))
    source_dest = os.path.join(sources_dest, '{:03d}.jpg'.format(i))
    source.save(source_dest)
    label_dest = os.path.join(labels_dest, '{:03d}.png'.format(i))
    label.save(label_dest)
