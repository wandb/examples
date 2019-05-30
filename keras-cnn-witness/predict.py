#!/usr/bin/env python3

from PIL import Image, ImageChops, ImageStat
import os
import numpy as np
import shutil
from keras.models import load_model
import keras_metrics as km
from keras import metrics
from data import adjust_image
from keras.preprocessing.image import img_to_array, array_to_img

custom_objects = {
    'binary_precision': km.precision(),
    'binary_recall': km.recall(),
    'loss': metrics.mean_squared_error
}

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

cur_path = os.path.dirname(__file__)

# cache predictions -- remove to clear
# shutil.rmtree(os.path.join(cur_path, 'predict'), ignore_errors=True)

shutil.rmtree(os.path.join(cur_path, 'debug'), ignore_errors=True)

model = load_model('model/unet-witness-4tier.hdf5', custom_objects=custom_objects)

orig_size = (1280, 720)
final_size = (512, 288)

setnames = ['valid', 'train']

for setname in setnames:
    data = []
    os.makedirs(os.path.join(cur_path, 'predict', setname), exist_ok=True)
    os.makedirs(os.path.join(cur_path, 'debug', setname), exist_ok=True)
    images_path = os.path.join(cur_path, 'data', setname, 'images')
    fnames = sorted(os.listdir(images_path))
    for fname in fnames:
        pngname = fname.replace('.jpg', '.png')
        source_path = 'data/{}/images/{}'.format(setname, fname)
        label_path = 'data/{}/labels/{}'.format(setname, pngname)
        dest_path = 'predict/{}/{}'.format(setname, pngname)
        source = (Image
            .open(source_path)
            .convert('RGB')
            .resize(final_size))
        label = (Image
            .open(label_path)
            .convert('L')
            .resize(final_size, Image.NEAREST))

        if os.path.exists(dest_path):
            prediction = Image.open(dest_path)
        else:
            source_arr = adjust_image(img_to_array(source))
            input_arr = source_arr[np.newaxis, ...]
            prediction_arr = model.predict([input_arr])[0]
            prediction = array_to_img(prediction_arr, scale=True)
            prediction.save(dest_path, 'png')

        debug_path = 'debug/{}/{}'.format(setname, fname)
        width, height = final_size
        debug = Image.new('RGB', (width * 2, height * 2))
        
        mask = Image.new('L', (width, height), 200)

        intersection = ImageChops.multiply(prediction, label)
        prediction_vs_label = Image.merge('RGB',
            [prediction, intersection, label])

        debug.paste(source, (0, 0))
        debug.paste(source, (width, 0))
        debug.paste(prediction_vs_label, (width, 0), mask)
        debug.paste(prediction, (0, height))
        debug.paste(label, (width, height), mask)
        debug.save(debug_path, 'jpeg')

        total = width * height
        diff = ImageChops.difference(prediction, label)
        sum_difference = ImageStat.Stat(diff).sum[0]
        sum_prediction = ImageStat.Stat(prediction).sum[0]
        sum_label = ImageStat.Stat(label).sum[0]
        sum_correct = ImageStat.Stat(intersection).sum[0]


        recall = sum_correct / sum_label if sum_label > 0 else 0
        precision = sum_correct / sum_prediction if sum_prediction > 0 else 0
        pct_error = sum_difference / float(total)
        pct_label = sum_label / float(total)
        pct_prediction = sum_prediction / float(total)
        pct_correct = sum_correct / float(total)

        data.append({
            'fname': fname,
            'label': pct_label,
            'prediction': pct_prediction,
            'correct': pct_correct,
            'error': pct_error,
            'recall': recall,
            'precision': precision
        })

    print(setname.upper())
    for item in reversed(sorted(data, key=lambda i: i['error'])):
        print('{fname}: label {label:03f}, predict {prediction:03f}, correct {correct:03f}, error {error:03f}, recall {recall:03f}, precision {precision:03f}'.format(**item))

