#!/usr/bin/env python3

from PIL import Image, ImageEnhance
import os
import math
import numpy as np
import shutil
from keras.models import load_model, Model
import keras_metrics as km
from keras import metrics
from data import adjust_image
from keras.preprocessing.image import img_to_array, array_to_img

custom_objects = {
    'binary_precision': km.precision(),
    'binary_recall': km.recall(),
    'loss': metrics.mean_squared_error
}

cur_path = os.path.dirname(__file__)

root_folder = 'network-visualizations/3tier'
shutil.rmtree(os.path.join(cur_path, root_folder), ignore_errors=True)

model = load_model('model/unet-witness-3tier-64.hdf5', custom_objects=custom_objects)

outputs = [l.output for l in model.layers[1:]]
intermediate = Model(inputs=model.input, outputs=outputs)
for i, layer in enumerate(model.layers[1:]):
    print(i, layer.__class__.__name__, tuple(layer.output.shape.as_list()))

orig_size = (1280, 720)
final_size = (512, 288)

setnames = ['valid']
# setnames = ['valid', 'train']

for setname in setnames:
    images_path = os.path.join(cur_path, 'data', setname, 'images')
    fnames = sorted(os.listdir(images_path))

    f_sources = []
    f_intermediates = []

    for fname in fnames:
        fnum, fext = fname.split('.')
        fpath = '{}/example-{}'.format(root_folder, fnum)
        os.makedirs(fpath, exist_ok=True)
        source_path = 'data/{}/images/{}'.format(setname, fname)
        source_img = (Image
            .open(source_path)
            .convert('RGB')
            .resize(final_size))

        source_arr = adjust_image(img_to_array(source_img))
        input_arr = source_arr[np.newaxis, ...]

        intermediate_arrs = intermediate.predict([input_arr])
        f_intermediates.append(intermediate_arrs)
        f_sources.append(source_img)

        for i, layer_arr in enumerate(intermediate_arrs):
            num_neurons = layer_arr.shape[-1]
            neuron_h, neuron_w = layer_arr.shape[1:-1]
            num_cols = int(num_neurons ** 0.5)
            num_rows = math.ceil(num_neurons / num_cols)
            layer_h = neuron_h * num_rows
            layer_w = neuron_w * num_cols
            
            layer_img = Image.new('L', (layer_w, layer_h), 0)
            color_img = Image.new('RGB', (layer_w, layer_h), 0)

            source_sized = source_img.resize((neuron_w, neuron_h))
            source_enhancer = ImageEnhance.Brightness(source_sized)
            source_sized = source_enhancer.enhance(1.8)

            for j in range(num_neurons):
                neuron_arr = layer_arr[..., j]
                neuron_reshaped = neuron_arr.reshape((
                    neuron_arr.shape[1],
                    neuron_arr.shape[2],
                    1))
                neuron_img = array_to_img(neuron_reshaped, scale=True)
                neuron_row = int(j / num_cols)
                neuron_col = j - (neuron_row * num_cols)
                neuron_x = neuron_col * neuron_w
                neuron_y = neuron_row * neuron_h
                layer_img.paste(neuron_img, (neuron_x, neuron_y))
                color_img.paste(source_sized, (neuron_x, neuron_y), neuron_img)

            layer_path = '{}/layer-{:02d}-activation.jpg'.format(fpath, i)
            layer_img.save(layer_path)
            color_path = '{}/layer-{:02d}-colored.jpg'.format(fpath, i)
            color_img.save(color_path)

    for layer_index in range(len(outputs)):
        lpath = '{}/layer-{:02d}'.format(root_folder, layer_index)
        os.makedirs(lpath, exist_ok=True)
        layer_arrs = [i[layer_index] for i in f_intermediates]
        num_neurons = layer_arrs[0].shape[-1]
        example_h, example_w = layer_arrs[0].shape[1:-1]
        num_examples = len(fnames)
        num_cols = int(num_examples ** 0.5)
        num_rows = math.ceil(num_examples / num_cols)
        neuron_h = example_h * num_rows
        neuron_w = example_w * num_cols

        for i in range(num_neurons):
            neuron_img = Image.new('L', (neuron_w, neuron_h), 0)
            color_img = Image.new('RGB', (neuron_w, neuron_h), 0)
            neuron_arrs = [l[..., i] for l in layer_arrs]

            for j, source_img in enumerate(f_sources):
                source_sized = source_img.resize((example_w, example_h))
                source_enhancer = ImageEnhance.Brightness(source_sized)
                source_sized = source_enhancer.enhance(1.8)

                neuron_arr = neuron_arrs[j]
                neuron_reshaped = neuron_arr.reshape((
                    neuron_arr.shape[1],
                    neuron_arr.shape[2],
                    1))
                example_img = array_to_img(neuron_reshaped, scale=True)
                example_row = int(j / num_cols)
                example_col = j - (example_row * num_cols)
                example_x = example_col * example_w
                example_y = example_row * example_h
                neuron_img.paste(example_img, (example_x, example_y))
                color_img.paste(source_sized, (example_x, example_y), 
                    example_img)

            layer_path = '{}/cell-{:03d}-activation.jpg'.format(lpath, i)
            neuron_img.save(layer_path)
            color_path = '{}/cell-{:03d}-colored.jpg'.format(lpath, i)
            color_img.save(color_path)
