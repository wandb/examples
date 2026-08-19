# /// script
# dependencies = ["wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/tensorflow/convert_imagenette_tfrecord.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell
def _(subprocess):
    # packages added via marimo's package management: wandb !pip install -q wandb
    #! wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
    subprocess.call(['wget', 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'])
    #! tar -xzf imagenette2-320.tgz
    subprocess.call(['tar', '-xzf', 'imagenette2-320.tgz'])
    #! rm imagenette2-320.tgz
    subprocess.call(['rm', 'imagenette2-320.tgz'])
    return


@app.cell
def _():
    import os
    import cv2
    import math
    import wandb
    import random
    import numpy as np
    from glob import glob
    from PIL import Image
    import tensorflow as tf
    from tqdm.auto import tqdm
    import matplotlib.pyplot as plt

    return Image, glob, math, os, plt, random, tf, tqdm, wandb


@app.cell
def _():
    LABEL_DICT = {
        "n01440764": ["tench", 0],
        "n02102040": ["english_springer", 1],
        "n02979186": ["cassette_player", 2],
        "n03000684": ["chain_saw", 3],
        "n03028079": ["church", 4],
        "n03394916": ["french_horn", 5],
        "n03417042": ["grabage_truck", 6],
        "n03425413": ["gas_pump", 7],
        "n03445777": ["golf_ball", 8],
        "n03888257": ["parachute", 9]
    }
    return (LABEL_DICT,)


@app.cell
def _(wandb):
    wandb.init(
        project="simple-training-loop",
        entity="jax-series",
        job_type="tfrecord"
    )
    return


@app.cell
def _(LABEL_DICT, tf):
    def create_example(image_file, label):
        feature = {
            "image": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[tf.io.read_file(image_file).numpy()]
                )
            ),
            "label": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[LABEL_DICT[label][1]])
            ),
            "label_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(
                    value=[LABEL_DICT[label][0].encode('utf8')]
                )
            )
        }
        return tf.train.Example(
            features=tf.train.Features(feature=feature)
        )

    return (create_example,)


@app.cell
def _(glob, os, random):
    train_images = glob(os.path.join("imagenette2-320", "train/*/*.JPEG"))
    random.shuffle(train_images)
    train_labels = [img.split("/")[-2] for img in train_images]

    val_images = glob(os.path.join("imagenette2-320", "val/*/*.JPEG"))
    random.shuffle(val_images)
    val_labels = [img.split("/")[-2] for img in val_images]
    return train_images, train_labels, val_images, val_labels


@app.cell
def _(Image, LABEL_DICT, create_example, math, os, tf, tqdm, wandb):
    def chunkify(input_list, chunk_size):
        chunk_size = max(1, chunk_size)
        return [
            input_list[i: i + chunk_size]
            for i in range(0, len(input_list), chunk_size)
        ]


    def create_tfrecords(images, labels, max_chunk_size: int, dump_dir: str):
        os.makedirs(dump_dir)
        num_chunks = math.ceil(len(images) / max_chunk_size)
        print("Total number of image-label pairs:", len(images))
        print("Total number of image-label pair chunks:", num_chunks)
        image_chunks = chunkify(images, max_chunk_size)
        label_chunks = chunkify(labels, max_chunk_size)
        table = wandb.Table(columns=[
            "Image", "Label-Name", "Label-ID", "Split-Name", "Chunk-ID"
        ])
        for idx in range(num_chunks):
            image_chunk = image_chunks[idx]
            label_chunk = label_chunks[idx]
            current_chunk_size = len(image_chunk)
            file_name = "%.2i-%.3i.tfrec" % (idx + 1, current_chunk_size)
            tfrecord_file = os.path.join(dump_dir, file_name)
            writer = tf.io.TFRecordWriter(tfrecord_file)
            progress_bar = tqdm(
                range(current_chunk_size),
                desc=f"Writing {file_name}"
            )
            for chunk_idx in progress_bar:
                image = Image.open(image_chunk[chunk_idx])
                table.add_data(
                    wandb.Image(image),
                    LABEL_DICT[label_chunk[chunk_idx]][0],
                    LABEL_DICT[label_chunk[chunk_idx]][1],
                    dump_dir.split("/")[-1],
                    idx
                )
                example = create_example(
                    image_chunk[chunk_idx], label_chunk[chunk_idx]
                )
                writer.write(example.SerializeToString())
            writer.close()
        return table

    return (create_tfrecords,)


@app.cell
def _(create_tfrecords, train_images, train_labels, val_images, val_labels):
    print("Creating TFRecords for train data...")
    train_table = create_tfrecords(
        train_images,
        train_labels,
        max_chunk_size=512,
        dump_dir="tfrecords/train"
    )

    print("Creating TFRecords for validation data...")
    val_table = create_tfrecords(
        val_images,
        val_labels,
        max_chunk_size=512,
        dump_dir="tfrecords/val"
    )
    return train_table, val_table


@app.cell
def _(train_table, val_table, wandb):
    wandb.log({"Train-Data": train_table})
    wandb.log({"Validation-Data": val_table})
    return


@app.cell
def _(glob, tf):
    def parse_tfrecord(example):
        example = tf.io.parse_single_example(
            example, {
                "image": tf.io.FixedLenFeature([], tf.string),
                "label": tf.io.VarLenFeature(tf.int64),
                "label_name": tf.io.VarLenFeature(tf.string)
            }
        )
        example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
        example["label"] = tf.sparse.to_dense(example["label"])
        example["label_name"] = tf.sparse.to_dense(example["label_name"])
        return example


    raw_dataset = tf.data.TFRecordDataset(glob("./tfrecords/train/*"))
    parsed_dataset = raw_dataset.map(parse_tfrecord)
    return (parsed_dataset,)


@app.cell
def _(parsed_dataset, plt):
    for features in parsed_dataset.take(1):
        plt.imshow(features["image"].numpy())
        label = features["label"].numpy()
        label_name = features["label_name"].numpy()
        plt.title(f"{label}-{label_name}")
        plt.show()
    return


@app.cell
def _(wandb):
    artifact = wandb.Artifact(
        'imagenette-tfrecords',
        type='dataset',
        metadata={
            "author": "Jeremy Howard",
            "title": "imagenette",
            "url": "https://github.com/fastai/imagenette/",
            "source": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        }
    )
    artifact.add_dir('tfrecords')
    wandb.log_artifact(artifact, aliases=["320px"])
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
