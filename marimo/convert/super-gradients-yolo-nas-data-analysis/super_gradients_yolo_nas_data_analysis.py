# /// script
# dependencies = ["pycairo", "roboflow", "super-gradients", "sweeps", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/super-gradients/yolo_nas_data_analysis.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell
def _(subprocess):
    #! sudo apt install libcairo2-dev pkg-config python3-dev -qq
    subprocess.call(['sudo', 'apt', 'install', 'libcairo2-dev', 'pkg-config', 'python3-dev', '-qq'])
    # packages added via marimo's package management: roboflow pycairo wandb sweeps !pip install roboflow pycairo wandb sweeps -qqq
    # packages added via marimo's package management: super_gradients !pip install super_gradients
    return


@app.cell
def _():
    import os
    import glob
    import torch
    import wandb
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from matplotlib import patches
    from google.colab import userdata
    from torchvision.io import read_image
    from torch.utils.data import DataLoader
    from super_gradients.training import models, dataloaders
    from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train, coco_detection_yolo_format_val
    )

    warnings.filterwarnings("ignore")

    os.environ["WANDB_API_KEY"] = userdata.get('wandb')
    os.environ["ROBOFLOW_API_KEY"] = userdata.get('roboflow')
    return (
        coco_detection_yolo_format_train,
        coco_detection_yolo_format_val,
        np,
        os,
        wandb,
    )


@app.cell
def _(os):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("easyhyeon").project("trash-sea")
    dataset = project.version(10).download("yolov5")
    return


@app.cell
def _():
    DATASET_PATH = "/content/trash-sea-10"
    WANDB_PROJECT_NAME = "fconn-yolo-nas"
    ENTITY = "ml-colabs"
    return DATASET_PATH, ENTITY, WANDB_PROJECT_NAME


@app.cell
def _(DATASET_PATH):
    dataset_params = {
        'data_dir':DATASET_PATH,
        'train_images_dir':'train/images',
        'train_labels_dir':'train/labels',
        'val_images_dir':'valid/images',
        'val_labels_dir':'valid/labels',
        'test_images_dir':'test/images',
        'test_labels_dir':'test/labels',
        'classes': ["Buoy", "Can", "Paper", "Plastic Bag", "Plastic Bottle"]
    }
    return (dataset_params,)


@app.cell
def _(
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
    dataset_params,
):
    from IPython.display import clear_output

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes'],
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes'],
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['test_images_dir'],
            'labels_dir': dataset_params['test_labels_dir'],
            'classes': dataset_params['classes'],
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )

    train_data.dataset.transforms = train_data.dataset.transforms[5:]
    return (train_data,)


@app.cell
def _():
    colors = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'yellow',
        4: 'black'
    }
    classes = {
        0:"Buoy",
        1:"Can",
        2:"Paper",
        3:"Plastic Bag",
        4:"Plastic Bottle"
    }
    return (classes,)


@app.cell
def _(ENTITY, WANDB_PROJECT_NAME, classes, train_data, wandb):
    def _process_bounding_boxes_list(annots):
        result = []
        class_count = {i: 0 for i in range(0, 5)}
        for annot_idx, annotation in enumerate(annots):
            class_count[int(annotation[1])] += 1
            result.append({'position': {'middle': [float(annotation[2]), float(annotation[3])], 'width': float(annotation[4]), 'height': float(annotation[5])}, 'domain': 'pixel', 'class_id': int(annotation[1]), 'box_caption': classes[int(annotation[1])]})
        return (result, class_count)

    def populate_wandb_image_samples(train_data):
        wandb.init(project=WANDB_PROJECT_NAME, entity=ENTITY, id='add-image-samples', job_type='add-tables', resume='allow')
        class_set = wandb.Classes([{'name': 'Buoy', 'id': 0}, {'name': 'Can', 'id': 1}, {'name': 'Paper', 'id': 2}, {'name': 'Plastic Bag', 'id': 3}, {'name': 'Plastic Bottle', 'id': 4}])
        table = wandb.Table(columns=['Annotated-Image', 'Number-of-objects', 'Number-Buoy', 'Number-Can', 'Number-Paper', 'Number-Plastic-Bag', 'Number-Plastic-Bottle'])
        img_count = 0
        for batch_idx, batch_sample in enumerate(train_data):
            batch_images = batch_sample[0]
            batch_annotations = batch_sample[1]
            annots_dict = {i: [] for i in range(0, batch_images.shape[0])}
            for annot in batch_annotations:
                annots_dict[int(annot[0])].append(annot)
            for idx, image in enumerate(batch_images):
                bbox, class_count = _process_bounding_boxes_list(annots_dict[idx])
                image = image.flip(0)
                img = wandb.Image(image, boxes={'ground_truth': {'box_data': bbox, 'class_labels': classes}}, classes=class_set)
                table.add_data(img, len(bbox), class_count[0], class_count[1], class_count[2], class_count[3], class_count[4])
                img_count += 1
            print(f'{img_count}/{len(train_data) * 16} completed')
        wandb.log({'ground_truth_dataset': table})
        wandb.finish()
    populate_wandb_image_samples(train_data)
    return


@app.cell
def _(ENTITY, WANDB_PROJECT_NAME, classes, train_data, wandb):
    def _process_bounding_boxes_list(annots):
        result = []
        class_count = {i: 0 for i in range(0, 5)}
        for annot_idx, annotation in enumerate(annots):
            class_count[int(annotation[1])] += 1
            result.append({'position': {'middle': [float(annotation[2]), float(annotation[3])], 'width': float(annotation[4]), 'height': float(annotation[5])}, 'domain': 'pixel', 'class_id': int(annotation[1]), 'box_caption': classes[int(annotation[1])]})
        return (result, class_count)

    def populate_wandb_bbox(train_data):
        wandb.init(project=WANDB_PROJECT_NAME, entity=ENTITY, id='add-bbox-data', job_type='add-tables', resume='allow')
        class_set = wandb.Classes([{'name': 'Buoy', 'id': 0}, {'name': 'Can', 'id': 1}, {'name': 'Paper', 'id': 2}, {'name': 'Plastic Bag', 'id': 3}, {'name': 'Plastic Bottle', 'id': 4}])
        table = wandb.Table(columns=['Image-Id', 'BBox-Height', 'BBox-Width', 'Class-Id'])
        img_count = 0
        for batch_idx, batch_sample in enumerate(train_data):
            batch_images = batch_sample[0]
            batch_annotations = batch_sample[1]
            annots_dict = {i: [] for i in range(0, batch_images.shape[0])}
            for annot in batch_annotations:
                annots_dict[int(annot[0])].append(annot)
            for idx, image in enumerate(batch_images):
                result, class_count = _process_bounding_boxes_list(annots_dict[idx])
                for bbox in result:
                    height = bbox['position']['height']
                    width = bbox['position']['width']
                    class_id = bbox['class_id']
                    table.add_data(img_count, height, width, classes[class_id])
                img_count += 1
            print(f'{img_count}/{len(train_data) * 16} completed')
        wandb.log({'bounding_box_information': table})
        wandb.finish()
    populate_wandb_bbox(train_data)
    return


@app.cell
def _(ENTITY, WANDB_PROJECT_NAME, classes, np, train_data, wandb):
    def populate_wandb_spatial_heatmaps(train_data):
        wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=ENTITY,
            id='add-heatmap',
            job_type="add-tables",
            resume='allow'
        )

        class_set = wandb.Classes(
            [
                {"name": "Buoy", "id": 0},
                {"name": "Can", "id": 1},
                {"name": "Paper", "id": 2},
                {"name": "Plastic Bag", "id": 3},
                {"name": "Plastic Bottle", "id": 4},
            ]
        )
        heatmaps = [np.zeros((224, 224, 1), dtype=np.float32) for _ in classes]
        annotation_counts = {i:0 for i in range(len(classes))}

        table = wandb.Table(columns=["Class-Id", "Class-Name", "Spatial-Heatmap",
                                     "Num-Total-Objects"])

        for batch_idx, batch_sample in enumerate(train_data):
            batch_images = batch_sample[0]
            batch_annotations = batch_sample[1]

            annots_dict = {i:[] for i in range(0, batch_images.shape[0])}

            for annot in batch_annotations:
                class_idx = int(annot[1])

                midpoint_x = int(annot[2])
                midpoint_y = int(annot[3])
                width = int(annot[4])
                height = int(annot[5])

                x_min = midpoint_x - (width//2)
                x_max = midpoint_x + (width//2)

                y_min = midpoint_y - (height//2)
                y_max = midpoint_y + (height//2)

                heatmaps[class_idx][y_min:y_max, x_min:x_max] += 1

                annotation_counts[class_idx] += 1

            print(f"{batch_idx+1}/{len(train_data)} batches completed")

        for class_idx in range(len(classes)):
            heatmap = wandb.Image(
                heatmaps[class_idx],
                caption=classes[class_idx]
            )
            table.add_data(class_idx, classes[class_idx], heatmap, annotation_counts[class_idx])

        wandb.log({"spatial_heatmap_information": table})
        wandb.finish()

    populate_wandb_spatial_heatmaps(train_data)
    return


if __name__ == "__main__":
    app.run()
