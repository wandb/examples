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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/yolo/Logging_YOLOv5_Experiments_with_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{yolov5-log} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://user-images.githubusercontent.com/26833433/82952157-51b7db00-9f5d-11ea-8f4b-dda1ffecf992.jpg">

    <!--- @wandbcode{yolov5-log} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    # You Always Log Everything (YALE)

    ### Logging YOLOv5 Experiments with W&B
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [YOLO](https://github.com/ultralytics/yolov5) ("You Only Look Once") provides tools for real-time object detection with convolutional neural networks.

    YOLO now works with [Weights & Biases](http://wandb.com),
    an experiment tracking toolkit, so you can
    keep track of all the hyperparameters you've tried,
    view real-time updates on system and model metrics,
    version and store datasets and models,
    [and more](http://github.com/wandb/examples)!

    In this colab, we'll show you how to use YOLO and W&B together.
    **It's as easy as running a single `pip install` before you run your YOLO experiments!**

    <h3> Follow along with a <a href="http://wandb.me/yolov5-video"> video tutorial</a> on YouTube. </h3>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0. Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First, let's get ourselves organized: clone the repo, install our dependencies, and confirm we've got PyTorch and a GPU.
    """)
    return


@app.cell
def _(subprocess):
    #! git clone --depth 1 https://github.com/ultralytics/yolov5
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/ultralytics/yolov5'])
    # clone repo
    import os
    os.chdir('yolov5')
    # '%pip install -qr requirements.txt # install dependencies' command supported automatically in marimo

    import torch
    from IPython.display import Image, clear_output  # to display images

    clear_output()
    print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
    return Image, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Inference

    Now, let's apply a pre-trained, already existing object detection network.

    `detect.py` runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

    Here, we'll just run a sample image through that network to make sure everything is working.
    """)
    return


@app.cell
def _(Image, subprocess):
    #! python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/
    subprocess.call(['python', 'detect.py', '--weights', 'yolov5s.pt', '--img', '640', '--conf', '0.25', '--source', 'data/images/'])
    Image(filename='runs/detect/exp/bus.jpg', width=600)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Results are saved to `runs/detect`. A full list of available inference sources:
    <img src="https://user-images.githubusercontent.com/26833433/98274798-2b7a7a80-1f94-11eb-91a4-70c73593e26b.jpg" width="900">
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Training to Fine-Tune

    For applications, it's often important to take a pre-trained model
    and fine-tune it to work on a specific dataset --
    [for example, in a construction safety application](https://wandb.ai/authors/artifact-workplace-safety/reports/Organize-Your-Machine-Learning-Pipelines-with-Artifacts--VmlldzoxODQwNTY), we might use fine-tuning to specialize our network in detecting the presence/absence of protective equipment.

    We'll mimic this process on the
    [COCO128](https://www.kaggle.com/ultralytics/coco128) image tutorial dataset.
    """)
    return


@app.cell
def _(subprocess, torch):
    2# Download COCO128
    torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'tmp.zip')
    #! unzip -q tmp.zip -d ../ && rm tmp.zip
    subprocess.call(['unzip', '-q', 'tmp.zip', '-d', '../', '&&', 'rm', 'tmp.zip'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Weights & Biases Logging (🚀 NEW)

    [Weights & Biases](https://www.wandb.com/) (W&B) is now integrated with YOLOv5 for real-time visualization and cloud logging of training runs. This allows for better run comparison and introspection, as well improved visibility and collaboration among team members. To enable W&B logging install `wandb`, and then train normally (you will be guided through setting up `wandb` account during your first use).
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install "wandb==0.12.10"
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And that's it! So long as W&B is installed, you'll get rich, detailed metrics in a live dashboard accessible from a browser on any device.

    Just click the link that appears below next to `wandb` and the 🚀 emoji.
    """)
    return


@app.cell
def _(subprocess):
    # Train YOLOv5s on COCO128 for 5 epochs
    #! python train.py --img 640 --batch 64 --epochs 5 --data coco128.yaml --weights yolov5s.pt
    subprocess.call(['python', 'train.py', '--img', '640', '--batch', '64', '--epochs', '5', '--data', 'coco128.yaml', '--weights', 'yolov5s.pt'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With W&B, during training you will see live updates on the dashboard at [wandb.ai](https://www.wandb.ai/), including interactive bounding box visualizations (look for a panel called "Images" in the Media panel section), and you can create and share detailed [Reports](https://wandb.ai/glenn-jocher/yolov5_tutorial/reports/YOLOv5-COCO128-Tutorial-Results--VmlldzozMDI5OTY) of your results. For more information see the [YOLOv5 Weights & Biases Tutorial](https://github.com/ultralytics/yolov5/issues/1289)
    or check out the [video tutorial for this notebook](http://wandb.me/yolov5-video).

    <img src="https://user-images.githubusercontent.com/26833433/98184457-bd3da580-1f0a-11eb-8461-95d908a71893.jpg" width="800">
    """)
    return


if __name__ == "__main__":
    app.run()
