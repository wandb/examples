# /// script
# dependencies = ["-", "https://download-openmmlab-com/mmcv/dist/cu111/torch1-9-0/index-html", "https://download-pytorch-org/whl/torch-stable-html", "mmcv-full", "torch", "torchvision", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/mmdetection/Train_Object_Detector_with_MMDetection_and_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{mmdetection-wandb-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases"/> <br>

    <!--- @wandbcode{mmdetection-wandb-colab, v=1} -->

    <img src="http://wandb.me/mini-diagram" width="600" alt="Weights & Biases"/>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 💡 Train an Object Detector with MMDetection and Weights and Biases

    In this colab, we will train an object detector using [MMDetection](https://mmdetection.readthedocs.io/en/latest/1_exist_data_model.html) on a tiny [Kitti](https://paperswithcode.com/dataset/kitti) dataset. Through this colab you will learn to:

    * use MMDetection to train an object detector on a custom dataset,
    * use [Weights and Biases](https://wandb.ai/site) to log training and validation metrics, visualize model predictions, version raw validation dataset, and more.

    This colab in particular, will showcase a dedicated `MMDetWandbHook` for MMDetection that can be used to:

    ✅ Log training and evaluation metrics. <br>
    ✅ Log versioned model checkpoints. <br>
    ✅ Log versioned validation dataset with ground truth bounding boxes. <br>
    ✅ Log and visualize model predictions.

    But before we continue, here's a quick summary of MMDetection and W&B if you are not familiar with them.

    ### 📸 MMDetection

    MMDetection is an open source object detection toolbox based on PyTorch. It provides composable components that are easy to customize and has out-of-box support for single and multi GPU training/inference. It also has hundreds of pretrained detection models in Model Zoo, and supports multiple standard datasets. Check out the GitHub repository [here](https://github.com/open-mmlab/mmdetection).

    ### 📸 Weights and Biases

    Consider **[Weights and Biases](https://wandb.ai/site)** (W&B) to be the GitHub for machine learning. Use W&B for machine learning experiment tracking, dataset and model versioning, project collaboration, hyperparameter optimization, dataset exploration, model evaluation and so much more. If you are new to W&B, check out this [intro colab](https://wandb.me/intro).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ⚽️ Imports and Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1️⃣ Install MMDetection

    MMDetection is heavily dependent on the [MMCV](https://mmcv.readthedocs.io/en/latest/#installation) library. We will have to install the version of MMCV that is compatible with the given PyTorch version. Check out the [Installation documentation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) for more details.
    """)
    return


@app.cell
def _(subprocess):
    # install dependencies: (use cu111 because colab has CUDA 11.1)
    # packages added via marimo's package management: torch==1.9.0+cu111 torchvision==0.10.0+cu111 https://download.pytorch.org/whl/torch_stable.html !pip install -qq torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

    # install mmcv-full thus we could use CUDA operators
    # packages added via marimo's package management: mmcv-full https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html !pip install -qq mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

    # Install mmdetection
    #! rm -rf mmdetection
    subprocess.call(['rm', '-rf', 'mmdetection'])
    #! git clone -b wandb2 https://github.com/ayulockin/mmdetection/
    subprocess.call(['git', 'clone', '-b', 'wandb2', 'https://github.com/ayulockin/mmdetection/'])
    import os
    os.chdir('mmdetection')

    # packages added via marimo's package management: . !pip install -e .
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2️⃣ Install Weights and Biases

    Install the latest version of W&B.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -qU wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3️⃣ General Imports
    """)
    return


@app.cell
def _():
    import os.path as osp
    import torch
    import torchvision
    import numpy as np
    import mmdet
    print(mmdet.__version__)
    # MMDetection
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector
    from mmdet.apis import train_detector
    from mmdet.datasets.builder import DATASETS
    from mmdet.datasets.custom import CustomDataset
    from mmdet.apis import set_random_seed
    import mmcv
    from mmcv import Config
    import wandb
    # MMCV
    # Weights and Biases
    print(wandb.__version__)
    return (
        Config,
        CustomDataset,
        DATASETS,
        build_dataset,
        build_detector,
        mmcv,
        np,
        set_random_seed,
        train_detector,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4️⃣ Login with your W&B account

    Create a free W&B account (it's free for personal and academic usage). Create a new API key at [wandb.ai/settings](https://wandb.ai/settings) and store it securely. API keys can only be viewed once when created.
    """)
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏀 Dataset

    We will be using a tiny KITTI dataset for this colab notebook.

    Even though KITTI is a standard dataset for object detection, tiny KITTI can be considered as a custom dataset (lesser number of classes). MMDetection, recommends to convert the data into COCO or PASCAL VOC formats or the middle format.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1️⃣ Download the dataset
    """)
    return


@app.cell
def _(os, subprocess):
    os.chdir('../')
    subprocess.call(['wget', 'https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip'])
    #! wget https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip
    #! unzip -q kitti_tiny.zip
    subprocess.call(['unzip', '-q', 'kitti_tiny.zip'])
    return


@app.cell
def _(subprocess):
    #! ls kitti_tiny
    subprocess.call(['ls', 'kitti_tiny'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > Note: The `training` folder contains both training and validation data samples. This split is determined by the `train.txt` and `val.txt` files.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2️⃣ Build Custom Dataloader

    To support a new data format, it's recommended to convert the annotations to COCO format or PASCAL VOC format. You can also convert them the "middle format".

    If you are converting annotations to COCO format, do so offline and use the `CocoDataset` class. If you are converting it to the PASCAL format, use the `VOCDataset` class.

    In the example below, we are converting it to the middle format. The `KittiTinyDataset` class will thus inherit the `CustomDataset` class and override the `load_annotations` method.

    You can find more details about customizing the dataset [here](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html).
    """)
    return


@app.cell
def _(CustomDataset, DATASETS, mmcv, np, os):
    @DATASETS.register_module()
    class KittiTinyDataset(CustomDataset):

        CLASSES = ('Car', 'Pedestrian', 'Cyclist')

        def load_annotations(self, ann_file):
            cat2label = {k: i for i, k in enumerate(self.CLASSES)}
            # load image list from file
            image_list = mmcv.list_from_file(self.ann_file)
    
            data_infos = []
            # convert annotations to middle format
            for image_id in image_list:
                filename = f'{self.img_prefix}/{image_id}.jpeg'
                image = mmcv.imread(filename)
                height, width = image.shape[:2]
    
                data_info = dict(filename=f'{image_id}.jpeg', width=width, height=height)
    
                # load annotations
                label_prefix = self.img_prefix.replace('image_2', 'label_2')
                lines = mmcv.list_from_file(os.path.join(label_prefix, f'{image_id}.txt'))
    
                content = [line.strip().split(' ') for line in lines]
                bbox_names = [x[0] for x in content]
                bboxes = [[float(info) for info in x[4:8]] for x in content]
    
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
    
                # filter 'DontCare'
                for bbox_name, bbox in zip(bbox_names, bboxes):
                    if bbox_name in cat2label:
                        gt_labels.append(cat2label[bbox_name])
                        gt_bboxes.append(bbox)
                    else:
                        gt_labels_ignore.append(-1)
                        gt_bboxes_ignore.append(bbox)

                data_anno = dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=np.long),
                    bboxes_ignore=np.array(gt_bboxes_ignore,
                                           dtype=np.float32).reshape(-1, 4),
                    labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

                data_info.update(ann=data_anno)
                data_infos.append(data_info)

            return data_infos

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏈 Model

    There are over hundred pre-trained object detectors provided by MMDetection via Model Zoo. Check out the Model Zoo [documentation](https://mmdetection.readthedocs.io/en/v2.21.0/model_zoo.html) page.

    You can also customize the model's backbone, neck, head, ROI, and loss. More on customizing the model [here](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_models.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1️⃣ Download the model

    We will be using a pretrained model checkpoint to fine tune on our custom dataset. Let's download the model in the `checkpoints` directory.
    """)
    return


@app.cell
def _(subprocess):
    #! mkdir checkpoints
    subprocess.call(['mkdir', 'checkpoints'])
    #! wget -c https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth -O checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth
    subprocess.call(['wget', '-c', 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth', '-O', 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ⚾️ Configuration

    MMDetection relies heavily on a config system. In the cell below, we will be loading a config file and modify few of the methods as per the need of this notebook.

    Note that both train and test dataloaders will use the same training samples. This is not a recommended practice but for the sake of a simplified notebook, let's use it.

    Learn more about the MMDetection Config system [here](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1️⃣ Load the config file
    """)
    return


@app.cell
def _(Config):
    cfg = Config.fromfile('mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
    return (cfg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2️⃣ Modify data config
    """)
    return


@app.cell
def _(cfg):
    # Define type and path to the images.
    cfg.dataset_type = 'KittiTinyDataset'
    cfg.data_root = 'kitti_tiny/'

    cfg.data.test.type = 'KittiTinyDataset'
    cfg.data.test.data_root = 'kitti_tiny/'
    cfg.data.test.ann_file = 'train.txt'
    cfg.data.test.img_prefix = 'training/image_2'

    cfg.data.train.type = 'KittiTinyDataset'
    cfg.data.train.data_root = 'kitti_tiny/'
    cfg.data.train.ann_file = 'train.txt'
    cfg.data.train.img_prefix = 'training/image_2'

    cfg.data.val.type = 'KittiTinyDataset'
    cfg.data.val.data_root = 'kitti_tiny/'
    cfg.data.val.ann_file = 'val.txt'
    cfg.data.val.img_prefix = 'training/image_2'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3️⃣ Modify model config
    """)
    return


@app.cell
def _(cfg):
    # The number of unique objects in the training data.
    cfg.model.roi_head.bbox_head.num_classes = 3
    # Use the pretrained model.
    cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4️⃣ Modify training config
    """)
    return


@app.cell
def _(cfg, set_random_seed):
    # The original learning rate (LR) is set for 8-GPU training.
    # We divide it by 8 since we only use one GPU.
    cfg.optimizer.lr = 0.02 / 8
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # Epochs
    cfg.runner.max_epochs = 12

    # Set seed thus the results are more reproducible
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # ⭐️ Set the checkpoint interval.
    cfg.checkpoint_config.interval = 1

    # Set up working dir to save files and logs.
    cfg.work_dir = './tutorial_exps'
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 5️⃣ Modify evaluation config
    """)
    return


@app.cell
def _(cfg):
    # Change the evaluation metric since we use customized dataset.
    cfg.evaluation.metric = 'mAP'

    # ⭐️ Set the evaluation interval.
    cfg.evaluation.interval = 1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🎾 Define Weights and Biases Hook

    MMDetection comes with a dedicated Weights and Biases Hook - `MMDetWandHook`. MMCV, the parent repository, has a `WandbLoggerHook` that can be used to for basic logging.

    With this dedicated hook, you can:

    * log train and eval metrics along with system (CPU/GPU) metrics,
    * visualize the validation dataset as interactive [W&B Tables](https://docs.wandb.ai/guides/data-vis),
    * visualize the model prediction as interactive W&B Tables, and
    * save the model checkpoints as [W&B Artifacts](https://docs.wandb.ai/guides/artifacts).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To use this hook, you can append a dict to `log_config.hooks`. The `log_config` wraps multiple logger hooks like  the `TextLoggerHook` used below.

    There are four important arguments in the `MMDetWandbHook` that can help you get the most out of MMDetection.

    - `init_kwargs`: Use this argument to in-turn pass arguments to `wandb.init`. You can use it to set the W&B project name, set the team name entity if you want to log the runs to a team account, pass the configuration, and more. Check out what all can you pass to `wandb.init` [here](https://docs.wandb.ai/ref/python/init).

    - `log_checkpoint`: The model checkpoints are saved at intervals determined by `checkpoint_config.interval` (starred above). If `log_checkpoint` is `True` the saved checkpoints will be saved as versioned W&B Artifact. Note that this feature is dependent on MMCV's [`CheckpointHook`](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.CheckpointHook).

    - `log_checkpoint_metadata`: If `log_checkpoint_metadata` is True, every checkpoint artifact will have a metadata associated with it. The metadata contains the evaluation metrics computed on validation data with that checkpoint along with the current epoch. If True, it also marks the checkpoint version with the best evaluation metric with a `best` alias. You can choose the best checkpoint in the W&B Artifacts UI using this.

    - `num_eval_images`: At every evaluation interval, the `MMDetWandbHook` logs the model prediction as interactive W&B Tables. The eval interval is determined by `evaluation.interval` (starred above). The number of samples logged is given by `num_eval_images`. The predicted bounding boxes along with the ground truth are logged at every evaluation interval. However, the validation data is logged just once. This Feature is dependent on MMCV's [`EvalHook`](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.EvalHook) or [`DistEvalHook`](https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.DistEvalHook).
    """)
    return


@app.cell
def _(cfg):
    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
             init_kwargs={'project': 'MMDetection-tutorial'},
             interval=10,
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=10)]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏐 Train

    Now that we have the dataset, pretrained model weight, and have defined the configs. Let's stitch them together to train an object detector.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1️⃣ Build the Dataset
    """)
    return


@app.cell
def _(build_dataset, cfg):
    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    return (datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2️⃣ Build the Model
    """)
    return


@app.cell
def _(build_detector, cfg, datasets):
    # Build the detector
    model = build_detector(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3️⃣ Train with W&B
    """)
    return


@app.cell
def _(cfg, datasets, model, train_detector):
    # Create work_dir
    # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4️⃣ Notes on using `MMDetWandbHook`.

    Using `MMDetWandbHook` is easy and in most cases it will throw friendly `UserWarning` if something is not quite right. However in the best interest, here are some of things and best practices you should keep in mind:

    * The `MMDetWandbHook` depends on `CheckpointHook` for logging the checkpoints as W&B Artifacts and `EvalHook`/`DistEvalHook` for logging validation data and model predictions. If anyone or both aren't available, this hook will give `UserWarning` and not cause any error.

    * The priority of both `CheckpointHook` and `EvalHook`/`DistEvalHook` should be more than `MMDetWandbHook`.

    * The validation data is logged once as `val_data` W&B Table. The evaluation tables, use reference to this data thus you will not be uploading the same data multiple times.

    * If you want to log the configuration to W&B, pass this key-value pair `'config': cfg._cfg_dict.to_dict()` to `init_kwargs`.
    """)
    return


if __name__ == "__main__":
    app.run()
