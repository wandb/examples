# /// script
# dependencies = ["super-gradients", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/super-gradients/yolo_nas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥 Fine-tuning YOLO-NAS using Super-Gradients and Weights & Biases 🐝

    This notebook demonstrates how to fine-tune YOLO-NAS on a custom dataset using the [Super-Graidents](https://github.com/Deci-AI/super-gradients) library and performing experiment-tracking, logging and versioning model checkpoints, and viusalizing your detection datasets and prediction results during training.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installing Dependencies

    We install [Super-Graidents](https://github.com/Deci-AI/super-gradients) and [Weights & Biases](wandb.ai/site).
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: super_gradients wandb !pip install -qq super_gradients wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment Setup

    First, we initialize a W&B [run](https://docs.wandb.ai/guides/runs) using `wandb.init`.
    """)
    return


@app.cell
def _():
    import wandb


    wandb.init(project="yolo-nas-integration-2")
    return (wandb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we will initialize our trainer which will be in charge of the whole workflow, including the likes of training, evaluation, saving checkpoints, visualization of results, etc.
    """)
    return


@app.cell
def _():
    from super_gradients.training import Trainer


    trainer = Trainer(
        experiment_name='transfer_learning_object_detection_yolo_nas', ckpt_root_dir="./checkpoints/"
    )
    return (trainer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setting up the Dataset

    Next, we fetch a subset of the [BDD100K dataset](https://www.vis.xyz/bdd100k/) hosted on Weights & Biases as a [dataset artifact](https://docs.wandb.ai/guides/artifacts). Hosting the dataset as an artifact not only enables us to maintain different versions of our datasets, but also enables us to keep track of the runs that produced it or the runs that are using it via the [lineage panel](https://docs.wandb.ai/guides/app/pages/project-page#lineage-panel).
    """)
    return


@app.cell
def _(wandb):
    artifact = wandb.use_artifact('geekyrakshit/yolo-nas-integration/bdd100k-subset-yolo:v2', type='dataset')
    artifact_dir = artifact.download()
    return (artifact_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we set up the dataset parameters, and create the dataloaders for training, validation and testing using the `coco_detection_yolo_format_train` and the `coco_detection_yolo_format_val` functions from [Super-Graidents](https://github.com/Deci-AI/super-gradients), that would automatically create the dataset loading, pre-processing and augmentation pipelines.
    """)
    return


@app.cell
def _(artifact_dir):
    import os
    from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train, coco_detection_yolo_format_val
    )


    with open(os.path.join(artifact_dir, "labels.txt"), "r") as f:
        labels = f.read().split("\n")

    dataset_params = {
        'data_dir': artifact_dir,
        'train_images_dir':'train/images',
        'train_labels_dir':'train/labels',
        'val_images_dir':'val/images',
        'val_labels_dir':'val/labels',
        'test_images_dir':'test/images',
        'test_labels_dir':'test/labels',
        'classes': labels
    }

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
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
            'classes': dataset_params['classes']
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
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size':16,
            'num_workers':2
        }
    )
    return dataset_params, labels, test_data, train_data, val_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can visualize our datasets using the `plot_detection_dataset_on_wandb` function on our Weights & Biases dashboard. The datasets are logged into a [Weights & Biases Table](https://docs.wandb.ai/guides/tables), in which the images can be visualized overlayed with an [interactive overlays for computer vision tasks](https://docs.wandb.ai/guides/track/log/media#image-overlays).
    """)
    return


@app.cell
def _(test_data, train_data, val_data):
    from super_gradients.common.plugins.wandb import plot_detection_dataset_on_wandb


    plot_detection_dataset_on_wandb(train_data.dataset, max_examples=20, dataset_name="Train-Dataset")
    plot_detection_dataset_on_wandb(val_data.dataset, max_examples=20, dataset_name="Validation-Dataset")
    plot_detection_dataset_on_wandb(test_data.dataset, max_examples=20, dataset_name="Test-Dataset")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's how the datasets look on Weights & Biases 👇

    ![](./assets/data-vis.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Performing Transfer Learning

    For performing transfer learning, we first define the `YOLO_NAS_S` model as the pre-trained backbone from [Super-Graidents](https://github.com/Deci-AI/super-gradients). You can check the [Super-Gradients Model Zoo](https://docs.deci.ai/super-gradients/documentation/source/model_zoo.html#computer-vision-models-pretrained-checkpoints) for all the available pre-trained models for object detection.
    """)
    return


@app.cell
def _(dataset_params):
    from super_gradients.training import models
    from super_gradients.common.object_names import Models


    net = models.get(Models.YOLO_NAS_S, pretrained_weights="coco", num_classes=len(dataset_params["classes"]))
    return (net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment Tracking with Weights & Biases

    Now, we define the training parameters for the experiment. In order to perform experiment tracking with Weights & Biases, in the training parameters, we set the value of `sg_logger` to `wandb_sg_logger`. You can also set the value of additional parameters for the `wandb_sg_logger` by defining them under `sg_logger_params` like

    ```python
    train_params = {
        ...
        "sg_logger": "wandb_sg_logger",
        "sg_logger_params": {
            "save_checkpoints_remote": True,
            "save_tensorboard_remote": True,
            "save_logs_remote": True,
            "save_checkpoint_as_artifact": True,
        },
    }
    ```

    Here's a complete list of params for the `wandb_sg_logger`:

    |Parameter|Description|Default Value|
    |---|---|---|
    |experiment_name|Name used for logging and loading purposes|   |
    |storage_location|If set to 's3' (i.e. s3://my-bucket) saves the Checkpoints in AWS S3 otherwise saves the Checkpoints Locally|   |
    |resumed|If true, then old tensorboard files will **NOT** be deleted when tb_files_user_prompt=True|   |
    |training_params|training_params for the experiment|   |
    |checkpoints_dir_path|Local root directory path where all experiment logging directories will reside.|   |
    |tb_files_user_prompt|Asks user for Tensorboard deletion prompt.|   |
    |launch_tensorboard|Whether to launch a TensorBoard process.|False|
    |tensorboard_port|Specific port number for the tensorboard to use when launched (when set to None, some free port number will be used)|False|
    |save_checkpoints_remote|Saves checkpoints in s3.|True|
    |save_tensorboard_remote|Saves tensorboard in s3.|True|
    |save_logs_remote|Saves log files in s3.|True|
    |save_code|Save current code to wandb|False|
    |save_checkpoint_as_artifact|Save model checkpoint using Weights & Biases Artifact. Note that setting this option to True would save model checkpoints every epoch as a versioned artifact, which will result in use of increased storage usage on Weights & Biases.|False|

    The Weights & Biases integration for Super Gradients also come with the callback `WandBDetectionValidationPredictionLoggerCallback` for logging object detection predictions to Weights & Biases during training. In order to log object detection predictions to Weights & Biases, we can include this callback in the training parameters:

    ```python
    train_params = {
        ...
        "sg_logger": "wandb_sg_logger",
        "sg_logger_params": {
            "save_checkpoints_remote": True,
            "save_tensorboard_remote": True,
            "save_logs_remote": True,
            "save_checkpoint_as_artifact": True,
        },
        "phase_callbacks": [
            WandBDetectionValidationPredictionLoggerCallback(class_names=labels),
        ]
    }
    ```
    """)
    return


@app.cell
def _(dataset_params, labels):
    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import DetectionMetrics_050
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    from super_gradients.common.plugins.wandb.validation_logger import WandBDetectionValidationPredictionLoggerCallback


    train_params = {
        'silent_mode': True,
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": 10,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                # NOTE: num_classes needs to be defined here
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50',
        "sg_logger": "wandb_sg_logger",
        "sg_logger_params": {
            "save_checkpoints_remote": True,
            "save_tensorboard_remote": True,
            "save_logs_remote": True,
            "save_checkpoint_as_artifact": True,
        },
        "phase_callbacks": [
            WandBDetectionValidationPredictionLoggerCallback(class_names=labels),
        ]
    }
    return (train_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we perform the training.
    """)
    return


@app.cell
def _(net, train_data, train_params, trainer, val_data):
    trainer.train(model=net, training_params=train_params, train_loader=train_data, valid_loader=val_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's how the validation predictions look at the end of the training 👇

    ![](./assets/val-table.gif)
    """)
    return


if __name__ == "__main__":
    app.run()
