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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/super-gradients/yolo_nas_sweep_run.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installation and Imports
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
    import pandas as pd

    from google.colab import userdata
    from torchvision.io import read_image
    from torch.utils.data import DataLoader
    from IPython.display import clear_output

    from super_gradients.training import models, Trainer, dataloaders
    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import DetectionMetrics_050
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
    from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

    os.environ["WANDB_API_KEY"] = userdata.get('wandb')
    os.environ["ROBOFLOW_API_KEY"] = userdata.get('roboflow')
    return (
        DetectionMetrics_050,
        PPYoloELoss,
        PPYoloEPostPredictionCallback,
        Trainer,
        coco_detection_yolo_format_train,
        coco_detection_yolo_format_val,
        models,
        os,
        torch,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Definitions
    """)
    return


@app.cell
def _(torch):
    seed = 42
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download and Register dataset
    """)
    return


@app.cell
def _(os):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
    project = rf.workspace("easyhyeon").project("trash-sea")
    dataset = project.version(10).download("yolov5")
    return


@app.cell
def _():
    ENTITY = "ml-colabs"
    SWEEP_NUM_RUNS = 100
    WANDB_PROJECT_NAME = "fconn-yolo-nas"
    DATASET_PATH = "/content/trash-sea-10"
    return DATASET_PATH, ENTITY, SWEEP_NUM_RUNS, WANDB_PROJECT_NAME


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Define Sweep Configuration and functions
    """)
    return


@app.cell
def _(WANDB_EXP_NAME):
    sweep_configuration = {
        "name": WANDB_EXP_NAME,
        "metric": {"name": "Valid_mAP@0.50", "goal": "maximize"},
        "method": "bayes",
        "parameters": {
            "batch_size": {"values": [16, 24, 32]},
            "optimizer": {"values": ["Adam", "SGD", "RMSProp", "AdamW"]},
            "ema_decay": {"min":0.5, "max":0.9},
            "ema_decay_type": {"values": ["constant", "threshold"]},
            "cosine_lr_ratio": {"min": 0.01, "max": 0.4},
            "iou_loss_weight": {"min": 0.25, "max": 2.0},
            "dfl_loss_weight": {"min": 0.25, "max": 2.0},
            "classification_loss_weight": {"min": 0.25, "max": 2.0},
            "model_flavor": {"values": ["yolo_nas_s", "yolo_nas_m", "yolo_nas_l"]},
            "weight_decay": {"min": 0.0001, "max": 0.01},
        },
    }
    return (sweep_configuration,)


@app.cell
def _(
    DATASET_PATH,
    DetectionMetrics_050,
    ENTITY,
    PPYoloELoss,
    PPYoloEPostPredictionCallback,
    Trainer,
    WANDB_EXP_NAME,
    WANDB_PROJECT_NAME,
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
    models,
    wandb,
):
    def main_call():

        CHECKPOINT_DIR = 'checkpoints'

        wandb.init(
            project=WANDB_PROJECT_NAME,
            entity=ENTITY,
            resume="allow",
            save_code=True,
            id=WANDB_EXP_NAME
        )

        config = wandb.config

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

        train_data = coco_detection_yolo_format_train(
            dataset_params={
                'data_dir': dataset_params['data_dir'],
                'images_dir': dataset_params['train_images_dir'],
                'labels_dir': dataset_params['train_labels_dir'],
                'classes': dataset_params['classes'],
            },
            dataloader_params={
                'batch_size':config["batch_size"],
                'num_workers':4
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
                'batch_size':config["batch_size"],
                'num_workers':4
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
                'batch_size':config["batch_size"],
                'num_workers':4
            }
        )

        train_data.dataset.transforms = train_data.dataset.transforms[1:]

        model = models.get(
            config["model_flavor"],
            num_classes=len(dataset_params['classes']),
            pretrained_weights="coco"
        )

        train_params = {
            'silent_mode': False,
            "average_best_models":True,
            "warmup_mode": "linear_epoch_step",
            "warmup_initial_lr": 1e-6,
            "lr_warmup_epochs": 3,
            "initial_lr": 1e-3,
            "lr_mode": "cosine",
            "cosine_final_lr_ratio": config["cosine_lr_ratio"],
            "optimizer": config["optimizer"],
            "optimizer_params": {
                "weight_decay": config["weight_decay"]
            },
            "zero_weight_decay_on_bias_and_bn": True,
            "ema": True,
            "ema_params": {
                "decay": config["ema_decay"],
                "decay_type": config["ema_decay_type"]
            },
            "max_epochs": 5,
            "mixed_precision": False,
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=len(dataset_params['classes']),
                reg_max=16,
                iou_loss_weight=config["iou_loss_weight"],
                dfl_loss_weight=config["dfl_loss_weight"],
                classification_loss_weight=config["classification_loss_weight"]
            ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
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
                "project_name": WANDB_PROJECT_NAME,
                "save_checkpoints_remote": True,
                "save_tensorboard_remote": True,
                "save_logs_remote": True,
                "entity": ENTITY
            }
        }

        trainer = Trainer(
            experiment_name=WANDB_EXP_NAME,
            ckpt_root_dir=CHECKPOINT_DIR
        )

        trainer.train(
            model=model,
            training_params=train_params,
            train_loader=train_data,
            valid_loader=val_data
        )

        wandb.finish()

    return (main_call,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Execute Sweep
    """)
    return


@app.cell
def _(SWEEP_NUM_RUNS, main_call, sweep_configuration, wandb):
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project="yolo-nas-sweep"
    )

    wandb.agent(sweep_id, function=main_call, count=SWEEP_NUM_RUNS)
    return


if __name__ == "__main__":
    app.run()
