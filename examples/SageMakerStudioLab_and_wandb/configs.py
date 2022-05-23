import ml_collections
from fastai.vision.all import *


def get_wandb_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    
    config.name = None
    config.project = "sagemaker_camvid_demo"
    config.entity = "capecape"
    config.job_type = "sweep"
    config.artifact_id = "camvid-dataset:latest"

    return config


def get_experiment_configs() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.seed = 123
    config.batch_size = 8
    config.image_height = 720
    config.image_width = 960
    config.image_resize_factor = 4
    config.validation_split = 0.2
    config.backbone = "mobilenetv2_100"
    config.hidden_dims = 256
    config.num_epochs = 5
    config.loss_function = "categorical_cross_entropy"
    config.weight_decay = 1e-2
    config.learning_rate = 1e-3
    config.fit = "fit"

    return config


def get_loss_mappings() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.categorical_cross_entropy = CrossEntropyLossFlat
    config.focal = FocalLossFlat
    config.dice = DiceLoss

    return config

def get_inference_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    
    config.batch_size = 2
    config.warmup = 10
    config.num_iter = 50
    config.resize_factor = 2
    return config

def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.experiment_configs = get_experiment_configs()
    config.wandb_configs = get_wandb_configs()
    config.loss_mappings = get_loss_mappings()
    config.inference = get_inference_config()
    config.sweep_count = 25
    config.sweep_method = "bayes"
    config.sweep_metric_name = "foreground_acc"
    config.sweep_goal = "maximize"
    config.early_terminate_type = "hyperband"
    config.early_terminate_min_iter = 5

    return config
