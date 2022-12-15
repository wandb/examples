import os
import sys

import wandb
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv


category = sys.argv[1]

wandb.init(project="point-cloud-segmentation", name=f"visualize-{category}", entity="geekyrakshit", job_type="visualize")

config = wandb.config
config.category = category

path = os.path.join('ShapeNet', config.category)
pre_transform = T.NormalizeScale()
train_dataset = ShapeNet(path, config.category, split='trainval', pre_transform=pre_transform)
test_dataset = ShapeNet(path, config.category, split='test', pre_transform=pre_transform)

segmentation_class_frequency = {}
for idx in tqdm(range(len(train_dataset))):
    pc_viz = train_dataset[idx].pos.numpy().tolist()
    segmentation_label = train_dataset[idx].y.numpy().tolist()
    for label in set(segmentation_label):
        segmentation_class_frequency[label] = segmentation_label.count(label)

class_offset = min(list(segmentation_class_frequency.keys()))

table = wandb.Table(columns=[
    "Point-Cloud", "Segmentation-Class-Frequency", "Model-Category", "Split"
])
for idx in tqdm(range(len(train_dataset))):
    pc_viz = train_dataset[idx].pos.numpy().tolist()
    segmentation_label = train_dataset[idx].y.numpy().tolist()
    
    frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
    for label in set(segmentation_label):
        frequency_dict[label] = segmentation_label.count(label)
    
    for j in range(len(pc_viz)):
        pc_viz[j] += [segmentation_label[j] + 1 - class_offset]
    
    table.add_data(
        wandb.Object3D(np.array(pc_viz)), frequency_dict, config.category, "Train-Val"
    )

data = [[key, segmentation_class_frequency[key]] for key in segmentation_class_frequency.keys()]
wandb.log({
    f"ShapeNet Class-Frequency Distribution for Train-Val Set" : wandb.plot.bar(
        wandb.Table(data=data, columns = ["Class", "Frequency"]),
        "Class", "Frequency",
        title=f"ShapeNet Class-Frequency Distribution for Train-Val Set"
    )
})

segmentation_class_frequency = {}
for idx in tqdm(range(len(test_dataset))):
    pc_viz = train_dataset[idx].pos.numpy().tolist()
    segmentation_label = train_dataset[idx].y.numpy().tolist()
    for label in set(segmentation_label):
        segmentation_class_frequency[label] = segmentation_label.count(label)

class_offset = min(list(segmentation_class_frequency.keys()))

for idx in tqdm(range(len(test_dataset))):
    pc_viz = train_dataset[idx].pos.numpy().tolist()
    segmentation_label = train_dataset[idx].y.numpy().tolist()
    
    frequency_dict = {key: 0 for key in segmentation_class_frequency.keys()}
    for label in set(segmentation_label):
        frequency_dict[label] = segmentation_label.count(label)
    
    for j in range(len(pc_viz)):
        pc_viz[j] += [segmentation_label[j] + 1 - class_offset]
    
    table.add_data(
        wandb.Object3D(np.array(pc_viz)), frequency_dict, config.category, "Test"
    )

wandb.log({"ShapeNet-Dataset": table})

data = [[key, segmentation_class_frequency[key]] for key in segmentation_class_frequency.keys()]
wandb.log({
    f"ShapeNet Class-Frequency Distribution for Test Set" : wandb.plot.bar(
        wandb.Table(data=data, columns = ["Class", "Frequency"]),
        "Class", "Frequency",
        title=f"ShapeNet Class-Frequency Distribution for Test Set"
    )
})

wandb.finish()
