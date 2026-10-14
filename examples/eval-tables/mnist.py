"""MNIST data, model and training helpers for eval_tables_demo.py."""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset
from torchvision import datasets, transforms

NUM_CLASSES = 10


def load_data(data_dir, train_size, val_rows):
    """Return a training subset and a fixed validation batch."""
    to_tensor = transforms.ToTensor()
    train = datasets.MNIST(data_dir, train=True, download=True, transform=to_tensor)
    val = datasets.MNIST(data_dir, train=False, download=True, transform=to_tensor)
    val_images = torch.stack([val[i][0] for i in range(val_rows)])
    val_labels = torch.tensor([val[i][1] for i in range(val_rows)])
    return Subset(train, range(train_size)), val_images, val_labels


def build_model(hidden_size):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, NUM_CLASSES),
    )


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        optimizer.zero_grad()
        loss = F.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, images, labels):
    """Return per-row predictions, confidences and losses."""
    model.eval()
    with torch.inference_mode():
        logits = model(images)
        losses = F.cross_entropy(logits, labels, reduction="none")
        confidence, preds = logits.softmax(dim=1).max(dim=1)
    return preds, confidence, losses
