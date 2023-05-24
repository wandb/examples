from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
import wandb
from tqdm.auto import tqdm

from utils import load_model

PROJECT = 'fashion-launch'
ENTITY = "capecape"

defaults = SimpleNamespace(
    bs=128,
    num_workers=0,
    device = "cuda:0" if torch.cuda.is_available() else "cpu",
    model_artifact = "capecape/fashion-launch/uvef8vsn_resnest14d:v0",
    log_images = False,
)

def get_valid_dl(config=defaults):
    "Get the validation dataloader"
    valid_tfms = T.Compose([
        T.Resize(32, antialias=True),
        T.ToTensor()])
    valid_ds = FashionMNIST(".", train=False, download=True, transform=valid_tfms)
    valid_dl = DataLoader(valid_ds, batch_size=config.bs, num_workers=config.num_workers)
    return valid_dl
   

def validate_model(model, valid_dl, config=defaults):
    "Compute performance of the model on the validation dataset and log a wandb.Table"
    loss_func=nn.CrossEntropyLoss()
    model.to(config.device)
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in tqdm(enumerate(valid_dl), total=len(valid_dl)):
            images, labels = images.to(config.device), labels.to(config.device)

            # Forward pass ➡
            outputs = model(images)
            val_loss += loss_func(outputs, labels)*labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Log validation predictions and images to the dashboard
            if config.log_images:
                if i == 0:
                    # 🐝 Create a wandb Table to log images, labels and predictions to
                    table = wandb.Table(columns=["image", "label", "pred"]+[f"score_{i}" for i in range(10)])
                
                probs = outputs.softmax(dim=1)
                for img, label, pred, prob in zip(images.to("cpu"), labels.to("cpu"), predicted.to("cpu"),  probs.to("cpu")):
                        # table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())
                        table.add_data(wandb.Image(img[0].numpy()), label, pred, *prob.numpy())
        
        if config.log_images:
            wandb.log({"val_table/predictions_table":table}, commit=False)

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

def eval(config):
    # Initialize W&B run
    run = wandb.init(project=PROJECT, entity=ENTITY, tags=["eval"], job_type="eval", config=config)

    config = wandb.config

    # Get the validation dataloader
    valid_dl = get_valid_dl(config)

    # Load the model
    model = load_model(config.model_artifact)

    # Log the validation loss and accuracy
    val_loss, val_acc = validate_model(model, valid_dl, config)
    wandb.summary["val_loss"] = val_loss
    wandb.summary["val_acc"] = val_acc

    run.finish()

if __name__ == "__main__":
    eval(defaults)
