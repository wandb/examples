from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
import torchvision.transforms as T
import wandb
from tqdm.auto import tqdm

from utils import load_model

defaults = SimpleNamespace(
    bs=128,
    num_workers=0,
    device = "cuda:0" if torch.cuda.is_available() else "cpu",
    model_artifact = "model-registry/FMNIST_Classifier:latest",
    log_images = True,
    val_set = 1280,
)

def get_valid_dl(config=defaults):
    "Get the validation dataloader"
    valid_tfms = T.Compose([
        T.Resize(32, antialias=True),
        T.ToTensor()])
    all_val_data = FashionMNIST(".", train=False, download=True, transform=valid_tfms)
    val_ds = Subset(all_val_data, torch.arange(config.val_set))
    val_dl = DataLoader(val_ds, batch_size=config.bs, num_workers=config.num_workers)
    return val_dl
   

def validate_model(model, valid_dl, config=defaults):
    "Compute performance of the model on the validation dataset and log a Table"
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
                    # 🐝 Create a wandb Table to log images, labels and predictions
                    table = wandb.Table(columns=["image", "label", "pred"]+[f"score_{i}" for i in range(10)])
                
                probs = outputs.softmax(dim=1)
                for img, label, pred, prob in zip(images.to("cpu"), labels.to("cpu"), predicted.to("cpu"),  probs.to("cpu")):
                        table.add_data(wandb.Image(img[0].numpy()), label, pred, *prob.numpy())
        
        if config.log_images:
            wandb.log({"predictions":table}, commit=False)

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

def eval(config):
    # Initialize W&B run
    run = wandb.init(project="automations_demo", tags=["eval"], job_type="eval", config=config)
    run.log_code(name="evaluate_fmnist")

    config = run.config

    # Get the validation dataloader
    valid_dl = get_valid_dl(config)

    # Load the model
    model = load_model(config.model_artifact)

    # Log the validation loss and accuracy
    val_loss, val_acc = validate_model(model, valid_dl, config)
    run.log({"val_loss": val_loss, "val_acc": val_acc})

    run.finish()

if __name__ == "__main__":
    eval(defaults)
