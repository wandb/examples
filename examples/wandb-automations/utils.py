from pathlib import Path
from types import SimpleNamespace

import wandb
import torch
import timm

def first(iterable, default=None):
    "Returns first element of `iterable` that is not None"
    return next(filter(None, iterable), default)

def save_model(model, model_name, models_folder="models", metadata=None, link=False):
    """Save the model to wandb as an artifact
    Args:
        model (nn.Module): Model to save.
        model_name (str): Name of the model.
        models_folder (str, optional): Folder to save the model. Defaults to "models".
        metadata (dict, optional): Metadata to save with the model. Defaults to None.
    """
    model_name = f"{wandb.run.id}_{model_name}"
    file_name = Path(f"{models_folder}/{model_name}.pth")
    file_name.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu")
    torch.save(model.state_dict(), file_name)
    at = wandb.Artifact(model_name, 
                        type="model", 
                        description="Model checkpoint from TIMM",
                        metadata=metadata)
    at.add_file(file_name)
    wandb.log_artifact(at)
    if link:
        wandb.run.link_artifact(at, 'model-registry/FMNIST_Classifier')

def load_model(model_artifact_name, eval=True):
    """Load the model from wandb artifacts
    Args:
        model_artifact_name (str): Name of the model artifact.
        eval (bool, optional): If True, sets the model to eval mode. Defaults to True.
    Returns:
        model (nn.Module): Loaded model.
    """
    artifact = wandb.use_artifact(model_artifact_name, type="model")
    model_path = Path(artifact.download()).absolute()
    model_config = SimpleNamespace(**artifact.metadata)
    
    # Load model
    model_weights = torch.load(first(model_path.glob("*.pth")))  # get first file
    model = timm.create_model(model_config.model_name, 
                              pretrained=False, 
                              num_classes=model_config.num_classes,
                              in_chans=model_config.in_chans)
    model.load_state_dict(model_weights)
    if eval:
        model.eval()
    return model