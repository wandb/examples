import wandb
import torch
from tqdm import tqdm
from typing import Tuple, List, Dict

from fastai.vision.all import *
from fastai.callback.wandb import WandbCallback

from .camvid_utils import get_dataloader
from .model import SegmentationModel
from .metrics import create_dice_table


def get_model_parameters(model):
    with torch.no_grad():
        num_params = sum(p.numel() for p in model.parameters())
    return num_params


def get_predictions(learner, test_dl=None, max_n=None):
    """Return the samples = (x,y) and outputs (model predictions decoded), and predictions (raw preds)"""
    test_dl = learner.dls.valid if test_dl is None else test_dl
    inputs, predictions, targets, outputs = learner.get_preds(
        dl=test_dl, with_input=True, with_decoded=True
    )
    x, y, samples, outputs = learner.dls.valid.show_results(
        tuplify(inputs) + tuplify(targets), outputs, show=False, max_n=max_n
    )
    return samples, outputs, predictions


def benchmark_inference_time(
    model_file,
    image_shape: tuple[int, int],
    batch_size: int,
    num_warmup_iters: int,
    num_iter: int,
    resize_factor: int
):
    model = torch.jit.load(model_file).cuda()
    
    dummy_input = torch.randn(
        batch_size, 3, image_shape[0] // resize_factor, 
        image_shape[0] // resize_factor, dtype=torch.float
    ).to("cuda")

    starter, ender = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    timings = np.zeros((num_iter, 1))

    print("Warming up GPU...")
    for _ in progress_bar(range(num_warmup_iters)):
        _ = model(dummy_input)

    print(
        f"Computing inference time over {num_iter} iterations with batches of {batch_size} images..."
    )

    with torch.no_grad():
        for step in progress_bar(range(num_iter)):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            timings[step] = starter.elapsed_time(ender)

    return np.sum(timings) / (num_iter * batch_size)

def get_learner(
    data_loader,
    backbone: str,
    hidden_dim: int,
    num_classes: int,
    checkpoint_file: Union[None, str, Path],
    loss_func,
    metrics: List,
    log_preds: bool = False,
):
    model = SegmentationModel(backbone, hidden_dim, num_classes=num_classes)
    mixed_precision_callback = MixedPrecision()
    wandb_callback = WandbCallback(log_model=False, log_preds=log_preds)
    nan_callback = TerminateOnNaNCallback()
    learner = Learner(
        data_loader,
        model,
        loss_func=loss_func,
        metrics=metrics,
        cbs=[mixed_precision_callback, wandb_callback, nan_callback],
    )
    if checkpoint_file is not None:
        load_model(checkpoint_file, learner.model, opt=None, with_opt=False)
        # learner.load(checkpoint_file)
    return learner


def table_from_dl(learn, test_dl, class_labels):
    samples, outputs, predictions = get_predictions(learn, test_dl)
    table = create_dice_table(samples, outputs, predictions, class_labels)
    return table


def save_model_to_artifacts(
    model,
    model_name: str,
    image_shape: Tuple[int, int],
    artifact_name: str,
    metadata: Dict,
):
    print("Saving model checkpoint")
    torch.save(model, model_name + ".pth")

    print("Saving model using scripting...")
    saved_model_script = torch.jit.script(model)
    saved_model_script.save(model_name + "_script.pt")

    print("Done!!!")
    example_forward_input = torch.randn(
        1, 3, image_shape[0] // 2, image_shape[0] // 2, dtype=torch.float
    ).to("cuda")
    print("Saving model using tracing...")
    saved_model_traced = torch.jit.trace(model, example_inputs=example_forward_input)
    saved_model_traced.save(model_name + "_traced.pt")
    print("Done!!!")
    # artifact = wandb.Artifact(artifact_name, type="model", metadata=metadata)
    # artifact.add_file(model_name + ".pth")
    # artifact.add_file(model_name + "_script.pt")
    # artifact.add_file(model_name + "_traced.pt")
    # wandb.log_artifact(artifact)
