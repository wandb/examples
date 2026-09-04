# /// script
# dependencies = ["accelerate", "datasets", "diffusers", "ml-collections", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/diffusers-image-generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{unconditional-diffusers-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Unconditional Image Generation using 🤗 Diffusers + Weights & Biases 🪄🐝

    <!--- @wandbcode{unconditional-diffusers-colab} -->

    **Reference:** [Official Diffusers Example for Unconditional Image Generation](https://github.com/huggingface/diffusers/tree/main/examples/unconditional_image_generation)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: accelerate diffusers datasets wandb ml_collections !pip install -qq accelerate diffusers datasets wandb ml_collections
    return


@app.cell
def _(subprocess):
    #! accelerate config
    subprocess.call(['accelerate', 'config'])
    return


@app.cell
def _(subprocess):
    #! accelerate env
    subprocess.call(['accelerate', 'env'])
    return


@app.cell
def _():
    import math
    import os
    from pathlib import Path
    from typing import Optional

    import torch
    import torch.nn.functional as F
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        RandomHorizontalFlip,
        Resize,
        ToTensor,
    )

    from accelerate import Accelerator
    from accelerate import notebook_launcher
    from accelerate.logging import get_logger

    from datasets import load_dataset

    from diffusers import UNet2DModel
    from diffusers import DDPMPipeline, DDPMScheduler
    from diffusers import DDIMPipeline, DDIMScheduler
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid

    from ml_collections import ConfigDict
    from tqdm.auto import tqdm
    import wandb

    return (
        Accelerator,
        CenterCrop,
        Compose,
        ConfigDict,
        DDIMPipeline,
        DDIMScheduler,
        DDPMPipeline,
        DDPMScheduler,
        EMAModel,
        F,
        InterpolationMode,
        Normalize,
        RandomHorizontalFlip,
        Resize,
        ToTensor,
        UNet2DModel,
        get_scheduler,
        load_dataset,
        math,
        notebook_launcher,
        torch,
        tqdm,
        wandb,
    )


@app.cell
def _(ConfigDict):
    config = ConfigDict()

    ##################### Dataset Configs #####################

    # The name of the Dataset (from the HuggingFace hub or WandB Artifacts) to train on (could be your own,
    # possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,
    # or to a folder containing files that HF Datasets can understand.
    config.dataset_name = "geekyrakshit/diffusers-image-generation/anime-faces:v0" #@param {type:"string"}

    # Is the config.dataset_name a Weights & Biase artifact or not
    config.is_dataset_wandb_artifact = True  #@param {type:"boolean"}

    # The config of the Dataset, leave as None if there's only one config.
    config.dataset_config_name = None #@param {type:"raw"}

    # A folder containing the training data. Folder contents must follow the structure described in
    # https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file
    # must exist to provide the captions for the images. Ignored if `dataset_name` is specified.
    config.train_data_dir = None #@param {type:"raw"}

    # The output directory where the model predictions and checkpoints will be written.
    config.output_dir = "ddpm-model-64" #@param {type:"string"}

    # The directory where the downloaded models and datasets will be stored.
    config.cache_dir = None #@param {type:"raw"}


    ##################### Training Configs #####################

    # Type of Diffusion pipeline
    config.diffusion_pipeline = "ddim" #@param ["ddpm", "ddim"] {type:"string"}

    # The resolution for input images, all the images in the train/validation dataset will be resized to
    # this resolution.
    config.resolution = 64 #@param {type:"slider", min:64, max:1024, step:4}

    # Batch size (per device) for the training dataloader.
    config.train_batch_size = 64 #@param {type:"slider", min:16, max:256, step:16}

    # The number of images to generate for evaluation.
    config.eval_batch_size = 64 #@param {type:"slider", min:16, max:256, step:16}

    # The number of subprocesses to use for data loading. 0 means that the data will be loaded in the
    # main process.
    config.dataloader_num_workers = 0 #@param {type:"slider", min:0, max:16, step:1}

    # Number of diffusion steps used to train the model.
    config.num_train_timesteps = 1000 #@param {type:"slider", min:0, max:5000, step:100}

    # Number of training epochs
    config.num_epochs = 100 #@param {type:"slider", min:0, max:500, step:1}

    # How often to save images during training.
    config.save_images_epochs = 10 #@param {type:"slider", min:0, max:100, step:5}

    # How often to save the model during training.
    config.save_model_epochs = 10 #@param {type:"slider", min:0, max:100, step:5}

    # Number of updates steps to accumulate before performing a backward/update pass.
    config.gradient_accumulation_steps = 1 #@param {type:"slider", min:0, max:10, step:1}

    # Initial learning rate (after the potential warmup period) to use.
    config.learning_rate = 1e-4 #@param {type:"number"}

    # The scheduler type to use. Choose between
    # ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
    config.lr_scheduler = "cosine" #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"] {type:"string"}

    # Number of steps for the warmup in the learning rate scheduler.
    config.lr_warmup_steps = 500 #@param {type:"slider", min:0, max:1000, step:50}

    # The exponential decay rate for the 1st moment estimates (the beta1 parameter for the Adam optimizer).
    config.adam_beta1 = 0.95 #@param {type:"number"}

    # The exponential decay rate for the 2nd moment estimates (the beta2 parameter for the Adam optimizer).
    config.adam_beta2 = 0.999 #@param {type:"number"}

    # Weight decay magnitude for the Adam optimizer.
    config.adam_weight_decay = 1e-6 #@param {type:"number"}

    # Epsilon value for the Adam optimizer.
    config.adam_epsilon = 1e-08 #@param {type:"number"}

    # Whether to use Exponential Moving Average for the final model weights.
    config.use_ema = True #@param {type:"boolean"}

    # The inverse gamma value for the EMA decay.
    config.ema_inv_gamma = 1.0 #@param {type:"number"}

    # The power value for the EMA decay.
    config.ema_power = 3 / 4 #@param {type:"raw"}

    # The maximum decay magnitude for EMA.
    config.ema_max_decay = 0.9999 #@param {type:"number"}

    # For distributed training: local_rank
    config.local_rank = -1 #@param {type:"number"}

    # Number of processes
    config.num_processes = 1 #@param {type:"number"}

    # Whether to use mixed precision.
    # Choose between "no", "fp16" and "bf16" (bfloat16).
    # Note that Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.
    config.mixed_precision = "no" #@param ["no", "fp16", "bf16"] {type:"raw"}


    ##################### Weights & Biases Configs #####################

    # Weights & Biases Project
    config.wandb_project = "diffusers-image-generation" #@param {type:"string"}

    # Weights & Biases Entity
    config.wandb_entity = "geekyrakshit" #@param {type:"string"}

    # Number of images to be visualized in a table
    config.num_images_in_table = 6 #@param {type:"slider", min:1, max:50, step:1}
    return (config,)


@app.cell
def _(UNet2DModel, config):
    def build_unet_model():
        return UNet2DModel(
            sample_size=config.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    return (build_unet_model,)


@app.cell
def _(
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    config,
    load_dataset,
    torch,
    wandb,
):
    def transforms(examples):
        augmentations = Compose(
            [
                Resize(
                    config.resolution,
                    interpolation=InterpolationMode.BILINEAR
                ),
                CenterCrop(config.resolution),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )
        images = [
            augmentations(image.convert("RGB"))
            for image in examples["image"]
        ]
        return {"input": images}


    def build_dataloader():
        if not config.is_dataset_wandb_artifact:
            dataset = (
                load_dataset(
                    config.dataset_name,
                    config.dataset_config_name,
                    cache_dir=config.cache_dir,
                    split="train",
                )
                if config.dataset_name is not None else
                load_dataset(
                    "imagefolder",
                    data_dir=config.train_data_dir,
                    cache_dir=config.cache_dir,
                    split="train"
                )
            )
        else:
            artifact = wandb.use_artifact(config.dataset_name, type='dataset')
            artifact_dir = artifact.download()
            config.train_data_dir = artifact_dir
            dataset = load_dataset(
                "imagefolder",
                data_dir=config.train_data_dir,
                cache_dir=config.cache_dir,
                split="train"
            )

        dataset.set_transform(transforms)
        return torch.utils.data.DataLoader(
            dataset, batch_size=config.train_batch_size,
            shuffle=True,
            num_workers=config.dataloader_num_workers
        )

    return (build_dataloader,)


@app.cell
def _(
    Accelerator,
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    EMAModel,
    F,
    build_dataloader,
    build_unet_model,
    config,
    get_scheduler,
    math,
    torch,
    tqdm,
    wandb,
):
    def training_loop():
        # Initialize Accelerator
        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="wandb"
        )

        if accelerator.is_main_process:
            accelerator.init_trackers(
                project_name=config.wandb_project, 
                init_kwargs={
                    "wandb": {
                        'entity': config.wandb_entity,
                        'config': config.to_dict()
                    }
                }
            )
            wandb_table = wandb.Table(
                columns=['Epoch', 'Step', 'Generated-Images']
            )
    
        # Initialize Train Dataloader
        train_dataloader = build_dataloader()
    
        # Initialize Model
        model = build_unet_model()
    
        # Initialize Diffusion Pipeline
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps
        ) if config.diffusion_pipeline == "ddpm" else DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps
        )
    
        # Initialize AdamW optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon,
        )
    
        # Initialize Learning Rate Scheduler
        lr_scheduler = get_scheduler(
            config.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * config.num_epochs
            ) // config.gradient_accumulation_steps,
        )

        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / config.gradient_accumulation_steps
        )

        ema_model = EMAModel(
            model,
            inv_gamma=config.ema_inv_gamma,
            power=config.ema_power,
            max_value=config.ema_max_decay
        )
    
        global_step = 0
        for epoch in range(config.num_epochs):
            model.train()

            progress_bar = tqdm(
                total=num_update_steps_per_epoch,
                disable=not accelerator.is_local_main_process
            )
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch["input"]
                # Sample noise that we'll add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bsz = clean_images.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude
                # at each timestep (this is the forward diffusion process)
                noisy_images = noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                )

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps).sample
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    if config.use_ema:
                        ema_model.step(model)
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step
                # behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "step": global_step
                }
                if config.use_ema:
                    logs["ema_decay"] = ema_model.decay
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            accelerator.log({'epoch':epoch}, step=global_step)
            progress_bar.close()

            accelerator.wait_for_everyone()

            # Generate sample images for visual inspection
            if accelerator.is_main_process:
                if epoch % config.save_images_epochs == 0 or epoch == config.num_epochs - 1:
                    pipeline = DDPMPipeline(
                        unet=accelerator.unwrap_model(
                            ema_model.averaged_model if config.use_ema else model
                        ),
                        scheduler=noise_scheduler,
                    ) if config.diffusion_pipeline == "ddpm" else DDIMPipeline(
                        unet=accelerator.unwrap_model(
                            ema_model.averaged_model if config.use_ema else model
                        ),
                        scheduler=noise_scheduler,
                    )

                    generator = torch.manual_seed(0)
                    # run pipeline in inference (sample random noise and denoise)
                    images = pipeline(
                        generator=generator,
                        batch_size=config.eval_batch_size,
                        output_type="numpy"
                    ).images

                    # denormalize the images and save to wandb
                    images_processed = (images * 255).round().astype("uint8")
                    wandb_images = [wandb.Image(i) for i in images_processed]

                
                    wandb_table.add_data(
                        epoch,
                        global_step,
                        wandb_images[:config.num_images_in_table]
                    )

                    wandb.log({'generated_images':wandb_images,}, step=global_step)

                if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    # save the model
                    pipeline.save_pretrained(config.output_dir)

                    # log wandb artifact
                    model_artifact = wandb.Artifact(
                        f'{wandb.run.id}-{config.output_dir}', 
                        type='model'
                        )
                    model_artifact.add_dir(config.output_dir)
                    wandb.log_artifact(
                        model_artifact,
                        aliases=[f'step_{global_step}', f'epoch_{epoch}']
                    )

            accelerator.wait_for_everyone()

        wandb.log({'Generated-Images-Table': wandb_table})
        accelerator.end_training()

    return (training_loop,)


@app.cell
def _(config, notebook_launcher, training_loop):
    notebook_launcher(training_loop, num_processes=config.num_processes)
    return


if __name__ == "__main__":
    app.run()
