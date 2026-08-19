# /// script
# dependencies = ["", "accelerate", "diffusers", "ftfy", "install-log", "sentencepiece", "transformers", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/pixart-alpha-diffusers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image Generation with Pixart-α using 🤗 Diffusers

    This notebook demonstrates the following:
    - Performing text-conditional image-generations with the [Pixart-α model](https://huggingface.co/docs/diffusers/v0.23.1/en/api/pipelines/pixart) using [🤗 Diffusers](https://huggingface.co/docs/diffusers).
    - Manage image generation experiments using [Weights & Biases](http://wandb.ai/site).
    - Log the prompts, generated images and experiment configs to [Weigts & Biases](http://wandb.ai/site) for visalization.

    ![](./assets/diffusers-autolog-1.gif)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: diffusers transformers accelerate sentencepiece ftfy wandb > install.log !pip install diffusers transformers accelerate sentencepiece ftfy wandb > install.log
    return


@app.cell
def _():
    import random

    import torch
    from diffusers import PixArtAlphaPipeline

    import wandb
    from wandb.integration.diffusers import autolog

    return PixArtAlphaPipeline, autolog, random, torch, wandb


@app.cell
def _(PixArtAlphaPipeline, torch):
    # Load the pre-trained checkpoints from HuggingFace Hub to the PixArtAlphaPipeline
    pipe = PixArtAlphaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16
    )

    # Enable offloading the weights to the CPU and only loading them on the GPU when
    # performing the forward pass can also save memory.
    pipe.enable_model_cpu_offload()
    return (pipe,)


@app.cell
def _(random, torch):
    wandb_project = "pixart-alpha" # @param {type:"string"}

    prompt = "a traveler navigating via a boat in countless mountains, Chinese ink painting" # @param {type:"string"}
    negative_prompt = "" # @param {type:"string"}
    num_inference_steps = 25  # @param {type:"slider", min:10, max:50, step:1}
    guidance_scale = 4.5  # @param {type:"slider", min:0, max:10, step:0.1}
    num_images_per_prompt = 1 # @param {type:"slider", min:0, max:10, step:0.1}
    height = 1024 # @param {type:"slider", min:512, max:2560, step:32}
    width = 1024 # @param {type:"slider", min:512, max:2560, step:32}
    seed = None # @param {type:"raw"}


    def autogenerate_seed():
        max_seed = int(1024 * 1024 * 1024)
        seed = random.randint(1, max_seed)
        seed = -seed if seed < 0 else seed
        seed = seed % max_seed
        return seed


    seed = autogenerate_seed() if seed is None else seed

    # Make the experiment reproducible by controlling randomness.
    # The seed would be automatically logged to WandB.
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return (
        generator,
        guidance_scale,
        height,
        negative_prompt,
        num_images_per_prompt,
        num_inference_steps,
        prompt,
        wandb_project,
        width,
    )


@app.cell
def _(
    autolog,
    generator,
    guidance_scale,
    height,
    negative_prompt,
    num_images_per_prompt,
    num_inference_steps,
    pipe,
    prompt,
    wandb,
    wandb_project,
    width,
):
    # Call WandB Autolog for Diffusers. This would automatically log
    # the prompts, generated images, pipeline architecture and all
    # associated experiment configs to Weights & Biases, thus making your
    # image generation experiments easy to reproduce, share and analyze.
    autolog(init=dict(project=wandb_project))

    # Generate the images by calling the PixArtAlphaPipeline
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        generator=generator,
    ).images[0]

    # End the experiment
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
