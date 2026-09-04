# /// script
# dependencies = ["", "accelerate", "diffusers", "install-log", "transformers", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/lcm-diffusers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{lcm-diffusers-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image Generation with Consistency Models using 🤗 Diffusers

    <!--- @wandbcode{lcm-diffusers-colab} -->

    This notebook demonstrates the following:
    - Performing text-conditional image-generations with the [Consistency Models](https://huggingface.co/docs/diffusers/api/pipelines/consistency_models) using [🤗 Diffusers](https://huggingface.co/docs/diffusers).
    - Manage image generation experiments using [Weights & Biases](http://wandb.ai/site).
    - Log the prompts, generated images and experiment configs to [Weigts & Biases](http://wandb.ai/site) for visalization.

    ![](./assets/diffusers-autolog-4.gif)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: diffusers transformers accelerate wandb > install.log !pip install diffusers transformers accelerate wandb > install.log
    return


@app.cell
def _():
    import random

    import torch
    from diffusers import DiffusionPipeline

    import wandb
    from wandb.integration.diffusers import autolog

    return DiffusionPipeline, autolog, torch, wandb


@app.cell
def _(DiffusionPipeline, torch):
    # Initialize the diffusion pipeline for latent consistency model
    pipeline = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    pipeline = pipeline.to(torch_device="cuda", torch_dtype=torch.float32)
    return (pipeline,)


@app.cell
def _(torch):
    # Define the prompts, negative prompts, and seed.
    prompt = [
        "a photograph of an astronaut riding a horse",
        "a photograph of a dragon"
    ]

    # Make the experiment reproducible by controlling randomness.
    # The seed would be automatically logged to WandB.
    generator = torch.Generator(device="cpu").manual_seed(10)
    return generator, prompt


@app.cell
def _(autolog, generator, pipeline, prompt, wandb):
    # Call WandB Autolog for Diffusers. This would automatically log
    # the prompts, generated images, pipeline architecture and all
    # associated experiment configs to Weights & Biases, thus making your
    # image generation experiments easy to reproduce, share and analyze.
    autolog(init=dict(project="diffusers_logging"))

    # call the pipeline to generate the images
    images = pipeline(
        prompt,
        num_images_per_prompt=2,
        generator=generator,
        num_inference_steps=10,
    )

    # End the experiment
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
