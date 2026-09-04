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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-diffusers.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{sdxl-diffusers-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Image Generation with Stable Diffusion XL using 🤗 Diffusers

    <!--- @wandbcode{sdxl-diffusers-colab} -->

    This notebook demonstrates the following:
    - Performing text-conditional image-generations with the [Stable Diffusion XL](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/stable_diffusion_xl) using [🤗 Diffusers](https://huggingface.co/docs/diffusers).
    - Manage image generation experiments using [Weights & Biases](http://wandb.ai/site).
    - Log the prompts, generated images and experiment configs to [Weigts & Biases](http://wandb.ai/site) for visalization.

    ![](./assets/diffusers-autolog-5.gif)
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
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

    import wandb
    from wandb.integration.diffusers import autolog

    return (
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
        autolog,
        random,
        torch,
        wandb,
    )


@app.cell
def _(StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline, torch):
    base_model_id = "stabilityai/stable-diffusion-xl-base-1.0" # @param ["stabilityai/stable-diffusion-xl-base-1.0", "segmind/SSD-1B", "stabilityai/sdxl-turbo"]

    base_pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    base_pipeline.enable_model_cpu_offload()

    refiner_pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base_pipeline.text_encoder_2,
        vae=base_pipeline.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner_pipeline.enable_model_cpu_offload()
    return base_pipeline, refiner_pipeline


@app.cell
def _(random, torch):
    wandb_project = "pixart-alpha" # @param {type:"string"}

    prompt_1 = "a photograph of an evil and vile looking demon in Bengali attire eating fish. The demon has large and bloody teeth. The demon is sitting on the branches of a giant Banyan tree, dimly lit, bluish and dark color palette, realistic, 8k" # @param {type:"string"}
    prompt_2 = "" # @param {type:"string"}
    negative_prompt_1 = "static, frame, painting, illustration, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing" # @param {type:"string"}
    negative_prompt_2 = "static, frame, painting, illustration, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing" # @param {type:"string"}
    num_inference_steps = 50  # @param {type:"slider", min:10, max:100, step:1}
    guidance_scale = 5.0  # @param {type:"slider", min:0, max:10, step:0.1}
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
    generator_base = torch.Generator(device="cuda").manual_seed(seed)
    generator_refiner = torch.Generator(device="cuda").manual_seed(seed)
    return (
        generator_base,
        generator_refiner,
        guidance_scale,
        negative_prompt_1,
        negative_prompt_2,
        num_inference_steps,
        prompt_1,
        prompt_2,
        wandb_project,
    )


@app.cell
def _(
    autolog,
    base_pipeline,
    generator_base,
    generator_refiner,
    guidance_scale,
    negative_prompt_1,
    negative_prompt_2,
    num_inference_steps,
    prompt_1,
    prompt_2,
    refiner_pipeline,
    wandb,
    wandb_project,
):
    # Call WandB Autolog for Diffusers. This would automatically log
    # the prompts, generated images, pipeline architecture and all
    # associated experiment configs to Weights & Biases, thus making your
    # image generation experiments easy to reproduce, share and analyze.
    autolog(init=dict(project=wandb_project))

    image = base_pipeline(
        prompt=prompt_1,
        prompt_2=prompt_2,
        negative_prompt=negative_prompt_1,
        negative_prompt_2=negative_prompt_2,
        num_inference_steps=num_inference_steps,
        output_type="latent",
        generator=generator_base,
        guidance_scale=guidance_scale,
    ).images[0]

    image = refiner_pipeline(
        prompt=prompt_1,
        prompt_2=prompt_2,
        negative_prompt=negative_prompt_1,
        negative_prompt_2=negative_prompt_2,
        image=image[None, :],
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator_refiner,
    ).images[0]

    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
