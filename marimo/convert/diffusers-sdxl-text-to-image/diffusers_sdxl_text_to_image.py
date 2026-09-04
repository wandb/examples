# /// script
# dependencies = ["diffusers", "transformers", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/diffusers/sdxl-text-to-image.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stable-Diffusion XL 1.0 using 🤗 Diffusers

    This notebook demonstrates the following:
    - Performing text-conditional image-generations using [🤗 Diffusers](https://huggingface.co/docs/diffusers).
    - Using the Stable Diffusion XL Refiner pipeline to further refine the outputs of the base model.
    - Manage image generation experiments using [Weights & Biases](http://wandb.ai/geekyrakshit).
    - Log the prompts and generated images to [Weigts & Biases](http://wandb.ai/geekyrakshit) for visalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Installing the Dependencies
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: diffusers[torch] transformers wandb !pip install -qq diffusers["torch"] transformers wandb
    return


@app.cell
def _():
    import torch
    import wandb
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        EulerDiscreteScheduler
    )

    return (
        EulerDiscreteScheduler,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLPipeline,
        torch,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experiment Management using Weights & Biases

    Managing our image generation experiments is crucial for the sake of reproducibility. Hence we sync all the configs of our experiments with our Weights & Biases run. This stores all the configs of the experiments, right from the prompts to the refinement technque and the configuration of the scheduler.
    """)
    return


@app.cell
def _(wandb):
    project_name = "stable-diffusion-xl" # @param {type:"string"}

    # initialize a wandb run
    wandb.init(project=project_name, job_type="text-to-image")

    # define experiment configs
    config = wandb.config
    config.stable_diffusion_checkpoint = "stabilityai/stable-diffusion-xl-base-1.0" # @param ["stabilityai/stable-diffusion-xl-base-1.0", "stabilityai/stable-diffusion-xl-base-0.9"] {allow-input: true}
    config.refiner_checkpoint = "stabilityai/stable-diffusion-xl-refiner-1.0" # @param ["stabilityai/stable-diffusion-xl-refiner-1.0", "stabilityai/stable-diffusion-xl-refiner-0.9"] {allow-input: true}
    config.compile_model = False
    config.prompt_1 = "a photograph of an evil and vile looking demon in Bengali attire eating fish. The demon has large and bloody teeth. The demon is sitting on the branches of a giant Banyan tree, dimly lit, bluish and dark color palette, realistic, 8k" # @param {type:"string"}
    config.prompt_2 = "" # @param {type:"string"}
    config.negative_prompt_1 = "static, frame, painting, illustration, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing" # @param {type:"string"}
    config.negative_prompt_2 = "static, frame, painting, illustration, sd character, low quality, low resolution, greyscale, monochrome, nose, cropped, lowres, jpeg artifacts, deformed iris, deformed pupils, bad eyes, semi-realistic worst quality, bad lips, deformed mouth, deformed face, deformed fingers, deformed toes standing still, posing" # @param {type:"string"}
    config.base_guidance_scale = 5.0 # @param {type:"slider", min:1, max:10, step:0.1}
    config.seed = 0 # @param {type:"raw"}
    config.num_inference_steps = 100 # @param {type:"slider", min:1, max:500, step:1}

    config.enable_cpu_offload_base = True  # @param {type:"boolean"}
    config.enable_cpu_offload_refiner = True  # @param {type:"boolean"}

    config.compile_base_model = False # @param {type:"boolean"}

    # Enable refinement only if high-ram instance
    config.enable_refinement = False  # @param {type:"boolean"}
    config.compile_refinement_model = False # @param {type:"boolean"}
    config.refiner_guidance_scale = 5.0 # @param {type:"slider", min:1, max:10, step:0.1}
    config.num_refinement_steps = 150 # @param {type:"slider", min:1, max:500, step:1}

    # Set explicitly only if config.use_ensemble_of_experts is True
    config.high_noise_fraction = 0.8 # @param {type:"slider", min:0, max:1, step:0.1}

    beta_schedule = "scaled_linear" # @param ["linear", "scaled_linear"]
    interpolation_type =  "linear" # @param ["linear", "log_linear"] {allow-input: true}
    prediction_type = "epsilon" # @param ["epsilon", "sample", "v_prediction"]
    timestep_spacing =  "leading" # @param ["linspace", "leading"] {allow-input: true}

    # configs for diffusers.EulerDiscreteScheduler
    scheduler_kwargs = {
        "beta_end": 0.012,
        "beta_schedule": beta_schedule,
        "beta_start": 0.00085,
        "interpolation_type": interpolation_type,
        "num_train_timesteps": 1000,
        "prediction_type": prediction_type,
        "steps_offset": 1,
        "timestep_spacing": timestep_spacing,
        "trained_betas": None,
        "use_karras_sigmas": False,
    }

    config.scheduler_kwargs = scheduler_kwargs
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can make the experiment deterministic based on the seed specified in the experiment configs.
    """)
    return


@app.cell
def _(config, torch):
    generator = [torch.Generator(device="cuda")]
    if config.seed:
        generator = [g.manual_seed(config.seed) for g in generator]
    return (generator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Base Diffusion Pipelines

    For performing text-conditional image generation, we use the `diffusers` library to define the diffusion pipelines corresponding to the base SDXL model and the SDXL refinement model.

    1. We define the base diffusion pipeline using `diffusers.DiffusionPipeline` and load the pre-trained weights for SDXL 1.0 by calling the `from_pretrained` function on it. We also pass the scheduler as `diffusers.EulerDiscreteScheduler` in this  step.

    2. In case we don't have a GPU with large enough GPU, it's recommended to enable CPU offloading. Otherwise, we load the model on the GPU. In case you're curious how HuggingFace manages CPU offloading in the most optimized manner, we recommend you read this port by [Sylvain Gugger](https://huggingface.co/sgugger): [How 🤗 Accelerate runs very large models thanks to PyTorch](https://huggingface.co/blog/accelerate-large-models).

    3. We can compile model using `torch.compile`, this might give a significant speedup.

    4. We generate the image from the prompts and negative prompts using the base pipeline.
    """)
    return


@app.cell
def _(
    EulerDiscreteScheduler,
    StableDiffusionXLPipeline,
    config,
    generator,
    torch,
):
    # Define the Base Pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        config.stable_diffusion_checkpoint,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        scheduler=EulerDiscreteScheduler(**config.scheduler_kwargs),
    )

    if config.enable_cpu_offload_base:
        # Offload base pipeline to CPU
        pipe.enable_model_cpu_offload()
    else:
        # Load base pipeline to GPU
        pipe.to("cuda")

    # Compile model using `torch.compile`, this might give a significant speedup
    if config.compile_base_model:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    # Generate image from the prompts and negative prompts using the base pipeline
    generated_image = pipe(
        prompt=config.prompt_1,
        prompt_2=config.prompt_2,
        negative_prompt=config.negative_prompt_1,
        negative_prompt_2=config.negative_prompt_2,
        guidance_scale=config.base_guidance_scale,
        output_type="latent" if config.enable_refinement else "pil",
        num_inference_steps=config.num_inference_steps,
        denoising_end=config.high_noise_fraction if config.enable_refinement else None,
        generator=generator,
    ).images
    return generated_image, pipe


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Refining the Generated Image

    For refining the image generated by the base pipeline, we using the SDXL Refiner pipeline using the base and refiner model as an ensemble of expert of denoisers. In this case, the base model should serve as the expert for the high-noise diffusion stage and the refiner serves as the expert for the low-noise diffusion stage.

    1. We define the diffusion pipeline for the refiner using `diffusers.DiffusionPipeline` and load the pre-trained weights for SDXL 1.0 refiner by calling the `from_pretrained` function on it. We also pass the scheduler as `diffusers.EulerDiscreteScheduler` in this  step.

    2. In case we don't have a GPU with large enough GPU, it's recommended to enable CPU offloading. Otherwise, we load the model on the GPU. In case you're curious how HiggingFace manages CPU offloading in the most optimized manner, we recommend you read this port by [Sylvain Gugger](https://huggingface.co/sgugger): [How 🤗 Accelerate runs very large models thanks to PyTorch](https://huggingface.co/blog/accelerate-large-models).

    3. We can compile model using `torch.compile`, this might give a significant speedup.

    4. We refine the latents generated by the base model from the same set of prompts and negative prompts using the refiner pipeline.
    """)
    return


@app.cell
def _(
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    config,
    generated_image,
    generator,
    pipe,
    torch,
):
    if config.enable_refinement:
        refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(config.refiner_checkpoint, text_encoder_2=pipe.text_encoder_2, vae=pipe.vae, torch_dtype=torch.float16, use_safetensors=True, variant='fp16', scheduler=EulerDiscreteScheduler(**config.scheduler_kwargs))
        if config.enable_cpu_offload_refiner:
            refiner.enable_model_cpu_offload()
        else:
            refiner.to('cuda')
        if config.compile_refinement_model:
            refiner.unet = torch.compile(pipe.unet, mode='reduce-overhead', fullgraph=True)
        generated_image_1 = refiner(prompt=config.prompt_1, prompt_2=config.prompt_2, negative_prompt=config.negative_prompt_1, negative_prompt_2=config.negative_prompt_2, guidance_scale=config.refiner_guidance_scale, image=generated_image, num_inference_steps=config.num_refinement_steps, denoising_start=config.high_noise_fraction, generator=generator).images  # Compile model using `torch.compile`, this might give a significant speedup
    return (generated_image_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Logging the Images to Weights & Biases

    Now, we log the images to Weights & Biases. This enables us to:

    - Visualize our generations
    - Examine the generated images across different images
    - Ensure reproducibility of the experiments
    """)
    return


@app.cell
def _(config, generated_image_1, wandb):
    # Create a [wandb table](https://docs.wandb.ai/guides/tables)
    table = wandb.Table(columns=['Prompt-1', 'Prompt-2', 'Negative-Prompt-1', 'Negative-Prompt-2', 'Generated-Image'])
    generated_image_2 = wandb.Image(generated_image_1[0])
    table.add_data(config.prompt_1, config.prompt_2, config.negative_prompt_1, config.negative_prompt_2, generated_image_2)
    wandb.log({'Generated-Image': generated_image_2, 'Text-to-Image': table})
    # Add the images to the table
    # Log the images and table to wandb
    # finish the experiment
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's how you can examine your generations across multiple experiments 👇

    ![](https://i.imgur.com/zNynGye.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's how you can manage your prompts and your generations across experiments 👇

    ![](https://i.imgur.com/JVEXkx0.png)
    """)
    return


if __name__ == "__main__":
    app.run()
