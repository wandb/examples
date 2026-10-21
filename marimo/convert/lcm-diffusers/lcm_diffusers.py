# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "accelerate>=0.31",
#     "diffusers>=0.40",
#     "marimo>=0.24.0",
#     "torch>=2.6",
#     "torchvision>=0.21",
#     "transformers>=4.41.2",
#     "wandb>=0.18",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(auto_download=["html"])

with app.setup:
    import marimo as mo
    import torch
    from diffusers import DiffusionPipeline

    import wandb
    from wandb.integration.diffusers import autolog

    MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Image Generation with Consistency Models using Diffusers

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/lcm-diffusers/lcm_diffusers.py/server)

    This notebook demonstrates how to:

    - Generate images from text with [Latent Consistency Models](https://huggingface.co/docs/diffusers/api/pipelines/latent_consistency_models) and [Diffusers](https://huggingface.co/docs/diffusers).
    - Manage image-generation experiments with [Weights & Biases](https://wandb.ai/site).
    - Log prompts, generated images, and experiment configuration to Weights & Biases for visualization.

    ![Diffusers autologging in Weights & Biases](https://raw.githubusercontent.com/wandb/examples/main/colabs/diffusers/assets/diffusers-autolog-4.gif)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `diffusers_logging`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="diffusers_logging", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    entity = _entity
    project = _project
    wandb_settings = {"entity": entity, "project": project}
    mo.md("✓ Connected to Weights & Biases.")
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Optional Hugging Face authentication

    A [Hugging Face token](https://huggingface.co/settings/tokens) avoids anonymous Hub rate limits. Leave the field blank to use cached Hugging Face credentials.
    """)
    return


@app.cell(hide_code=True)
def _():
    hf_token_form = mo.ui.text(kind="password", label="Hugging Face token (optional)").form(
        submit_button_label="Set Hugging Face token"
    )
    hf_token_form
    return (hf_token_form,)


@app.cell(hide_code=True)
def _(hf_token_form):
    hf_token = (hf_token_form.value or "").strip() or None
    return (hf_token,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate and log images

    The example uses two prompts and a fixed seed so you can reproduce the result. When you click the button below, the notebook downloads `SimianLuo/LCM_Dreamshaper_v7`, generates four images, and creates one Weights & Biases run in your selected project. It uses CUDA, Apple Silicon's MPS backend, or CPU, in that order.
    """)
    return


@app.cell
def _():
    # Define the prompts and seed.
    prompt = [
        "a photograph of an astronaut riding a horse",
        "a photograph of a dragon",
    ]

    # Make the experiment reproducible by controlling randomness.
    # The seed is automatically logged to Weights & Biases.
    seed = 10

    # LCMs are designed for fast inference in very few denoising steps.
    num_inference_steps = 4

    # Keep the original four-image output. Set this to 1 for a faster smoke test.
    num_images_per_prompt = 2
    return num_images_per_prompt, num_inference_steps, prompt, seed


@app.function
def generate_images_with_wandb(
    prompt,
    seed,
    num_inference_steps,
    num_images_per_prompt,
    wandb_settings,
    hf_token,
):
    """Generate images and let Weights & Biases Diffusers autolog capture the experiment."""
    if wandb.run is not None:
        wandb.finish()

    if torch.cuda.is_available():
        torch_device = "cuda"
    elif torch.backends.mps.is_available():
        torch_device = "mps"
    else:
        torch_device = "cpu"
    torch_dtype = (
        torch.float16 if torch_device in {"cuda", "mps"} else torch.float32
    )

    # Initialize the diffusion pipeline for a latent consistency model.
    pipeline = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        dtype=torch_dtype,
        token=hf_token,
    )
    pipeline = pipeline.to(torch_device)
    if torch_device == "mps":
        # Reduce memory pressure and swapping on Apple Silicon systems below 64 GB.
        pipeline.enable_attention_slicing()

    # A CPU generator produces repeatable results on both CPU and GPU runtimes.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Weights & Biases Diffusers autologging records the prompts, generated images, pipeline
    # architecture, and experiment configuration used by the pipeline call.
    try:
        autolog(init=wandb_settings)
        images = pipeline(
            prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
        ).images
        run_url = wandb.run.url if wandb.run is not None else None
    finally:
        autolog.disable()
        if wandb.run is not None:
            wandb.finish()

    return images, run_url, torch_device


@app.cell(hide_code=True)
def _(
    num_images_per_prompt,
    num_inference_steps,
    prompt,
    seed,
    wandb_settings,
):
    generate_button = mo.ui.run_button(
        label="Download the model, generate images, and log to Weights & Biases"
    )
    _entity_note = (
        f" for entity `{wandb_settings['entity']}`"
        if wandb_settings["entity"]
        else " using your default entity"
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"This runs `{generate_images_with_wandb.__name__}`, downloads "
                    f"model weights, generates {len(prompt) * num_images_per_prompt} "
                    f"images in {num_inference_steps} denoising steps with seed "
                    f"`{seed}`, and creates one remote Weights & Biases run{_entity_note}."
                ),
                kind="warn",
            ),
            generate_button,
        ]
    )
    return (generate_button,)


@app.cell(hide_code=True)
def _(
    generate_button,
    hf_token,
    num_images_per_prompt,
    num_inference_steps,
    prompt,
    seed,
    wandb_settings,
):
    mo.stop(
        not generate_button.value,
        mo.callout(
            mo.md("Click the button above when you're ready to run the example."),
            kind="info",
        ),
    )
    generation_request = {
        "prompt": tuple(prompt),
        "seed": seed,
        "num_inference_steps": num_inference_steps,
        "num_images_per_prompt": num_images_per_prompt,
        "wandb_settings": dict(wandb_settings),
        "hf_token": hf_token,
    }
    return (generation_request,)


@app.cell
def _(generation_request):
    images, run_url, torch_device = generate_images_with_wandb(
        prompt=list(generation_request["prompt"]),
        seed=generation_request["seed"],
        num_inference_steps=generation_request["num_inference_steps"],
        num_images_per_prompt=generation_request["num_images_per_prompt"],
        wandb_settings=generation_request["wandb_settings"],
        hf_token=generation_request["hf_token"],
    )
    generation_result = {
        "images": images,
        "run_url": run_url,
        "torch_device": torch_device,
    }
    return (generation_result,)


@app.cell(hide_code=True)
def _(generation_result):
    _run_url = generation_result["run_url"]
    _run_message = (
        f"Generation finished on `{generation_result['torch_device']}`. "
        f"[Open the Weights & Biases run]({_run_url}) to inspect the prompts, images, and configuration."
        if _run_url
        else f"Generation finished on `{generation_result['torch_device']}`."
    )
    mo.vstack(
        [
            mo.callout(mo.md(_run_message), kind="success"),
            mo.hstack(
                [
                    mo.image(
                        _image,
                        alt=f"Generated image {_index}",
                        rounded=True,
                    )
                    for _index, _image in enumerate(
                        generation_result["images"], start=1
                    )
                ],
                wrap=True,
                widths="equal",
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
