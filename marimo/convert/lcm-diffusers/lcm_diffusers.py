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

__generated_with = "0.24.2"
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
    - Log prompts, generated images, and experiment configuration to W&B for visualization.

    ![Diffusers autologging in W&B](https://raw.githubusercontent.com/wandb/examples/main/colabs/diffusers/assets/diffusers-autolog-4.gif)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, your team or entity. You can leave the key blank when this environment already has W&B credentials. If this is your first time using W&B, [create a free account](https://wandb.ai/signup).

    A [Hugging Face token](https://huggingface.co/settings/tokens) is optional. Supplying one avoids anonymous Hub rate limits and can make the initial model download more reliable.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use cached credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="W&B entity or team (optional)",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    _hf_token_input = mo.ui.text(
        kind="password",
        label="Hugging Face token (optional)",
        placeholder="Uses cached Hugging Face credentials when blank",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{hf_token}")
        .batch(
            api_key=_api_key_input,
            entity=_entity_input,
            hf_token=_hf_token_input,
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Connect to W&B above before generating and logging images."),
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _entity = wandb_login_form.value["entity"].strip()
    hf_token = wandb_login_form.value["hf_token"].strip() or None
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
        _login_error = None
    except wandb.errors.Error as _error:
        _login_ok = False
        _login_error = str(_error)

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. "
                f"Check the API key and try again.\n\nW&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {
        "project": "diffusers_logging",
        "entity": _entity or None,
    }
    mo.callout(mo.md("Connected to W&B."), kind="success")
    return hf_token, wandb_settings


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate and log images

    The example uses two prompts and a fixed seed so you can reproduce the result. When you click the button below, the notebook downloads `SimianLuo/LCM_Dreamshaper_v7`, generates four images, and creates one W&B run in the `diffusers_logging` project. It uses CUDA, Apple Silicon's MPS backend, or CPU, in that order.
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
    # The seed is automatically logged to W&B.
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
    """Generate images and let W&B Diffusers autolog capture the experiment."""
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

    # W&B Diffusers autologging records the prompts, generated images, pipeline
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
        label="Download the model, generate images, and log to W&B"
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
                    f"`{seed}`, and creates one remote W&B run{_entity_note}."
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
        f"[Open the W&B run]({_run_url}) to inspect the prompts, images, and configuration."
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
