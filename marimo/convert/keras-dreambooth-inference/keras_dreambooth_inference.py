# /// script
# dependencies = ["dreambooth-keras @ git+https://github.com/soumik12345/dreambooth-keras.git"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/dreambooth/inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{dreambooth-keras-inference} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🧨 Dreambooth-Keras + WandB 🪄🐝

    [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soumik12345/dreambooth-keras/blob/main/notebooks/inference_wandb.ipynb)

    <!--- @wandbcode{dreambooth-keras-inference} -->

    This notebook shows how to perform inference with a DreamBooth fine-tuned Stable Diffusion model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🌈 Install Dreambooth-Keras

    We would use [soumik12345/dreambooth-keras](https://github.com/soumik12345/dreambooth-keras) which is a fork of [sayakpaul/dreambooth-keras](https://github.com/sayakpaul/dreambooth-keras) developed by [**Sayak Paul**](https://github.com/sayakpaul) and [**Chansung Park**](https://github.com/deep-diver).
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: git+https://github.com/soumik12345/dreambooth-keras.git !pip install -q git+https://github.com/soumik12345/dreambooth-keras.git
    return


@app.cell
def _():
    import wandb
    from PIL import Image
    from dreambooth_keras.utils import load_model_from_wandb_artifact

    return Image, load_model_from_wandb_artifact, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🐝 Initialize WandB run

    We initialize a [Weights & Biases run](https://docs.wandb.ai/guides/runs) for storing generated images to a [Weights & Biases table](https://docs.wandb.ai/guides/data-vis).
    """)
    return


@app.cell
def _(wandb):
    wandb.init(project="dreambooth-keras", job_type="inference")

    config = wandb.config
    config.model_artifact_address = "geekyrakshit/dreambooth-keras/run_n5oakq7c_model:v0"
    config.image_resolution = 512
    config.num_diffusion_steps = 500
    config.batch_size = 5
    config.unique_id = "sks"
    config.class_category = "monkey"
    config.prompt = "a painting of sks monkey in the style of Michelangelo"
    config.unconditional_guidance_scale = 15


    wandb_table = wandb.Table(columns=[
        "prompt", "images", "unique-id", "class-category","image-resolution", "num-diffusion-steps"
    ])
    return config, wandb_table


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧑‍🎨 Perform Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    First we load our model from Weights & Biases artifacts created using the [`dreambooth_keras.utils.DreamBoothCheckpointCallback`](https://github.com/soumik12345/dreambooth-keras/blob/main/dreambooth_keras/utils.py#L93) which automatically logs model checkpoints as [Weights & Biases artifacts](https://docs.wandb.ai/guides/data-and-model-versioning) at the end of each epoch during training. We load these checkpoint using the simple utility [`dreambooth_keras.utils.load_model_from_wandb_artifact`](https://github.com/soumik12345/dreambooth-keras/blob/main/dreambooth_keras/utils.py#L23).
    """)
    return


@app.cell
def _(config, load_model_from_wandb_artifact):
    dreambooth_model = load_model_from_wandb_artifact(
        artifact_address=config.model_artifact_address,
        image_resolution=config.image_resolution
    )
    return (dreambooth_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we perform inference on our *dreamboothed* stable-diffusion model.
    """)
    return


@app.cell
def _(config, dreambooth_model):
    _images = dreambooth_model.text_to_image(config.prompt, batch_size=config.batch_size, num_steps=config.num_diffusion_steps, unconditional_guidance_scale=config.unconditional_guidance_scale)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we log our images to a [Weights & Biases table](https://docs.wandb.ai/guides/data-vis) that not only makes ut easier to visualize but also easily accessible for future reference.
    """)
    return


@app.cell
def _(Image, config, wandb, wandb_table):
    _images = [wandb.Image(Image.fromarray(image), caption=f'{i}: {config.prompt}') for i, image in enumerate(_images)]
    wandb_table.add_data(config.prompt, _images, config.unique_id, config.class_category, config.image_resolution, config.num_diffusion_steps)
    wandb.log({'Inference-Results': wandb_table})
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
