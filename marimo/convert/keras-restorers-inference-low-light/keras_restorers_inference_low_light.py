# /// script
# dependencies = ["pip", "restorers @ git+https://github.com/soumik12345/restorers.git", "setuptools"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/restorers/Inference_low_light.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{restorers-inference} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌈 Restorers + WandB 🪄🐝

    <!--- @wandbcode{restorers-mirnetv2-train} -->

    This notebook shows how to perform inference with a low-light enhancement using [**restorers**](https://github.com/soumik12345/restorers) and [**wandb**](https://wandb.ai/site). For more details regarding usage of restorers, refer to the following report:

    [![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/ml-colabs/low-light-enhancement/reports/Lighting-up-Images-in-the-Deep-Learning-Era--VmlldzozNzE4Njkz)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: pip setuptools !pip install -q --upgrade pip setuptools
    # packages added via marimo's package management: git+https://github.com/soumik12345/restorers.git !pip install git+https://github.com/soumik12345/restorers.git
    return


@app.cell
def _():
    import os
    import wandb
    from restorers.inference import LowLightInferer

    return LowLightInferer, os, wandb


@app.cell
def _(wandb):
    # initialize a wandb run for inference
    wandb.init(project="low-light-enhancement", job_type="inference")
    return


@app.cell
def _(os, wandb):
    images_artifact = wandb.use_artifact('ml-colabs/low-light-enhancement/run-7ngsohcn-DarkImagesTable:v0', type='run_table')
    images_artifact_dir = images_artifact.download()
    sample_image = os.path.join(images_artifact_dir, "media/images/0b63c6b0cfdfd95675f7/image_9.png")
    return (sample_image,)


@app.cell
def _(LowLightInferer, sample_image):
    # initialize the inferer
    inferer = LowLightInferer(
        resize_factor=1, model_alias="Zero-DCE"
    )
    # intialize the model from wandb artifacts
    inferer.initialize_model_from_wandb_artifact(
        # This artifact address corresponds to a Zero-DCE model trained on the LoL dataset
        "ml-colabs/low-light-enhancement/run_oaa25znm_model:v99"
    )
    # infer on a directory of images
    # inferer.infer("./dark_images")
    # or infer on a single image
    inferer.infer(sample_image)
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
