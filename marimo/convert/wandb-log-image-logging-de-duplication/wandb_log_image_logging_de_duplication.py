# /// script
# dependencies = ["wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Image_Logging_de_duplication.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qqq
    return


@app.cell
def _():
    import PIL
    import numpy as np

    def write_img(path):
        PIL.Image.fromarray(np.random.rand(128,128), mode="L").save(path)

    def setup_demo(num_images=10):
        paths = []
        for ndx in range(num_images):
          path = f"./img_{ndx}.png"
          write_img(path)
          paths.append(path)
        return paths

    return (setup_demo,)


@app.cell
def _(setup_demo):
    IMAGE_PATHS = setup_demo()
    return (IMAGE_PATHS,)


@app.cell
def _(IMAGE_PATHS):
    import wandb
    wandb.init(project='image_docs')
    # Step 1: Add your Images to an Artifact
    _art = wandb.Artifact('my_images', 'dataset')
    for path in IMAGE_PATHS:
        _art.add(wandb.Image(path), path)
    wandb.log_artifact(_art)
    wandb.finish()
    return (wandb,)


@app.cell
def _(IMAGE_PATHS, wandb):
    run = wandb.init(project='image_docs')
    _art = wandb.use_artifact('my_images:latest')
    img_1 = _art.get(IMAGE_PATHS[0])
    wandb.log({'image': img_1})
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
