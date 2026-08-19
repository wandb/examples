# /// script
# dependencies = ["ultralytics", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/ultralytics/00_inference.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{ultralytics-inference} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](./assets/interactive_bbox.gif)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Install Dependencies

    - Install Ultralytics using `pip install ultralytics`. In order to learn about more ways to install Ultralytics, you can check out the [official docs](https://docs.ultralytics.com/quickstart/#install-ultralytics).

    - Then, you need to install the [`feat/ultralytics`](https://github.com/wandb/wandb/tree/feat/ultralytics) branch from W&B, which currently houses the out-of-the-box integration for Ultralytics.
    """)
    return


@app.cell
def _():
    # Install WandB and Ultralytics
    # packages added via marimo's package management: wandb ultralytics !pip install -q -U wandb ultralytics
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note:** The Ultralytcs integration will be soon available as a fully supported feature on Weights & Biases once [this pull request](https://github.com/wandb/wandb/pull/5867) is merged.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using Ultralytics with Weights & Biases

    In order to use the W&B integration with Ultralytics, we need to import the `wandb.yolov8.add_wandb_callback` function.
    """)
    return


@app.cell
def _():
    import wandb
    from wandb.integration.ultralytics import add_wandb_callback

    from ultralytics import YOLO

    return YOLO, add_wandb_callback, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, let us download a few images to test the integration on. You can use your own images, videos or camera sources. For more information on inference sources, you can check out the [official docs](https://docs.ultralytics.com/modes/predict/).
    """)
    return


@app.cell
def _(subprocess):
    #! wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png
    subprocess.call(['wget', 'https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img1.png'])
    #! wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png
    subprocess.call(['wget', 'https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img2.png'])
    #! wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png
    subprocess.call(['wget', 'https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img4.png'])
    #! wget https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png
    subprocess.call(['wget', 'https://raw.githubusercontent.com/wandb/examples/ultralytics/colabs/ultralytics/assets/img5.png'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we initialize a W&B [run](https://docs.wandb.ai/guides/runs) using `wandb.init`.
    """)
    return


@app.cell
def _(wandb):
    # Initialize Weights & Biases run
    wandb.init(project="ultralytics", job_type="inference")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, we initialize the `YOLO` model of our choice, and invoke the `add_wandb_callback` function on it before performing inference with the model. This would ensure that when we perform inference, it would automatically log the images overlayed with our [interactive overlays for computer vision tasks](https://docs.wandb.ai/guides/track/log/media#image-overlays-in-tables) along with additional insights in a [`wandb.Table`](https://docs.wandb.ai/guides/data-vis).
    """)
    return


@app.cell
def _(YOLO, add_wandb_callback, wandb):
    model_name = 'yolov8n' #@param {type:"string"}

    # Initialize YOLO Model
    model = YOLO(f"{model_name}.pt")

    # Add Weights & Biases callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    # Perform prediction which automatically logs to a W&B Table
    # with interactive overlays for bounding boxes, segmentation masks
    model(["./assets/img1.jpeg", "./assets/img3.png", "./assets/img4.jpeg", "./assets/img5.jpeg"])

    # Finish the W&B run
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next, you can check out the following notebook to learn how to perform experiment tracking and visualize validation predictions during training using Weights & Biases in the following notebook:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/ultralytics-train)

    In order to learn more about using Weights & Biases with Ultralytics, you can also read the report: [**Supercharging Ultralytics with Weights & Biases**](https://wandb.ai/geekyrakshit/ultralytics/reports/Supercharging-Ultralytics-with-Weights-Biases--Vmlldzo0OTMyMDI4)
    """)
    return


if __name__ == "__main__":
    app.run()
