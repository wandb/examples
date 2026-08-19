# /// script
# dependencies = ["requirements-txt", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/kaolin_wisp/VQAD.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{kaolin-wisp-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Kaolin-Wisp + WandB Demo 🪄🐝

    <!--- @wandbcode{kaolin-wisp-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install Kaolin Core and Kaolin Wisp
    """)
    return


@app.cell
def _(subprocess):
    # Install OpenEXR
    #! sudo apt-get update
    subprocess.call(['sudo', 'apt-get', 'update'])
    #! sudo apt-get install libopenexr-dev
    subprocess.call(['sudo', 'apt-get', 'install', 'libopenexr-dev'])
    subprocess.call(['git', 'clone', '--recursive', 'https://github.com/NVIDIAGameWorks/kaolin'])
    # Install Kaolin
    #! git clone --recursive https://github.com/NVIDIAGameWorks/kaolin
    import os
    os.chdir('kaolin')
    subprocess.call(['python', 'setup.py', 'develop'])
    #! python setup.py develop
    subprocess.call(['python', '-c', 'import kaolin; print(kaolin.__version__)'])
    #! python -c "import kaolin; print(kaolin.__version__)"
    os.chdir('..')
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/NVIDIAGameWorks/kaolin-wisp'])
    os.chdir('kaolin-wisp')
    subprocess.call(['python', 'setup.py', 'develop'])
    # Install Kaolin-Wisp
    #! git clone --depth 1 https://github.com/NVIDIAGameWorks/kaolin-wisp
    # packages added via marimo's package management: requirements.txt !pip install -q -r requirements.txt
    # packages added via marimo's package management: wandb !pip install -q --upgrade wandb
    os.chdir('..')
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download Sample Data for a V8 Model Engine
    """)
    return


@app.cell
def _(subprocess):
    # Download Dataset
    #! gdown https://drive.google.com/uc?id=18hY0DpX2bK-q9iY_cog5Q0ZI7YEjephE
    subprocess.call(['gdown', 'https://drive.google.com/uc?id=18hY0DpX2bK-q9iY_cog5Q0ZI7YEjephE'])
    #! unzip -q V8.zip
    subprocess.call(['unzip', '-q', 'V8.zip'])
    #! rm V8.zip
    subprocess.call(['rm', 'V8.zip'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train VQAD
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A great aspect of Kaolin Wisp is that it comes with the goodness of [Weights & Biases](https://wandb.ai/site) integrated with itself!!!

    To track training and validation metrics, render 3D interactive plots, reproduce your configurations and results, and many more features in your Weights & Biases workspace just add the additional flag `--wandb_project <your-project-name>` when initializing the training script.

    The complete list of features supported by Weights & Biases:

    - Log training and validation metrics in real time.

    - Log system metrics in real time.

    - Log RGB, RGBA, Depth renderings etc. during training.

    - Log interactive 360 degree renderings post training
    in all levels of detail.

    - Log model checkpoints as [Weights & Biases artifacts](https://wandb.ai/site/artifacts).

    - Sync experiment configs for reproducibility.

    - Host Tensorboard instance inside Weights & Biases run.

    The full list of optional arguments related to logging on Weights & Biases include:

    - `--wandb_project`: Name of Weights & Biases project

    - `--wandb_run_name`: Name of Weights & Biases run [Optional]
    - `--wandb_entity`: Name of Weights & Biases entity under which your project resides [Optional]

    - `--wandb_viz_nerf_angles`: Number of angles in the 360 degree renderings [Optional, default set to 20]

    - `--wandb_viz_nerf_distance`: Camera distance to visualize Scene from for 360 degree renderings on Weights & Biases [Optional, default set to 3]
    """)
    return


@app.cell
def _(os, subprocess):
    os.chdir('kaolin-wisp')
    #! WISP_HEADLESS=1 python3 app/main.py --config configs/vqad_nerf.yaml --dataset-path ../V8_/ --dataset-num-workers 4 --wandb_project "vector-quantized-auto-decoder" --wandb_run_name test-vqad-nerf/V8 --wandb_viz_nerf_distance 5
    subprocess.call(['WISP_HEADLESS=1', 'python3', 'app/main.py', '--config', 'configs/vqad_nerf.yaml', '--dataset-path', '../V8_/', '--dataset-num-workers', '4', '--wandb_project', 'vector-quantized-auto-decoder', '--wandb_run_name', 'test-vqad-nerf/V8', '--wandb_viz_nerf_distance', '5'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you wish to train using one of the numerous scenes from the [RTMV Dataset](http://www.cs.umd.edu/~mmeshry/projects/rtmv/), you can replace the gdown URL with one of the tar files from [here](https://drive.google.com/drive/folders/1cc5ArA16pEznMd92z7pwgD1Z4uBqafUN). You also need to change the `--dataset-path` paramter while training to the respective path of the model that you wish to train on.
    """)
    return


if __name__ == "__main__":
    app.run()
