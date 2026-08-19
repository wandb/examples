# /// script
# dependencies = ["flax", "torch", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Run_names_visualized_using_min_dalle.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Visualize W&B Run Names using Craiyon
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use min-dalle, https://github.com/kuprel/min-dalle, a repo with the bare essentials necessary for doing inference on the Craiyon model. We use `wandb.Api` to get the project and run names from your account.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Setup
    """)
    return


@app.cell
def _(subprocess):
    #@title Setup
    #! git clone --depth 1 https://github.com/kuprel/min-dalle
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/kuprel/min-dalle'])
    #! git lfs install
    subprocess.call(['git', 'lfs', 'install'])
    #! git clone https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384 /content/min-dalle/pretrained/vqgan
    subprocess.call(['git', 'clone', 'https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384', '/content/min-dalle/pretrained/vqgan'])
    # packages added via marimo's package management: torch flax==0.4.2 wandb !pip install torch flax==0.4.2 wandb
    #! wandb artifact get --root=/content/min-dalle/pretrained/dalle_bart_mini dalle-mini/dalle-mini/mini-1:v0
    subprocess.call(['wandb', 'artifact', 'get', '--root=/content/min-dalle/pretrained/dalle_bart_mini', 'dalle-mini/dalle-mini/mini-1:v0'])
    return


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Generate Images using W&B Run Names
    """)
    return


@app.cell
def _():
    import wandb
    import os
    os.chdir('/content/min-dalle')
    from min_dalle.min_dalle_torch import MinDalleTorch
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    mega = False
    model = MinDalleTorch(mega)
    seed = 7
    api = wandb.Api()
    projects = [project.name for project in api.projects()]

    project_dropdown = widgets.Dropdown(
        options = projects,
        description = 'Projects:',
    )

    run_dropdown = widgets.Dropdown(
        options=[run.name for run in api.runs(project_dropdown.value)],
        description = 'Run names:',
    )

    image_output = widgets.Output()

    def on_project_value_change(change):
        run_dropdown.options = [run.name for run in api.runs(project_dropdown.value)]

    button = widgets.Button(
        description='Generate',
        tooltip='Click me to create an image from the currently selected run name',
    )
    def generate(b):
        button.disabled = True
        run_name = run_dropdown.value.replace('-', ' ')
        image = model.generate_image(run_name, seed=seed)
        with image_output:
          clear_output(wait=True)
          display(image, run_name)
        button.disabled = False
    button.on_click(generate)

    project_dropdown.observe(on_project_value_change, 'value')
    widgets.VBox([project_dropdown, run_dropdown, button, image_output])
    return (wandb,)


if __name__ == "__main__":
    app.run()
