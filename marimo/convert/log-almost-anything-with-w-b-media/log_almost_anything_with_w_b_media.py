# /// script
# dependencies = ["soundfile", "wandb"]
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
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/master/marimo/convert/log-almost-anything-with-w-b-media/log_almost_anything_with_w_b_media.py/server)
    <!--- @wandbcode{media-video} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{media-video} -->

    Use [Weights & Biases](https://wandb.com) for machine learning experiment tracking, dataset versioning, and project collaboration.

    <div><img /></div>

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    <div><img /></div>

    # Log (Almost) Anything with W&B Media
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In this notebook, we'll show you how to visualize a model's predictions with Weights & Biases – images, videos, audio, tables, HTML, metrics, plots, 3D objects and point clouds.

    ### Follow along with a [video tutorial →](http://wandb.me/media-video)!
    #### View plots in interactive [dashboard →](https://app.wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](https://paper-attachments.dropbox.com/s_C0EC7008D045FC80715C08E7386E0BBDA59DC92DEE34C734FEA67BF25E4BA5CC_1578297638486_image.png)
    """)
    return


@app.cell
def _(subprocess):
    # Fetch audio, video and other data files to log
    import subprocess as _sp
    _sp.run(['git', 'clone', '--depth', '1', 'https://github.com/wandb/examples.git'],
            capture_output=True, text=True)

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import wandb

    return np, pd, wandb


@app.cell
def _(mo):
    mo.md("""
    ## Authentication

    Optionally provide your W&B API key below. If left blank, W&B will use
    cached credentials or prompt for login.
    """)
    return


@app.cell
def _(mo):
    api_key_field = mo.ui.text(
        kind="password",
        label="W&B API Key (optional)",
        placeholder="Leave blank to use cached credentials"
    )
    return (api_key_field,)


@app.cell
def _(api_key_field):
    if api_key_field.value:
        import os
        os.environ["WANDB_API_KEY"] = api_key_field.value
    return


@app.cell
def _(mo):
    mo.md("""
    ## Run Demonstrations

    Click the button below to run all W&B logging demonstrations.
    Each section will create a separate W&B run showing different
    logging capabilities.
    """)
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="▶ Run Demonstrations")
    return (run_button,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log metrics
    """)
    return


@app.cell
def _(pd):
    # Get Apple stock price data from
    # https://www.macrotrends.net/stocks/charts/AAPL/apple/stock-price-history
    apple = pd.read_csv("examples/data/apple.csv")
    apple = apple[-1000:]
    return (apple,)


@app.cell
def _(apple, wandb, run_button, mo):
    mo.stop(not run_button.clicked)

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="metrics") as run:
        # Log the metric on each step
        for price in apple['close']:
            run.log({"Stock Price": price})

    mo.toast("✓ Metrics logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log plots
    """)
    return


@app.cell
def _(run_button, mo):
    mo.stop(not run_button.clicked)

    import matplotlib.pyplot as plt
    import wandb

    # Initialize a new run
    with wandb.init(project='visualize-predictions', name='plots') as run:
        fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
        fig, ax = plt.subplots()
        ax.plot(fibonacci)
        ax.set_ylabel('Fibonacci values')
        run.log({'plot': fig})

    mo.toast("✓ Plot logged to W&B")
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log Histograms
    """)
    return


@app.cell
def _(np, run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project='visualize-predictions', name='histograms') as run:
        fibonacci = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
        for i in range(1, 10):
            run.log({'histograms': wandb.Histogram(fibonacci / i)})

    mo.toast("✓ Histograms logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log images
    """)
    return


@app.cell
def _(plt, run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project='visualize-predictions', name='images') as run:
        path_to_img = 'examples/data/cafe.jpg'
        im = plt.imread(path_to_img)
        run.log({'img': [wandb.Image(im, caption='Cafe')]})

    mo.toast("✓ Image logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log videos
    """)
    return


@app.cell
def _(run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="videos") as run:
        path_to_video = "examples/data/openai-gym.mp4"
        run.log({"video": wandb.Video(path_to_video, fps=4, format="gif")})

    mo.toast("✓ Video logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](https://api.wandb.ai/files/lavanyashukla/visualize-predictions/0gv4owya/media/videos/openai-gym_89a16925.mp4)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log audio
    """)
    return


@app.cell
def _(run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="audio_file") as run:
        path_to_audio = "examples/data/piano.wav"
        run.log({"examples": [wandb.Audio(path_to_audio, caption="Piano", sample_rate=32)]})

    mo.toast("✓ Audio file logged to W&B")
    return


@app.cell
def _(np, run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="audio_generated") as run:
        fs = 44100  # sampling frequency, Hz
        length = 3  # length, seconds
        xs = np.linspace(0, length, num=fs * length)
        waveform = np.sin(fs * 2 * np.pi / 40 * xs ** 2)
        run.log({"examples": [wandb.Audio(waveform, caption="Boop", sample_rate=fs)]})

    mo.toast("✓ Generated audio logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log tables
    """)
    return


@app.cell
def _(run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="tables") as run:
        # Create tabular data, method 1
        data = [["I love my phone", "1", "1"], ["My phone sucks", "0", "-1"]]
        run.log({"a_table": wandb.Table(data=data, columns=["Text", "Predicted Label", "True Label"])})

        # Create tabular data, method 2
        table = wandb.Table(columns=["Text", "Predicted Label", "True Label"])
        table.add_data("I love my phone", "1", "1")
        table.add_data("My phone sucks", "0", "-1")
        run.log({"another_table": table})

    mo.toast("✓ Tables logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log HTML
    """)
    return


@app.cell
def _(run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="html") as run:
        # Log HTML from file
        path_to_html = "examples/data/some_html.html"
        run.log({"custom_file": wandb.Html(open(path_to_html))})

        # Log raw HTML strings
        run.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})

    mo.toast("✓ HTML logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log 3D Objects
    """)
    return


@app.cell
def _(run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="3d_objects") as run:
        path_to_obj = "examples/data/wolf.obj"
        run.log({"3d_object": wandb.Object3D(open(path_to_obj))})

    mo.toast("✓ 3D object logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log Point Clouds
    """)
    return


@app.cell
def _(np, run_button, mo):
    mo.stop(not run_button.clicked)

    import wandb

    # Initialize a new run
    with wandb.init(project="visualize-predictions", name="point_clouds") as run:
        # Generate a cloud of points
        points = np.random.uniform(size=(250, 3))

        # Log points and boxes in W&B
        run.log(
            {
                "point_scene": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": points,
                        "boxes": np.array(
                            [
                                {
                                    "corners": [
                                        [0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1],
                                        [1, 0, 0],
                                        [1, 1, 0],
                                        [0, 1, 1],
                                        [1, 0, 1],
                                        [1, 1, 1],
                                    ],
                                    "label": "Box",
                                    "color": [123, 321, 111],
                                },
                                {
                                    "corners": [
                                        [0, 0, 0],
                                        [0, 2, 0],
                                        [0, 0, 2],
                                        [2, 0, 0],
                                        [2, 2, 0],
                                        [0, 2, 2],
                                        [2, 0, 2],
                                        [2, 2, 2],
                                    ],
                                    "label": "Box-2",
                                    "color": [111, 321, 0],
                                },
                            ]
                        ),
                        "vectors": np.array([]),
                    }
                )
            }
        )

    mo.toast("✓ Point cloud logged to W&B")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## More Resources
    Check out some other cool things you can do with Weights & Biases:
    * [Track model performance](https://app.wandb.ai/lavanyashukla/visualize-models/reports/Visualize-Model-Performance--Vmlldzo1NTk2MA)
    * [Visualize sklearn models](https://app.wandb.ai/lavanyashukla/visualize-sklearn/reports/Visualize-Sklearn-Model-Performance--Vmlldzo0ODIzNg)
    * [Visualize model predictions](https://app.wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA/)
    """)
    return


if __name__ == "__main__":
    app.run()
