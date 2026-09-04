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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W&B_Media.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
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
    # packages added via marimo's package management: wandb !pip install wandb -qq

    # Fetch audio, video and other data files to log
    #! git clone --depth 1 https://github.com/wandb/examples.git
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/wandb/examples.git'])
    # packages added via marimo's package management: soundfile !pip install soundfile -qq

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
def _(wandb):
    wandb.login()
    return


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
    # Read in dataset
    apple = pd.read_csv("examples/examples/data/apple.csv")
    apple = apple[-1000:]
    return (apple,)


@app.cell
def _(apple, wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="metrics")

    # Log the metric on each step
    for price in apple['close']:
        wandb.log({"Stock Price": price})

    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log plots
    """)
    return


@app.cell
def _(wandb):
    import matplotlib.pyplot as plt
    wandb.init(project='visualize-predictions', name='plots')
    # Initialize a new run
    _fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    fig, ax = plt.subplots()
    # Make the plot
    ax.plot(_fibonacci)
    ax.set_ylabel('some interesting numbers')
    wandb.log({'plot': fig})
    wandb.finish()
    # Log the plot
    fig
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log Histograms
    """)
    return


@app.cell
def _(np, wandb):
    # Initialize a new run
    wandb.init(project='visualize-predictions', name='histograms')
    _fibonacci = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    for i in range(1, 10):
        wandb.log({'histograms': wandb.Histogram(_fibonacci / i)})
    # Log a histogram on each step
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log images
    """)
    return


@app.cell
def _(plt, wandb):
    wandb.init(project='visualize-predictions', name='images')
    path_to_img = 'examples/examples/data/cafe.jpg'
    # Initialize a new run
    im = plt.imread(path_to_img)
    wandb.log({'img': [wandb.Image(im, caption='Cafe')]})
    # Generate an image
    # Log the image
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log videos
    """)
    return


@app.cell
def _(wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="videos")

    # Generate a video
    path_to_video = "examples/examples/data/openai-gym.mp4"

    # Log the video
    wandb.log({"video": wandb.Video(path_to_video, fps=4, format="gif")})

    wandb.finish()
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
def _(wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="audio")

    # Generate audio data
    path_to_audio = "examples/examples/data/piano.wav"

    # Log that audio data
    wandb.log({"examples":
               [wandb.Audio(path_to_audio, caption="Piano", sample_rate=32)]})

    wandb.finish()
    return


@app.cell
def _(np, wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="audio")

    # Generate audio data
    fs = 44100 # sampling frequency, Hz
    length = 3  # length, seconds
    xs = np.linspace(0, length, num=fs * length)
    waveform = np.sin(fs * 2 * np.pi / 40  * xs ** 2)

    # Log audio data
    wandb.log({"examples":
               [wandb.Audio(waveform, caption="Boop", sample_rate=fs)]})

    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log tables
    """)
    return


@app.cell
def _(wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="tables")

    # Create tabular data, method 1
    data = [["I love my phone", "1", "1"],["My phone sucks", "0", "-1"]]
    wandb.log({"a_table": wandb.Table(data=data, columns=["Text", "Predicted Label", "True Label"])})

    # Create tabular data, method 2
    table = wandb.Table(columns=["Text", "Predicted Label", "True Label"])
    table.add_data("I love my phone", "1", "1")
    table.add_data("My phone sucks", "0", "-1")
    wandb.log({"another_table": table})

    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log HTML
    """)
    return


@app.cell
def _(wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="html")

    # Generate HTML data
    path_to_html = "examples/examples/data/some_html.html"

    # Log an HTML file
    wandb.log({"custom_file": wandb.Html(open(path_to_html))})

    # Log raw HTML strings
    wandb.log({"custom_string": wandb.Html('<a href="https://mysite">Link</a>')})

    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log 3D Objects
    """)
    return


@app.cell
def _(wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="3d_objects")

    # Generate 3D object data
    path_to_obj = "examples/examples/data/wolf.obj"

    # Log the 3D object
    wandb.log({"3d_object": wandb.Object3D(open(path_to_obj))})

    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Log Point Clouds
    """)
    return


@app.cell
def _(np, wandb):
    # Initialize a new run
    wandb.init(project="visualize-predictions", name="point_clouds")

    # Generate a cloud of points
    points = np.random.uniform(size=(250, 3))

    # Log points and boxes in W&B
    wandb.log(
            {
                "point_scene": wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": points,
                        "boxes": np.array(
                            [
                                {
                                    "corners": [
                                        [0,0,0],
                                        [0,1,0],
                                        [0,0,1],
                                        [1,0,0],
                                        [1,1,0],
                                        [0,1,1],
                                        [1,0,1],
                                        [1,1,1]
                                    ],
                                    "label": "Box",
                                    "color": [123,321,111],
                                },
                                {
                                    "corners": [
                                        [0,0,0],
                                        [0,2,0],
                                        [0,0,2],
                                        [2,0,0],
                                        [2,2,0],
                                        [0,2,2],
                                        [2,0,2],
                                        [2,2,2]
                                    ],
                                    "label": "Box-2",
                                    "color": [111,321,0],
                                }
                            ]
                        ),
                        "vectors": np.array([])
                    }
                )
            }
        )

    wandb.finish()
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
