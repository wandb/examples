# /// script
# dependencies = [
#     "fsspec[http]==2026.7.0",
#     "matplotlib==3.11.1",
#     "numpy==2.5.2",
#     "pandas==3.0.5",
#     "plotly==7.0.0",
#     "soundfile==0.14.0",
#     "wandb==0.29.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(auto_download=["html"])

with app.setup:
    import io

    import fsspec
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import wandb


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-header-logo--dark {
      display: none;
    }

    body.dark .wandb-header-logo--light {
      display: none;
    }

    body.dark .wandb-header-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-header-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases" />
    <img class="wandb-header-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases" />

    Use [Weights & Biases](https://wandb.com) for machine learning experiment tracking, dataset versioning, and project collaboration.

    # Log (Almost) Anything with Weights & Biases Media
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/log-almost-anything-with-w-b-media/log_almost_anything_with_w_b_media.py/server)

    In this notebook, we'll show you how to visualize a model's predictions with Weights & Biases: images, videos, audio, tables, HTML, metrics, plots, 3D objects, and point clouds.

    Follow along with the video below, or [open the tutorial on YouTube](https://wandb.me/media-video). View the finished examples in the interactive [Weights & Biases dashboard](https://app.wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA).

    ![](https://paper-attachments.dropbox.com/s_C0EC7008D045FC80715C08E7386E0BBDA59DC92DEE34C734FEA67BF25E4BA5CC_1578297638486_image.png)

    Running the following cells uploads examples to Weights & Biases. Each section creates a
    separate named run; its context manager finishes the run when the block exits.
    Rerunning a logging cell creates a new run.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `visualize-predictions`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="visualize-predictions", label="Project", full_width=True),
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
    mo.md("✓ Connected to Weights & Biases.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load tutorial assets

    We read the original tutorial's stock prices, image, video, piano recording,
    HTML, and mesh from the Weights & Biases examples repository using `fsspec`.
    """)
    return


@app.cell
def _():
    fs = fsspec.filesystem("github", org="wandb", repo="examples")
    asset_base_url = "examples/data"
    return asset_base_url, fs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log metrics

    Read the last 1,000 Apple stock prices, then log each closing price with
    `metrics_run.log()`. Each call advances Weights & Biases's default step, producing a
    time-series chart. The source is
    [Macrotrends' Apple stock price history](https://www.macrotrends.net/stocks/charts/AAPL/apple/stock-price-history).
    """)
    return


@app.cell
def _(asset_base_url, fs):
    # Apple stock prices from
    # https://www.macrotrends.net/stocks/charts/AAPL/apple/stock-price-history
    with fs.open(f"{asset_base_url}/apple.csv", "rb") as apple_file:
        apple_prices = pd.read_csv(apple_file).tail(1000)
    apple_prices.head()
    return (apple_prices,)


@app.cell
def _(apple_prices, entity, project):
    with wandb.init(name="metrics", project=project, entity=entity) as metrics_run:
        for price in apple_prices["close"]:
            metrics_run.log({"Stock Price": price})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log plots

    Log a Matplotlib figure directly. Weights & Biases converts this line plot to an interactive
    Plotly chart, so you can inspect Fibonacci values in the run dashboard.
    """)
    return


@app.cell
def _(entity, project):
    plot_values = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    fig, ax = plt.subplots()
    ax.plot(plot_values)
    ax.set_ylabel("Fibonacci values")

    with wandb.init(name="plots", project=project, entity=entity) as plots_run:
        plots_run.log({"plot": wandb.Image(fig)})

    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log histograms

    Wrap an array in `wandb.Histogram` to track a distribution. Logging scaled
    Fibonacci values across nine steps shows how the distribution changes.
    """)
    return


@app.cell
def _(entity, project):
    histogram_values = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    with wandb.init(name="histograms", project=project, entity=entity) as histograms_run:
        for i in range(1, 10):
            histograms_run.log({"histograms": wandb.Histogram(histogram_values / i)})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log images

    Wrap image pixels in `wandb.Image` and add a caption. Weights & Biases displays the image
    in a Media panel, making it useful for inspecting inputs or predictions.
    """)
    return


@app.cell
def _(asset_base_url, entity, fs, project):
    with fs.open(f"{asset_base_url}/cafe.jpg", "rb") as image_file:
        im = plt.imread(image_file, format="jpg")

    with wandb.init(name="images", project=project, entity=entity) as images_run:
        images_run.log({"img": [wandb.Image(im, caption="Cafe")]})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log videos

    Read an existing MP4 into a byte buffer and pass it to `wandb.Video`.
    Weights & Biases stores the clip with the run so you can replay behavior such as an agent
    acting in an environment.
    """)
    return


@app.cell
def _(asset_base_url, entity, fs, project):
    with fs.open(f"{asset_base_url}/openai-gym.mp4", "rb") as video_file:
        video = io.BytesIO(video_file.read())

    with wandb.init(name="videos", project=project, entity=entity) as videos_run:
        videos_run.log({"video": wandb.Video(video, format="mp4")})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log audio

    Use `wandb.Audio` for either a recording or a generated waveform. Supply the
    sample rate so playback uses the correct timing, and a caption to identify
    each example. The next two cells log piano audio and a synthetic chirp.
    """)
    return


@app.cell
def _(asset_base_url, entity, fs, project):
    with fs.open(f"{asset_base_url}/piano.wav", "rb") as audio_file:
        samples, piano_sample_rate = sf.read(audio_file, dtype="float32")

    with wandb.init(
        name="audio_file", project=project, entity=entity
    ) as audio_file_run:
        audio_file_run.log(
            {
                "examples": [
                    wandb.Audio(
                        samples,
                        caption="Piano",
                        sample_rate=piano_sample_rate,
                    )
                ]
            }
        )
    return


@app.cell
def _(entity, project):
    generated_sample_rate = 44_100
    duration_seconds = 3
    sample_times = np.linspace(
        0, duration_seconds, num=generated_sample_rate * duration_seconds, endpoint=False
    )
    waveform = np.sin(generated_sample_rate * 2 * np.pi / 40 * sample_times**2)

    with wandb.init(
        name="audio_generated", project=project, entity=entity
    ) as generated_audio_run:
        generated_audio_run.log(
            {
                "examples": [
                    wandb.Audio(
                        waveform,
                        caption="Boop",
                        sample_rate=generated_sample_rate,
                    )
                ]
            }
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log tables

    Tables keep examples, predictions, and labels together for inspection.
    The following cell demonstrates two equivalent ways to build a `wandb.Table`:
    supply all rows at once, or add rows individually with `add_data()`.
    """)
    return


@app.cell
def _(entity, project):
    with wandb.init(name="tables", project=project, entity=entity) as tables_run:
        # Create tabular data, method 1.
        data = [
            ["I love my phone", "1", "1"],
            ["My phone sucks", "0", "-1"],
        ]
        tables_run.log(
            {
                "a_table": wandb.Table(
                    data=data,
                    columns=["Text", "Predicted Label", "True Label"],
                )
            }
        )

        # Create tabular data, method 2.
        table = wandb.Table(
            columns=["Text", "Predicted Label", "True Label"]
        )
        table.add_data("I love my phone", "1", "1")
        table.add_data("My phone sucks", "0", "-1")
        tables_run.log({"another_table": table})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log HTML

    Use `wandb.Html` for custom visualizations or reports. Here we log both a
    downloaded HTML document and an inline string. `data_is_not_path=True` tells
    Weights & Biases to treat the strings as HTML content.
    """)
    return


@app.cell
def _(asset_base_url, entity, fs, project):
    with fs.open(f"{asset_base_url}/some_html.html", "rt") as html_file:
        html = html_file.read()

    with wandb.init(name="html", project=project, entity=entity) as html_run:
        html_run.log(
            {
                "custom_file": wandb.Html(html, data_is_not_path=True),
                "custom_string": wandb.Html(
                    '<a href="https://mysite">Link</a>',
                    data_is_not_path=True,
                ),
            }
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log 3D objects

    Load the tutorial's wolf mesh and wrap its text buffer in `wandb.Object3D`.
    `file_type="obj"` identifies the mesh format for Weights & Biases's 3D viewer.
    """)
    return


@app.cell
def _(asset_base_url, entity, fs, project):
    with fs.open(f"{asset_base_url}/wolf.obj", "rt") as object_file:
        mesh_buffer = io.StringIO(object_file.read())

    with wandb.init(name="3d_objects", project=project, entity=entity) as object_3d_run:
        object_3d_run.log(
            {"3d_object": wandb.Object3D(mesh_buffer, file_type="obj")}
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Log point clouds

    Generate 250 reproducible 3D points and two labeled bounding boxes.
    `wandb.Object3D` also accepts a scene dictionary, allowing you to inspect
    point clouds and annotations together in Weights & Biases.
    """)
    return


@app.cell
def _(entity, project):
    points = np.random.default_rng(42).uniform(size=(250, 3))
    scene = {
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
                    "color": [123, 231, 111],
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
                    "color": [111, 231, 0],
                },
            ]
        ),
        "vectors": np.array([]),
    }

    with wandb.init(
        name="point_clouds", project=project, entity=entity) as point_cloud_run:
        point_cloud_run.log({"point_scene": wandb.Object3D(scene)})
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify and next steps

    Open the run URL printed by each demonstration and verify the result in the Weights & Biases workspace:

    - **Metrics, plots, and histograms:** inspect the run charts.
    - **Images, video, audio, HTML, and 3D data:** inspect the run's Media panels.
    - **Tables:** open the logged Tables panels and compare the two construction methods.

    Try changing the generated waveform or point cloud, then rerun that section's logging cell to create a new run you can compare.

    ## More resources

    - [Track model performance](https://app.wandb.ai/lavanyashukla/visualize-models/reports/Visualize-Model-Performance--Vmlldzo1NTk2MA)
    - [Visualize sklearn models](https://app.wandb.ai/lavanyashukla/visualize-sklearn/reports/Visualize-Sklearn-Model-Performance--Vmlldzo0ODIzNg)
    - [Visualize model predictions](https://app.wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA/)
    """)
    return


if __name__ == "__main__":
    app.run()
