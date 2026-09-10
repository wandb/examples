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

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])

with app.setup(hide_code=True):
    import io
    import warnings

    import fsspec
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import wandb

    warnings.filterwarnings("ignore", category=UserWarning)


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

    <!--- @wandbcode{media-video} -->

    Use [Weights & Biases](https://wandb.com) for machine learning experiment tracking, dataset versioning, and project collaboration.

    <div><img /></div>

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    <div><img /></div>

    # Log (Almost) Anything with W&B Media
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/log-almost-anything-with-w-b-media/log_almost_anything_with_w_b_media.py/server)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    In this notebook, we'll show you how to visualize a model's predictions with Weights & Biases: images, videos, audio, tables, HTML, metrics, plots, 3D objects, and point clouds.

    Follow along with the video below, or [open the tutorial on YouTube](https://wandb.me/media-video). View the finished examples in the interactive [W&B dashboard](https://app.wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.Html(r"""
    <iframe
      width="100%"
      height="450"
      src="https://www.youtube.com/embed/96MxRvx15Ts"
      title="Log (Almost) Anything with W&B Media"
      frameborder="0"
      allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
      referrerpolicy="strict-origin-when-cross-origin"
      allowfullscreen>
    </iframe>
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ![](https://paper-attachments.dropbox.com/s_C0EC7008D045FC80715C08E7386E0BBDA59DC92DEE34C734FEA67BF25E4BA5CC_1578297638486_image.png)
    """)
    return


@app.cell
def _():
    # Stream tutorial assets: audio, video and other data files to log
    fs = fsspec.filesystem("github", org="wandb", repo="examples")
    asset_base_url = "examples/data"
    return asset_base_url, fs


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, your team or entity. You can leave the key blank when this environment already has W&B credentials.
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
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(api_key=_api_key_input, entity=_entity_input)
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Connect to W&B above before running a logging example."),
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _entity = wandb_login_form.value["entity"].strip()
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
                f"Check the API key and try again. \n\nW&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {
        "project": "visualize-predictions",
        "entity": _entity or None,
    }
    mo.callout(
        mo.md("Connected. Each button below creates one separate W&B run."),
        kind="success",
    )
    return (wandb_settings,)


@app.function
def finish_active_run():
    """Finish a run left open by an interrupted demonstration."""
    if wandb.run is not None:
        wandb.finish()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Run the demonstrations

    Use the button in each section to create only the run you want to inspect. This keeps reactive edits from launching every W&B logging example at once.
    """)
    return


@app.cell(hide_code=True)
def _():
    metrics_button = mo.ui.run_button(label="Log metrics")
    mo.vstack([mo.md("### Log metrics"), metrics_button])
    return (metrics_button,)


@app.cell
def _(asset_base_url, fs):
    # Apple stock prices from
    # https://www.macrotrends.net/stocks/charts/AAPL/apple/stock-price-history
    with fs.open(f"{asset_base_url}/apple.csv", "rb") as _apple_file:
        apple_prices = pd.read_csv(_apple_file).tail(1000)
    apple_prices.head()
    return (apple_prices,)


@app.cell
def _(apple_prices, metrics_button, wandb_settings):
    mo.stop(not metrics_button.value)
    finish_active_run()

    with wandb.init(name="metrics", **wandb_settings) as _run:
        for price in apple_prices["close"]:
            _run.log({"Stock Price": price})
        _run_url = _run.url

    mo.callout(
        mo.md(f"Metrics logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    plots_button = mo.ui.run_button(label="Log a plot")
    mo.vstack([mo.md("### Log plots"), plots_button])
    return (plots_button,)


@app.cell
def _(plots_button, wandb_settings):
    mo.stop(not plots_button.value)
    finish_active_run()

    _fibonacci = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
    fig, ax = plt.subplots()
    ax.plot(_fibonacci)
    ax.set_ylabel("Fibonacci values")

    with wandb.init(name="plots", **wandb_settings) as _run:
        _run.log({"plot": fig})
        _run_url = _run.url

    mo.vstack(
        [
            fig,
            mo.callout(
                mo.md(f"Plot logged. [Open the W&B run]({_run_url})."),
                kind="success",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    histograms_button = mo.ui.run_button(label="Log histograms")
    mo.vstack([mo.md("### Log histograms"), histograms_button])
    return (histograms_button,)


@app.cell
def _(histograms_button, wandb_settings):
    mo.stop(not histograms_button.value)
    finish_active_run()

    _fibonacci = np.array([0, 1, 1, 2, 3, 5, 8, 13, 21, 34])
    with wandb.init(name="histograms", **wandb_settings) as _run:
        for i in range(1, 10):
            _run.log({"histograms": wandb.Histogram(_fibonacci / i)})
        _run_url = _run.url

    mo.callout(
        mo.md(f"Histograms logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    images_button = mo.ui.run_button(label="Log an image")
    mo.vstack([mo.md("### Log images"), images_button])
    return (images_button,)


@app.cell
def _(asset_base_url, fs, images_button, wandb_settings):
    mo.stop(not images_button.value)
    finish_active_run()

    with fs.open(f"{asset_base_url}/cafe.jpg", "rb") as _image_file:
        im = plt.imread(_image_file, format="jpg")

    with wandb.init(name="images", **wandb_settings) as _run:
        _run.log({"img": [wandb.Image(im, caption="Cafe")]})
        _run_url = _run.url

    mo.callout(
        mo.md(f"Image logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    videos_button = mo.ui.run_button(label="Log a video")
    mo.vstack([mo.md("### Log videos"), videos_button])
    return (videos_button,)


@app.cell
def _(asset_base_url, fs, videos_button, wandb_settings):
    mo.stop(not videos_button.value)
    finish_active_run()

    with fs.open(f"{asset_base_url}/openai-gym.mp4", "rb") as _video_file:
        _video = io.BytesIO(_video_file.read())

    with wandb.init(name="videos", **wandb_settings) as _run:
        _run.log({"video": wandb.Video(_video, format="mp4")})
        _run_url = _run.url

    mo.callout(
        mo.md(f"Video logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.Html(r"""
    <video controls preload="metadata" style="width: 100%; max-width: 720px;">
      <source src="https://api.wandb.ai/files/lavanyashukla/visualize-predictions/0gv4owya/media/videos/openai-gym_89a16925.mp4" type="video/mp4" />
      Your browser does not support embedded video.
    </video>
    """)
    return


@app.cell(hide_code=True)
def _():
    audio_file_button = mo.ui.run_button(label="Log the piano recording")
    generated_audio_button = mo.ui.run_button(label="Log generated audio")
    mo.vstack(
        [
            mo.md("### Log audio"),
            mo.hstack([audio_file_button, generated_audio_button], justify="start"),
        ]
    )
    return audio_file_button, generated_audio_button


@app.cell
def _(asset_base_url, audio_file_button, fs, wandb_settings):
    mo.stop(not audio_file_button.value)
    finish_active_run()

    with fs.open(f"{asset_base_url}/piano.wav", "rb") as _audio_file:
        _samples, _sample_rate = sf.read(_audio_file, dtype="float32")

    with wandb.init(name="audio_file", **wandb_settings) as _run:
        _run.log(
            {
                "examples": [
                    wandb.Audio(
                        _samples,
                        caption="Piano",
                        sample_rate=_sample_rate,
                    )
                ]
            }
        )
        _run_url = _run.url

    mo.callout(
        mo.md(f"Audio file logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell
def _(generated_audio_button, wandb_settings):
    mo.stop(not generated_audio_button.value)
    finish_active_run()

    _sample_rate = 44_100
    length = 3
    xs = np.linspace(0, length, num=_sample_rate * length)
    waveform = np.sin(_sample_rate * 2 * np.pi / 40 * xs**2)

    with wandb.init(name="audio_generated", **wandb_settings) as _run:
        _run.log(
            {
                "examples": [
                    wandb.Audio(
                        waveform,
                        caption="Boop",
                        sample_rate=_sample_rate,
                    )
                ]
            }
        )
        _run_url = _run.url

    mo.callout(
        mo.md(f"Generated audio logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    tables_button = mo.ui.run_button(label="Log tables")
    mo.vstack([mo.md("### Log tables"), tables_button])
    return (tables_button,)


@app.cell
def _(tables_button, wandb_settings):
    mo.stop(not tables_button.value)
    finish_active_run()

    with wandb.init(name="tables", **wandb_settings) as _run:
        # Create tabular data, method 1.
        data = [
            ["I love my phone", "1", "1"],
            ["My phone sucks", "0", "-1"],
        ]
        _run.log(
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
        _run.log({"another_table": table})
        _run_url = _run.url

    mo.callout(
        mo.md(f"Tables logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    html_button = mo.ui.run_button(label="Log HTML")
    mo.vstack([mo.md("### Log HTML"), html_button])
    return (html_button,)


@app.cell
def _(asset_base_url, fs, html_button, wandb_settings):
    mo.stop(not html_button.value)
    finish_active_run()

    with fs.open(f"{asset_base_url}/some_html.html", "rt") as _html_file:
        _html = _html_file.read()

    with wandb.init(name="html", **wandb_settings) as _run:
        _run.log(
            {
                "custom_file": wandb.Html(_html, data_is_not_path=True),
                "custom_string": wandb.Html(
                    '<a href="https://mysite">Link</a>',
                    data_is_not_path=True,
                ),
            }
        )
        _run_url = _run.url

    mo.callout(
        mo.md(f"HTML logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    objects_button = mo.ui.run_button(label="Log a 3D object")
    mo.vstack([mo.md("### Log 3D objects"), objects_button])
    return (objects_button,)


@app.cell
def _(asset_base_url, fs, objects_button, wandb_settings):
    mo.stop(not objects_button.value)
    finish_active_run()

    with fs.open(f"{asset_base_url}/wolf.obj", "rt") as _object_file:
        _object = io.StringIO(_object_file.read())

    with wandb.init(name="3d_objects", **wandb_settings) as _run:
        _run.log(
            {"3d_object": wandb.Object3D(_object, file_type="obj")}
        )
        _run_url = _run.url

    mo.callout(
        mo.md(f"3D object logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    point_clouds_button = mo.ui.run_button(label="Log a point cloud")
    mo.vstack([mo.md("### Log point clouds"), point_clouds_button])
    return (point_clouds_button,)


@app.cell
def _(point_clouds_button, wandb_settings):
    mo.stop(not point_clouds_button.value)
    finish_active_run()

    points = np.random.default_rng(42).uniform(size=(250, 3))
    _scene = {
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

    with wandb.init(name="point_clouds", **wandb_settings) as _run:
        _run.log({"point_scene": wandb.Object3D(_scene)})
        _run_url = _run.url

    mo.callout(
        mo.md(f"Point cloud logged. [Open the W&B run]({_run_url})."),
        kind="success",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify and next steps

    Open the run link produced by each demonstration and verify the result in the W&B workspace:

    - **Metrics, plots, and histograms:** inspect the run charts.
    - **Images, video, audio, HTML, and 3D data:** inspect the run's Media panels.
    - **Tables:** open the logged Tables panels and compare the two construction methods.

    Try changing the generated waveform or point cloud, then click that section's button again to create a new run you can compare.

    ## More resources

    - [Track model performance](https://app.wandb.ai/lavanyashukla/visualize-models/reports/Visualize-Model-Performance--Vmlldzo1NTk2MA)
    - [Visualize sklearn models](https://app.wandb.ai/lavanyashukla/visualize-sklearn/reports/Visualize-Sklearn-Model-Performance--Vmlldzo0ODIzNg)
    - [Visualize model predictions](https://app.wandb.ai/lavanyashukla/visualize-predictions/reports/Visualize-Model-Predictions--Vmlldzo1NjM4OA/)
    """)
    return


if __name__ == "__main__":
    app.run()
