# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#     "marimo>=0.24.0",
#     "numpy>=2,<2.6",
#     "opencv-python>=4.10,<5",
#     "pillow>=10.4,<13",
#     "pyyaml>=6",
#     "torch>=2.6,<2.12",
#     "torchvision>=0.21,<0.27",
#     "ultralytics>=8.4.149,<8.5",
#     "wandb>=0.18,<0.29",
# ]
#
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

"""Train a current Ultralytics YOLO model while tracking it with Weights & Biases.

Run:

    uvx marimo edit train_and_debug_yolov5_models_with_weights_biases.py --sandbox
"""

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Train and Debug Ultralytics YOLO Models with Weights & Biases",
)

with app.setup:
    import io
    import zipfile
    from pathlib import Path
    from urllib.request import urlopen

    import marimo as mo
    import torch
    import wandb
    import yaml
    from PIL import Image
    from ultralytics import YOLO, settings as yolo_settings
    from ultralytics.utils import ASSETS as yolo_assets

    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Avoid the observed YOLO26 loss-indexing failure on Apple's MPS backend.
    training_device = 0 if torch.cuda.is_available() else "cpu"


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    <style>
    .wandb-config-logo--dark {
      display: none;
    }

    body.dark .wandb-config-logo--light {
      display: none;
    }

    body.dark .wandb-config-logo--dark {
      display: block;
    }
    </style>

    <img class="wandb-config-logo--light" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_blackwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />
    <img class="wandb-config-logo--dark" src="https://raw.githubusercontent.com/wandb/docs/main/icons/Endorsed_primary_goldwhite.svg" width="400" alt="Weights & Biases by CoreWeave" />

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/train-and-debug-yolov5-models-with-weights-biases/train_and_debug_yolov5_models_with_weights_biases.py/server)

    # Train and Debug Ultralytics YOLO Models with Weights & Biases

    In This notebook, you will use a pretrained object detector, fine-tune it to
    recognize chess pieces, and use Weights & Biases to inspect its training metrics,
    predictions, dataset, and model checkpoint.

    The original Colab used YOLOv5. This version uses **YOLO26** through the
    maintained [`ultralytics`](https://docs.ultralytics.com/) package and its
    [Weights & Biases integration](https://docs.ultralytics.com/integrations/weights-biases/).
    The notebook still covers object detection and experiment tracking, using
    the Python API in place of the original command-line scripts.

    Open the notebook in the marimo editor and submit the setup form before
    running the training example. The prediction and data-preparation cells
    download their inputs when executed. Review the settings before running
    training. After changing a setting, rerun the dependent cells.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _notify_submission(_value):
        mo.status.toast(
            "Credentials submitted",
            "Run the next cell to verify them with Weights & Biases.",
        )

    wandb_api_form = (
        mo.md("{api_key}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="Forge API key",
                full_width=True,
            )
        )
        .form(
            submit_button_label="Connect to Weights & Biases",
            bordered=True,
            on_change=_notify_submission,
        )
    )

    wandb_api_form
    return (wandb_api_form,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    After submitting the API key, run the following cell to verify the
    credentials and load your available Weights & Biases entities.
    """)
    return


@app.cell(hide_code=True)
def _():
    from wandb.sdk.lib.service.service_connection import (
        WandbApiFailedError as _WandbApiFailedError,
    )

    mo.stop(wandb_api_form.value is None)

    _submission = wandb_api_form.value
    _api_key = _submission["api_key"].strip()

    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
            verify=True,
            prompt=False,
        )
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    try:
        _api = wandb.Api(timeout=15)
        _viewer = _api.viewer
    except (wandb.errors.Error, _WandbApiFailedError, ValueError):
        mo.stop(
            True,
            mo.callout(
                "Weights & Biases verified the credentials but could not retrieve your account entities.",
                kind="danger",
            ),
        )

    available_entities = sorted({
        _value
        for _value in (
            getattr(_viewer, "username", None),
            getattr(_viewer, "entity", None),
            *_viewer.teams,
        )
        if _value
    })
    mo.stop(
        not available_entities,
        mo.callout(
            "Weights & Biases authenticated successfully, but no personal or team entities were found.",
            kind="danger",
        ),
    )

    mo.callout(
        mo.md("✓ Credentials verified. Specify your team entity below."),
        kind="success",
    )
    return (available_entities,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Set your team and project

    Run the following cell and specify your:

    - Weights & Biases team entity.
    - A name for your [project](https://docs.coreweave.com/products/wandb/track/project-page#projects). The default project is `yolo-wandb-demo`.

    Then select **Use this Weights & Biases target**.

    > If the project doesn't exist, Weights & Biases creates it when the first run starts, provided you have write access.
    """)
    return


@app.cell(hide_code=True)
def _():
    def _validate_wandb_target(value):
        if value is None:
            return None
        if not value["entity"]:
            return "Select a Weights & Biases entity."
        if not value["project"].strip():
            return "Enter a project name."
        return None

    def _notify_wandb_submission(_value):
        mo.status.toast(
            "Weights & Biases target selected",
            "Run the next cell to confirm the destination.",
        )

    wandb_setup_form = (
        mo.md("{entity}\n\n{project}")
        .batch(
            entity=mo.ui.dropdown(
                options=available_entities,
                value=available_entities[0],
                label="Weights & Biases entity",
                searchable=True,
                full_width=True,
            ),
            project=mo.ui.text(
                value="yolo-wandb-demo",
                label="Project",
                full_width=True,
            ),
        )
        .form(
            submit_button_label="Use this Weights & Biases target",
            validate=_validate_wandb_target,
            on_change=_notify_wandb_submission,
            bordered=True,
        )
    )

    wandb_setup_form
    return (wandb_setup_form,)




@app.cell(hide_code=True)
def _():
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _entity = _submission["entity"]
    _project = _submission["project"].strip()

    entity = _entity
    project = _project

    return entity, project




@app.cell
def _():
    run_name = "chess-yolo26n"
    config = {
        "model": "yolo26n.pt",
        "epochs": 3,
        "imgsz": 640,
        "batch": 16,
    }
    yolo_settings.update({"wandb": True})
    return config, run_name


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Detect objects with a pretrained model

    YOLO26 is pretrained on the COCO dataset, which includes objects such as
    people, buses, and stop signs. If your application needs those classes, a
    pretrained model may already be useful without additional training.

    Run the detector on a sample street image. This downloads the small
    `yolo26n.pt` checkpoint. `conf=0.25` filters out predictions whose confidence
    is below 0.25. The image shows the remaining boxes, labels, and scores.
    """)
    return


@app.cell
def _(config):
    pretrained_model = YOLO(config["model"])
    prediction = pretrained_model.predict(
        source=str(yolo_assets / "bus.jpg"),
        imgsz=config["imgsz"],
        conf=0.25,
        device=device,
        save=False,
    )[0]
    # Ultralytics plots use BGR; PIL expects RGB.
    Image.fromarray(prediction.plot()[:, :, ::-1])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Prepare the chess-piece dataset

    Chess pieces are not among the COCO classes. To recognize them, we will
    fine-tune the pretrained detector on
    [chessboard images with labeled bounding boxes](https://public.roboflow.com/object-detection/chess-full).
    Starting from pretrained weights lets us reuse features the model has
    already learned.

    The download contains images, YOLO-format text labels, and `data.yaml`.
    The YAML file tells the trainer where the training and validation images
    are and how class IDs map to names. The helper at the end downloads the
    dataset into `data/chess` and fixes its paths for this runtime.
    """)
    return


@app.cell
def _():
    dataset_root, data_yaml_path = download_chess_dataset()
    print(data_yaml_path.read_text())
    return data_yaml_path, dataset_root


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fine-tune the detector and track the experiment

    Weights & Biases helps connect a model's behavior to the data and settings that produced
    it. In the next cell:

    1. `wandb.init` starts a run and records our configuration.
    2. `wandb.Artifact` and `run.log_artifact` version the dataset used by this run.
    3. `YOLO.train` fine-tunes the model. The built-in Weights & Biases integration logs
       losses, validation metrics, prediction images, and diagnostic plots.
       With `save=True`, it also logs the best checkpoint as a model Artifact.

    The run is closed when training completes or raises an error. Its link
    appears in the cell output while training is underway.

    The default three epochs demonstrate the workflow. A GPU can reduce training
    time for longer runs. Training uses CUDA when available and otherwise CPU,
    including on Macs, to avoid an observed MPS error during loss computation.
    CPU training can take longer. Local training
    outputs are stored under `runs/chess`, outside the versioned dataset.
    """)
    return


@app.cell
def _(config, data_yaml_path, dataset_root, entity, project, run_name):
    with wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="training",
        config=config,
    ) as run:
        dataset_artifact = wandb.Artifact(
            name="chess-pieces",
            type="dataset",
            description="Chess-piece images and YOLO bounding-box labels",
        )
        dataset_artifact.add_dir(str(dataset_root))
        run.log_artifact(dataset_artifact, aliases=["latest"])

        training_model = YOLO(run.config.model)
        training_results = training_model.train(
            data=str(data_yaml_path),
            epochs=run.config.epochs,
            imgsz=run.config.imgsz,
            batch=run.config.batch,
            device=training_device,
            workers=0,
            project=str(Path("runs/chess").resolve()),
            name=run.id,
            plots=True,
            save=True,
        )

    training_results.results_dict
    return




@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Resume interrupted training

    Ultralytics saves `last.pt` in the run's local `weights` directory. To resume
    an **unfinished** training job, use its actual checkpoint path:

    ```python
    model_to_resume = YOLO("runs/chess/<run-id>/weights/last.pt")
    model_to_resume.train(resume=True)
    ```

    This restores the training state stored in that checkpoint. Keep the
    dataset available at the paths recorded by the original job. The best-model
    Artifact is useful for inference or a new fine-tuning run. Resuming an
    interrupted job requires a checkpoint that includes its training state.

    See the [Ultralytics training guide](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings)
    for checkpoint resume behavior. The old YOLOv5 `wandb-artifact://` command
    syntax is not used by this notebook.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Dataset helper

    This helper downloads and unpacks the public dataset once, checks the
    archive paths, and updates its YAML file for the local folder layout.
    """)
    return


@app.function
def download_chess_dataset():
    dataset_root = Path("data/chess").resolve()
    data_yaml_path = dataset_root / "data.yaml"
    if not data_yaml_path.exists():
        dataset_url = "https://public.roboflow.com/ds/1BpjFZe9ST?key=KXD7eDvwTa"
        with urlopen(dataset_url, timeout=120) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
                for member in archive.infolist():
                    member_path = (dataset_root / member.filename).resolve()
                    if not member_path.is_relative_to(dataset_root):
                        raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
                archive.extractall(dataset_root)

    if not data_yaml_path.exists():
        raise FileNotFoundError("The downloaded dataset did not contain data.yaml")

    dataset_config = yaml.safe_load(data_yaml_path.read_text())
    dataset_config.pop("path", None)
    dataset_config.update({"train": "train/images", "val": "valid/images"})
    if (dataset_root / "test/images").exists():
        dataset_config["test"] = "test/images"
    else:
        dataset_config.pop("test", None)
    data_yaml_path.write_text(yaml.safe_dump(dataset_config, sort_keys=False))
    return dataset_root, data_yaml_path


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Verify the object-detection run

    Open the run link. Check **Config** for the model, image size, batch size, and
    epochs. Inspect training losses together with precision, recall, mAP50, and
    mAP50-95. Loss describes optimization; mAP summarizes whether classes and
    boxes match held-out labels.

    Open validation predictions, the confusion matrix, and performance curves to
    find missed pieces, duplicate boxes, or confused classes. In **Artifacts**,
    confirm the `chess-pieces` dataset and best checkpoint. The
    [Ultralytics integration guide](https://docs.wandb.ai/models/integrations/ultralytics)
    describes the logged metrics, media, and model outputs.

    ## Next steps

    Increase epochs or change image size, then rerun with the same dataset version.
    Compare mAP and inference-relevant cost, and check whether prediction images
    show fewer mistakes. Use `last.pt` only to resume an interrupted job; use the
    best-model Artifact for inference or a new tracked fine-tuning run.
    """)
    return


if __name__ == "__main__":
    app.run()
