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
# ///

"""Train a current Ultralytics YOLO model while tracking it with W&B.

Run:

    uvx marimo edit train_and_debug_yolov5_models_with_weights_biases.py --sandbox
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="Train and Debug Ultralytics YOLO Models with Weights & Biases",
)


@app.cell
def _():
    import io
    import tempfile
    import zipfile
    from pathlib import Path
    from urllib.request import urlopen

    import marimo as mo
    import torch
    import wandb
    import yaml
    from ultralytics import YOLO, settings as yolo_settings
    from ultralytics.utils import ASSETS as yolo_assets

    return (
        Path,
        YOLO,
        io,
        mo,
        tempfile,
        torch,
        urlopen,
        wandb,
        yaml,
        yolo_assets,
        yolo_settings,
        zipfile,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/train-and-debug-yolov5-models-with-weights-biases/train_and_debug_yolov5_models_with_weights_biases.py/server)

    # Train and Debug Ultralytics YOLO Models with Weights & Biases

    This notebook fine-tunes the current **Ultralytics YOLO26** model on a
    chess-piece dataset and tracks the experiment with Weights & Biases.
    You will:

    1. run pretrained object detection,
    2. version a custom dataset with a W&B Artifact,
    3. fine-tune YOLO26 with the built-in W&B callback, and
    4. inspect metrics, diagnostic plots, and the best-model Artifact.

    The notebook uses the maintained [`ultralytics`](https://pypi.org/project/ultralytics/)
    package and its [W&B integration](https://docs.ultralytics.com/integrations/weights-biases/).
    It does not clone or patch the legacy YOLOv5 repository.

    Opening the notebook does not create a W&B run or upload anything. Those
    actions begin only after you submit the training form.
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    if torch.cuda.is_available():
        training_device = 0
        _device_label = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        training_device = "mps"
        _device_label = "Apple Metal Performance Shaders"
    else:
        training_device = "cpu"
        _device_label = "CPU"

    mo.callout(
        mo.md(f"This runtime will use **{_device_label}**."),
        kind="info",
        title="Runtime",
    )
    return (training_device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare the data

    The tutorial uses a small public dataset whose images and labels already
    follow Ultralytics' YOLO format. The download is gated so opening the
    notebook does not write files.
    """)
    return


@app.function
def download_chess_dataset():
    """Download and validate the public chess-piece dataset."""
    workspace_path = Path(tempfile.mkdtemp(prefix="wandb-yolo26-"))
    dataset_url = "https://public.roboflow.com/ds/1BpjFZe9ST?key=KXD7eDvwTa"

    with urlopen(dataset_url, timeout=120) as dataset_response:
        with zipfile.ZipFile(io.BytesIO(dataset_response.read())) as archive:
            for member in archive.infolist():
                member_path = (workspace_path / member.filename).resolve()
                if (
                    workspace_path.resolve() != member_path
                    and workspace_path.resolve() not in member_path.parents
                ):
                    raise ValueError(
                        f"Unsafe path in dataset archive: {member.filename}"
                    )
            archive.extractall(workspace_path)

    data_yaml_path = workspace_path / "data.yaml"
    if not data_yaml_path.exists():
        raise FileNotFoundError("The downloaded dataset did not contain data.yaml")

    dataset_config = yaml.safe_load(data_yaml_path.read_text())
    dataset_config.update(
        {
            "path": str(workspace_path),
            "train": "train/images",
            "val": "valid/images",
        }
    )
    data_yaml_path.write_text(yaml.safe_dump(dataset_config, sort_keys=False))
    return workspace_path, data_yaml_path


@app.cell(hide_code=True)
def _(mo):
    prepare_data_button = mo.ui.run_button(
        label="Download the chess dataset",
        kind="success",
        tooltip="Downloads public files into a temporary workspace",
    )
    prepare_data_button
    return (prepare_data_button,)


@app.cell(hide_code=True)
def _(mo, prepare_data_button):
    mo.stop(
        not prepare_data_button.value,
        mo.callout(
            mo.md("Click the download button when you are ready."),
            kind="info",
        ),
    )

    dataset_root, data_yaml_path = download_chess_dataset()
    mo.callout(
        mo.md(f"The chess dataset is ready at `{dataset_root}`."),
        kind="success",
    )
    return data_yaml_path, dataset_root


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Detect objects with a pretrained model

    YOLO26 is pretrained on the COCO dataset. Before fine-tuning, run it on a
    sample street image to see the base model's predictions. This step downloads
    the small `yolo26n.pt` checkpoint but does not contact W&B.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    run_detection_button = mo.ui.run_button(
        label="Run pretrained YOLO26 detection",
        kind="success",
    )
    run_detection_button
    return (run_detection_button,)


@app.cell
def _(YOLO, mo, run_detection_button, training_device, yolo_assets):
    mo.stop(
        not run_detection_button.value,
        mo.callout(
            mo.md("Click the detection button above to run inference."),
            kind="info",
        ),
    )

    _prediction_model = YOLO("yolo26n.pt")
    _prediction = _prediction_model.predict(
        source=str(yolo_assets / "bus.jpg"),
        imgsz=640,
        conf=0.25,
        device=training_device,
        save=False,
        verbose=False,
    )[0]
    _detection_rows = _prediction.summary()
    _annotated_image = _prediction.plot()[:, :, ::-1]

    mo.vstack(
        [
            mo.image(
                _annotated_image,
                width=600,
                alt_text="YOLO26 detections on a street image",
            ),
            mo.ui.table(_detection_rows, label="Detections"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Connect to Weights & Biases

    Enter your [W&B API key](https://wandb.ai/authorize) and, if needed, your
    team or entity. Leave the key blank to use `WANDB_API_KEY` from molab's
    Secrets panel or credentials already configured in the runtime.

    Submitting this form authenticates the SDK. It does not create a run.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _api_key_input = mo.ui.text(
        value="",
        kind="password",
        label="W&B API key (blank uses runtime credentials)",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        value="",
        label="W&B entity or team (blank uses your default)",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}")
        .batch(api_key=_api_key_input, entity=_entity_input)
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(mo, wandb, wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Submit the authentication form to continue."),
            kind="info",
        ),
    )

    _submitted_login = wandb_login_form.value
    _api_key = _submitted_login["api_key"].strip()
    _requested_entity = _submitted_login["entity"].strip()
    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
        )
        _resolved_entity = _requested_entity or wandb.Api().default_entity
        _login_error = None
    except wandb.errors.Error as _error:
        _login_ok = False
        _resolved_entity = None
        _login_error = str(_error)

    mo.stop(
        not _login_ok or not _resolved_entity,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key and "
                "entity, then submit again.\n\n"
                f"W&B reported: `{_login_error or 'No default entity was found.'}`"
            ),
            kind="danger",
        ),
    )

    wandb_settings = {"entity": _resolved_entity}
    mo.callout(
        mo.md(f"Connected to W&B as entity `{_resolved_entity}`."),
        kind="success",
    )
    return (wandb_settings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fine-tune YOLO26 and track the run

    The current Ultralytics integration automatically logs training and
    validation metrics, diagnostic images, performance curves, system metrics,
    and the best checkpoint as a model Artifact.

    Dataset versioning is intentionally explicit below. The legacy YOLOv5
    `--upload_dataset` flag no longer exists, so this notebook creates a W&B
    dataset Artifact with `wandb.Artifact` and `run.log_artifact` before
    training. This keeps the W&B workflow visible and reusable with other
    training libraries.
    """)
    return


@app.cell(hide_code=True)
def _(dataset_root, mo, training_device, wandb_settings):
    _project_input = mo.ui.text(
        value="yolo-wandb-demo",
        label="W&B project",
        full_width=True,
    )
    _run_name_input = mo.ui.text(
        value="chess-yolo26n",
        label="W&B run name",
        full_width=True,
    )
    _epochs_input = mo.ui.number(
        start=1,
        stop=20,
        value=3,
        step=1,
        label="Training epochs",
    )
    yolo_training_form = (
        mo.md("{project}\n\n{run_name}\n\n{epochs}")
        .batch(
            project=_project_input,
            run_name=_run_name_input,
            epochs=_epochs_input,
        )
        .form(
            submit_button_label="Train YOLO26 and log to W&B",
            bordered=True,
        )
    )
    mo.vstack(
        [
            mo.md(
                f"The dataset is ready at `{dataset_root}`. Training will use "
                f"`{training_device}` and log under entity "
                f"`{wandb_settings['entity']}`."
            ),
            yolo_training_form,
        ]
    )
    return (yolo_training_form,)


@app.cell
def _(
    data_yaml_path,
    dataset_root,
    mo,
    training_device,
    wandb,
    wandb_settings,
    yolo_training_form,
):
    mo.stop(
        yolo_training_form.value is None,
        mo.callout(
            mo.md(
                "Review the settings, then submit the form when you are ready "
                "to create a W&B run and upload the dataset Artifact."
            ),
            kind="info",
        ),
    )

    _submitted_training = yolo_training_form.value
    _project = _submitted_training["project"].strip() or "yolo-wandb-demo"
    _run_name = _submitted_training["run_name"].strip() or "chess-yolo26n"
    _epochs = int(_submitted_training["epochs"])

    if wandb.run is not None:
        wandb.finish()

    _run = wandb.init(
        entity=wandb_settings["entity"],
        project=_project,
        name=_run_name,
        job_type="training",
        config={
            "model": "yolo26n.pt",
            "dataset": "chess-pieces",
            "epochs": _epochs,
            "device": str(training_device),
        },
    )
    training_run_url = _run.url

    _artifact_name = f"{_project}-chess-dataset".replace("/", "-").replace(
        " ", "-"
    )
    _dataset_artifact = wandb.Artifact(
        name=_artifact_name,
        type="dataset",
        description="Chess-piece object-detection dataset in YOLO format",
        metadata={"data_yaml": data_yaml_path.name},
    )
    _dataset_artifact.add_dir(str(dataset_root))
    _run.log_artifact(_dataset_artifact, aliases=["latest"])

    yolo_training_request = {
        "epochs": _epochs,
        "run_name": _run_name,
        "local_project_dir": str(dataset_root / "runs"),
    }
    return training_run_url, yolo_training_request


@app.cell(hide_code=True)
def _(mo, training_run_url):
    mo.callout(
        mo.md(
            f"[Open the live W&B run]({training_run_url}) while YOLO26 trains."
        ),
        kind="info",
    )
    training_run_link_ready = True
    return (training_run_link_ready,)


@app.cell
def _(
    Path,
    YOLO,
    data_yaml_path,
    training_device,
    training_run_link_ready,
    wandb,
    yolo_settings,
    yolo_training_request,
):
    assert training_run_link_ready
    _original_wandb_setting = yolo_settings["wandb"]
    yolo_settings.update({"wandb": True})
    try:
        _training_model = YOLO("yolo26n.pt")
        _training_results = _training_model.train(
            data=str(data_yaml_path),
            epochs=yolo_training_request["epochs"],
            imgsz=640,
            device=training_device,
            project=yolo_training_request["local_project_dir"],
            name=yolo_training_request["run_name"],
            exist_ok=True,
            plots=True,
            save=True,
        )
    finally:
        if wandb.run is not None:
            wandb.finish()
        yolo_settings.update({"wandb": _original_wandb_setting})

    training_output_dir = Path(_training_results.save_dir)
    training_metrics = {
        _name: round(float(_value), 4)
        for _name, _value in _training_results.results_dict.items()
    }
    return training_metrics, training_output_dir


@app.cell(hide_code=True)
def _(mo, training_metrics, training_output_dir, training_run_url):
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    "Training finished and the W&B run is closed. "
                    f"[Inspect the completed run]({training_run_url})."
                ),
                kind="success",
            ),
            mo.ui.table(
                [
                    {"metric": _name, "value": _value}
                    for _name, _value in training_metrics.items()
                ],
                label="Final validation metrics",
            ),
            mo.md(f"Local outputs: `{training_output_dir}`"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify and next steps

    In the W&B run, verify that you can find:

    - training and validation losses across epochs;
    - precision, recall, mAP50, and mAP50-95 charts;
    - training mosaics, validation predictions, and performance curves;
    - the input dataset under **Artifacts** with the `latest` alias; and
    - the best YOLO checkpoint as a model Artifact with the `best` alias.

    Next, increase the epoch count or switch from `yolo26n.pt` to a larger
    YOLO26 checkpoint and compare the runs in W&B. For interrupted training,
    use Ultralytics' current local checkpoint flow (`YOLO("last.pt").train(resume=True)`),
    or download a checkpoint Artifact with `run.use_artifact()` before
    resuming on another machine. The legacy `wandb-artifact://` YOLOv5 resume
    URI is not part of the current Ultralytics API.
    """)
    return


if __name__ == "__main__":
    app.run()
