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

"""Train a current Ultralytics YOLO model while tracking it with W&B.

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


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/train-and-debug-yolov5-models-with-weights-biases/train_and_debug_yolov5_models_with_weights_biases.py/server)

    # Train and Debug Ultralytics YOLO Models with Weights & Biases

    In this lesson, you will use a pretrained object detector, fine-tune it to
    recognize chess pieces, and use Weights & Biases to inspect its training metrics,
    predictions, dataset, and model checkpoint.

    The original Colab used YOLOv5. This version uses **YOLO26** through the
    maintained [`ultralytics`](https://docs.ultralytics.com/) package and its
    [Weights & Biases integration](https://docs.ultralytics.com/integrations/weights-biases/).
    The object-detection and experiment-tracking workflow is the same; the
    Python API replaces the original command-line scripts.

    Open the notebook in the marimo **editor**. Cells that authenticate,
    download files, or train start disabled: enable each from its cell menu
    when you reach that step, then run it if it hasn't started. Editing a
    setting marks dependent cells as stale until you run them again.
    """)
    return


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Set up your experiment and log in

    Get your [Forge API key](https://wandb.ai/authorize) and set `WANDB_API_KEY` in
    your runtime's environment or **Secrets** panel. Locally, you can also run
    `wandb login` in a terminal. A hosted runtime needs its own credentials.

    Set `entity` to your Weights & Biases team slug, or leave it as `None` to use your default
    entity. You can find the slug in `wandb.ai/<entity>/<project>`. Choose a
    project to group your experiments and a name for this run.

    `wandb.login()` authenticates the SDK. Enabling the Ultralytics Weights & Biases
    integration here ensures it is ready before we create either YOLO model.
    This cell does not create a training run.
    """)
    return


@app.cell
def _():
    entity = "wandb"
    project = "yolo-wandb-demo"
    run_name = "chess-yolo26n"
    config = {
        "model": "yolo26n.pt",
        "epochs": 3,
        "imgsz": 640,
        "batch": 16,
    }

    wandb.login()
    yolo_settings.update({"wandb": True})
    return config, entity, project, run_name


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Detect objects with a pretrained model

    YOLO26 is pretrained on the COCO dataset, which includes objects such as
    people, buses, and stop signs. If your application needs those classes, a
    pretrained model may already be useful without additional training.

    Run the detector on a sample street image. This downloads the small
    `yolo26n.pt` checkpoint. `conf=0.25` filters out predictions whose confidence
    is below 25%; the image shows the remaining boxes, labels, and scores.
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


@app.cell(hide_code=False)
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


@app.cell(hide_code=False)
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

    Three epochs provide a short demonstration; a GPU is recommended for
    longer runs, but this notebook also supports CPU training. Local training
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
            device=device,
            workers=0,
            project=str(Path("runs/chess").resolve()),
            name=run.id,
            plots=True,
            save=True,
        )

    training_results.results_dict
    return


@app.cell(hide_code=False)
def _():
    mo.md(r"""
    ## Verify and compare your experiments

    Open the Weights & Biases run link and inspect:

    - **Config:** the model, epoch count, image size, and batch size you chose.
    - **Charts:** training losses, precision, recall, mAP50, and mAP50-95.
      Losses describe training progress; mAP measures detection quality on
      held-out images, including whether the predicted boxes match the labels.
    - **Images and plots:** validation predictions, confusion matrix, and
      performance curves. Look for missed pieces or confused classes.
    - **Artifacts:** the `chess-pieces` dataset and the best model checkpoint.
      These record the input data and the model produced by this experiment.

    Increase `epochs` in the configuration, then rerun setup and training.
    Compare the runs in the same Weights & Biases project. Does validation mAP improve, and
    do the prediction images show fewer mistakes?
    """)
    return


@app.cell(hide_code=False)
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
    Artifact is useful for inference or a new fine-tuning run; it should not be
    assumed to contain the optimizer state needed to resume an interrupted job.

    See the [Ultralytics training guide](https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings)
    for checkpoint resume behavior. The old YOLOv5 `wandb-artifact://` command
    syntax is not used by this notebook.
    """)
    return


@app.cell(hide_code=False)
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


if __name__ == "__main__":
    app.run()
