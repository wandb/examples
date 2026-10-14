# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24",
#     "matplotlib>=3.8",
#     "pillow>=10.0",
#     "wandb>=0.19.10",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="W&B Artifact TTL Walkthrough")

with app.setup:
    import random
    import shutil
    import tarfile
    import tempfile
    from datetime import timedelta
    from pathlib import Path
    from urllib.request import urlretrieve

    import marimo as mo
    import matplotlib.pyplot as plt
    import wandb
    from PIL import Image


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # W&B Artifacts: Time-to-Live (TTL)

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/wandb-artifacts-time-to-live-ttl-walkthrough/wandb_artifacts_time_to_live_ttl_walkthrough.py/server)

    Use artifact TTL policies to schedule data retention. This lesson covers
    setting a policy before upload, changing a policy on an existing version,
    updating a collection of versions, and following artifact lineage.

    TTL is measured from an artifact version's creation time. Expiration
    soft-deletes the version; storage cleanup happens separately. Linking a
    version to Registry deactivates its TTL.

    ## 1. Authenticate and Choose a Project

    The next cell calls `wandb.login()` and sets the destination for this demo.
    Use a project for tutorial data and an account with permission to edit TTL
    policies. Set `entity` to your team name, or leave it as `None` to use your
    default entity.

    Configure `WANDB_API_KEY` in your environment or the molab Secrets panel
    before running. You can get a key from [W&B](https://wandb.ai/authorize).
    In a terminal-backed session, `wandb.login()` can also prompt for a key.

    Running the code cells downloads Imagenette and uploads sample artifacts.
    """)
    return


@app.cell
def _():
    entity = "your-entity"
    project = "artifacts-ttl-demo"
    authenticated = wandb.login()
    if not authenticated:
        raise RuntimeError("Configure W&B credentials before running the lesson.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Prepare Sample Images

    [Imagenette](https://github.com/fastai/imagenette) contains images from ten
    ImageNet classes. The next cell downloads and extracts it into a temporary
    workspace, then copies 5% of the training and validation images into sample
    folders. Sampling leaves the extracted dataset intact, so rerunning the cell
    does not repeatedly shrink it.

    We call the training sample "real" and the validation sample "fake" to
    illustrate different retention policies. Both contain actual photographs;
    these labels are only part of the example.
    """)
    return


@app.cell
def _():
    workspace = Path(tempfile.mkdtemp(prefix="wandb-ttl-"))
    archive = workspace / "imagenette.tgz"

    urlretrieve(
        "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
        archive,
    )

    # Validate archive entries before extracting downloaded files.
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            destination = (workspace / member.name).resolve()
            if not destination.is_relative_to(workspace.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
        tar.extractall(workspace, members=members)

    train_source_dir = workspace / "sample" / "train"
    val_source_dir = workspace / "sample" / "val"
    rng = random.Random(1)

    for split, sample_dir in [("train", train_source_dir), ("val", val_source_dir)]:
        _images = sorted((workspace / "imagenette2-160" / split).rglob("*.JPEG"))
        selected = rng.sample(_images, max(1, len(_images) // 20))
        for _image in selected:
            target = sample_dir / _image.parent.name / _image.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_image, target)
    return train_source_dir, val_source_dir, workspace


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Preview five images from the training sample before uploading them as an
    artifact. This cell only displays local files.
    """)
    return


@app.cell
def _(train_source_dir):
    _images = sorted(train_source_dir.rglob("*.JPEG"))[:5]
    figure, axes = plt.subplots(1, len(_images), figsize=(15, 4), squeeze=False)
    for axis, _image_path in zip(axes[0], _images):
        with Image.open(_image_path) as _image:
            axis.imshow(_image)
        axis.axis("off")
    figure.tight_layout()
    plt.close(figure)
    figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3. Set TTL When Creating an Artifact

    The next cell starts a run, creates dataset artifacts with
    `wandb.Artifact()`, adds files with `add_dir()`, and sets a ten-day TTL
    before calling `raw_data_run.log_artifact()`.

    We create two versions of `real-raw`, with aliases `extended` and
    `compliant`, so the next example can give them different policies. Different
    directory names inside their manifests ensure these are distinct versions,
    even though they contain the same images. Aliases name versions; they do
    not have separate TTL policies.

    We also log `fake-raw` with a ten-day TTL for the bulk-update example.
    `wait()` ensures each upload finishes before later cells fetch it.
    """)
    return


@app.cell
def _(entity, project, train_source_dir, val_source_dir):
    with wandb.init(entity=entity, project=project, job_type="raw-data") as raw_data_run:
        extended = wandb.Artifact("real-raw", type="dataset")
        extended.add_dir(str(train_source_dir), name="extended-snapshot")
        extended.ttl = timedelta(days=10)
        logged_extended = raw_data_run.log_artifact(extended, aliases=["extended"])
        logged_extended.wait()

        compliant = wandb.Artifact("real-raw", type="dataset")
        compliant.add_dir(str(train_source_dir), name="compliant-snapshot")
        compliant.ttl = timedelta(days=10)
        logged_compliant = raw_data_run.log_artifact(compliant, aliases=["compliant", "latest"])
        logged_compliant.wait()

        fake = wandb.Artifact("fake-raw", type="dataset")
        fake.add_dir(str(val_source_dir))
        fake.ttl = timedelta(days=10)
        logged_fake = raw_data_run.log_artifact(fake)
        logged_fake.wait()

        project_path = f"{raw_data_run.entity}/{raw_data_run.project}"
        raw_alias_paths = {
            "extended": f"{project_path}/real-raw:extended",
            "compliant": f"{project_path}/real-raw:compliant",
        }
    return project_path, raw_alias_paths


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 4. Change TTL on Existing Versions

    Suppose we can now keep one version for a year and another indefinitely.
    The next cell fetches each version by alias with `wandb.Api().artifact()`,
    changes its `ttl`, and calls `save()` to persist the change.

    A `timedelta` sets the retention duration; `None` deactivates TTL. Changing
    the duration does not reset the version's creation date. These metadata
    updates do not require a new run or another file upload.
    """)
    return


@app.cell
def _(raw_alias_paths):
    _api = wandb.Api()
    extended_artifact = _api.artifact(raw_alias_paths["extended"])
    extended_artifact.ttl = timedelta(days=365)
    extended_artifact.save()

    compliant_artifact = _api.artifact(raw_alias_paths["compliant"])
    compliant_artifact.ttl = None
    compliant_artifact.save()

    print(f"extended: {_api.artifact(raw_alias_paths['extended']).ttl}")
    print(f"compliant: {_api.artifact(raw_alias_paths['compliant']).ttl}")
    return (compliant_artifact,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5. Update All Versions in a Collection

    `wandb.Api().artifacts()` lists the versions in an artifact collection.
    The next cell queries `fake-raw` directly, then deactivates each version's
    TTL with the same `ttl = None` and `save()` pattern.

    This applies to every version of `fake-raw` in the selected project,
    including versions from previous executions of the lesson.

    Team admins can also configure [default TTL policies](https://docs.wandb.ai/models/artifacts/ttl#set-default-ttl-policies-for-a-team)
    for artifacts without a custom policy.
    """)
    return


@app.cell
def _(project_path):
    _api = wandb.Api()
    cleared_versions = []
    for _artifact in _api.artifacts(type_name="dataset", name=f"{project_path}/fake-raw"):
        _artifact.ttl = None
        _artifact.save()
        cleared_versions.append(_artifact.qualified_name)
        print(f"{_artifact.name}: {_artifact.ttl}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 6. Record Lineage During Preprocessing

    The next cell consumes the `compliant` version of `real-raw`, resizes its
    images to 64 by 64 pixels, and logs a new `real-prepro` artifact.

    `preprocessing_run.use_artifact()` records the input relationship, and
    `preprocessing_run.log_artifact()` records the output. Together they give W&B a lineage
    graph connecting the source data, preprocessing run, and processed data.
    """)
    return


@app.cell
def _(compliant_artifact, entity, project, workspace):
    with tempfile.TemporaryDirectory(dir=workspace) as _data_dir:
        output_dir = Path(_data_dir) / "processed"
        output_dir.mkdir()
        with wandb.init(entity=entity, project=project, job_type="preprocessing") as preprocessing_run:
            source = preprocessing_run.use_artifact(compliant_artifact.qualified_name)
            source_dir = Path(source.download(root=str(Path(_data_dir) / "raw")))
            for _image_path in source_dir.rglob("*.JPEG"):
                _target = output_dir / _image_path.relative_to(source_dir)
                _target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(_image_path) as _image:
                    _image.resize((64, 64)).save(_target)

            processed = wandb.Artifact("real-prepro", type="dataset")
            processed.add_dir(str(output_dir))
            processed_artifact = preprocessing_run.log_artifact(processed)
            processed_artifact.wait()
            print(f"View preprocessing run: {preprocessing_run.url}")
    return (processed_artifact,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 7. Follow Lineage to Set Downstream TTL

    TTL policies do not automatically propagate to derived artifacts. The next
    cell fetches our source version and uses `used_by()` to find its consuming
    runs. For each run, `logged_artifacts()` lists its outputs. We select dataset
    outputs, then give them a 30-day TTL and save the changes.

    This walks one hop downstream: source artifact, consuming run, output
    artifacts. A longer pipeline would repeat this traversal for each output.
    """)
    return


@app.cell
def _(compliant_artifact, processed_artifact):
    # Referencing the uploaded output orders this cell after preprocessing.
    print(f"Preprocessing complete: {processed_artifact.qualified_name}")
    _api = wandb.Api()
    _source = _api.artifact(compliant_artifact.qualified_name)
    downstream_ttls = {}
    for consumer_run in _source.used_by():
        for _artifact in consumer_run.logged_artifacts():
            if _artifact.type == "dataset" and _artifact.qualified_name not in downstream_ttls:
                _artifact.ttl = timedelta(days=30)
                _artifact.save()
                downstream_ttls[_artifact.qualified_name] = _artifact.ttl
                print(f"{_artifact.qualified_name}: {_artifact.ttl}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Inspect the Policies in W&B

    Open the project link printed earlier and select **Artifacts**. Check the
    versions and their TTL policies: `real-raw:extended` has 365 days,
    `real-raw:compliant` and `fake-raw` have TTL deactivated, and downstream
    datasets have 30 days. The lineage view shows the preprocessing relationship.

    The reusable pattern is to fetch a version, assign `artifact.ttl`, and call
    `artifact.save()`. See the [TTL documentation](https://docs.wandb.ai/models/artifacts/ttl)
    for permissions, team defaults, and expiration behavior.
    """)
    return


if __name__ == "__main__":
    app.run()
