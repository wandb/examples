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
    # Weights & Biases Artifacts Time-to-live (TTL) Walkthrough

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/wandb-artifacts-time-to-live-ttl-walkthrough/wandb_artifacts_time_to_live_ttl_walkthrough.py/server)

    Schedule when artifacts are deleted from Weights & Biases with a W&B Artifact time-to-live (TTL) policy. When you delete an artifact, Weights & Biases marks that artifact as a soft-delete. In other words, the artifact is marked for deletion but files are not immediately deleted from storage.

    This notebook shows how to:

    - Set a TTL policy when creating an Artifact
    - Retroactively set TTL policy for a specific Artifact alias
    - Using Weights & Biases Python SDK to set a TTL for all versions of an Artifact

    > Linking an Artifact version to Registry deactivates its TTL policy.

    ## Setup
    Let's do a few things before we get started. Below we will:

    - Authenticate
    - Download and sample a dataset

    The notebook dependencies install the required libraries automatically.

    ### Authenticate

    Run the following cell and provide your [Forge API key](https://wandb.ai/authorize) and the name of your team. Then click **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="Weights & Biases API key (optional)",
        placeholder="Paste a key or use configured credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="Weights & Biases entity or team (optional)",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    _project_input = mo.ui.text(
        value="artifacts-ttl-demo",
        label="Weights & Biases project",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}")
        .batch(
            api_key=_api_key_input,
            entity=_entity_input,
            project=_project_input,
        )
        .form(submit_button_label="Connect to Weights & Biases", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Submit the form above to authenticate before creating a sweep."),
            kind="info",
        ),
    )

    _submitted = wandb_login_form.value
    _api_key = _submitted["api_key"].strip()
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key or "
                "molab Secrets configuration and submit again."
            ),
            kind="danger",
        ),
    )

    mo.callout(mo.md("Connected to W&B."), kind="success")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Download images

    For the purposes of the walkthrough, we will sample from the Imagenette dataset and organize them into training and validation directories in our notebook session. The block below:

    - Creates folders for our sampled images if they don't already exist
    - Selects a random sample of images from the Imagenette dataset
    - Organizes the samples into training and validation directories

    *Note: we overwrite the files every time we execute this so we get new Artifact versions.*
    """)
    return


@app.cell
def _():
    imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

    imagenette_workspace = Path(
        tempfile.mkdtemp(prefix="wandb-ttl-imagenette-")
    )

    imagenette_archive = imagenette_workspace / "imagenette.tgz"

    urlretrieve(imagenette_url, imagenette_archive)
    return (imagenette_archive,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    The retrieved photos are saved as a `tar` file. The next cell defines a function that lets us access the tar files.
    """)
    return


@app.function
def untar_file(file_path, dest_path):
    destination = Path(dest_path).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(file_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if not member_path.is_relative_to(destination):
                raise ValueError(f"Unsafe path in archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsupported archive entry: {member.name}")
        tar.extractall(destination, members=members)


@app.cell
def _(imagenette_archive):
    untar_file(imagenette_archive, imagenette_archive.parent)
    dataset_dir = imagenette_archive.parent / "imagenette2-160"
    return (dataset_dir,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We are going to use Imagenette dataset for this example. [Imagenette](https://github.com/fastai/imagenette) is a subset of 10 classified classes from **Imagenet** (`tench`, `English springer`, `cassette player`, `chain saw`, `church`, `French horn`, `garbage truck`, `gas pump`, `golf ball`, and `parachute`).
    """)
    return


@app.cell
def _(dataset_dir):
    # let's keep 5% of the images
    for image in dataset_dir.rglob("*.JPEG"):
        if random.random() > 0.05:
            image.unlink()

    # we get two image folders: train and validation
    train_source_dir = dataset_dir / "train"
    val_source_dir = dataset_dir / "val"
    return train_source_dir, val_source_dir


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Let's view some of the images in the sampled dataset:
    """)
    return


@app.function
def show_sample_images(img_dir, num_images=5):
    images = list(img_dir.rglob("*.JPEG"))[:num_images]
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5), squeeze=False)

    # Iterate over the images and display them
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        axes[0, i].imshow(img)
        axes[0, i].axis("off")  # Turn off axis labels

    plt.tight_layout()
    return fig


@app.cell
def _(train_source_dir):
    sample_figure = show_sample_images(train_source_dir)
    sample_figure
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setting TTL when you create a new artifact
    Below we create two new Artifacts for our real and fake data. Because we have internal retention policies in hypothetical organization we'd like to remove any Artifact that has real data (potentially containing personal data). Below we:

    - Create a Weights & Biases Run to track the logging of these raw data Artifacts
    - Set the ttl attribute on the real raw data
    - Log our two Artifacts

    > We will use the train dataset as our real data and the validation dataset as our fake data.
    """)
    return


@app.function
def log_raw_artifacts(entity, project, train_source_dir, val_source_dir):
    with wandb.init(
        entity=entity,
        project=project,
        job_type="raw-data",
        reinit="create_new",
    ) as run:
        raw_extended_art = wandb.Artifact(
            "real-raw",
            type="dataset",
            description="Raw Imagenette sample approved for extended retention",
        )
        raw_extended_art.add_dir(train_source_dir, name="extended-snapshot")
        raw_extended_art.ttl = timedelta(days=10)
        logged_extended_art = run.log_artifact(
            raw_extended_art,
            aliases=["extended"],
        )
        logged_extended_art.wait()

        raw_compliant_art = wandb.Artifact(
            "real-raw",
            type="dataset",
            description="Raw Imagenette sample approved for indefinite retention",
        )
        raw_compliant_art.add_dir(train_source_dir, name="compliant-snapshot")
        raw_compliant_art.ttl = timedelta(days=10)
        logged_compliant_art = run.log_artifact(
            raw_compliant_art,
            aliases=["compliant", "latest"],
        )
        logged_compliant_art.wait()

        raw_fake_art = wandb.Artifact(
            "fake-raw",
            type="dataset",
            description="Raw sample from val Imagenette",
        )

        raw_fake_art.add_dir(val_source_dir)
        logged_fake_art = run.log_artifact(raw_fake_art)
        logged_fake_art.wait()
        run_url = run.url
    alias_paths = {
        "extended": f"{entity}/{project}/real-raw:extended",
        "compliant": f"{entity}/{project}/real-raw:compliant",
    }
    return run_url, alias_paths


@app.cell
def _(train_source_dir, val_source_dir, wandb_settings):
    raw_artifacts_run_url, raw_alias_paths = log_raw_artifacts(
        wandb_settings["entity"],
        wandb_settings["project"],
        train_source_dir,
        val_source_dir,
    )
    return (raw_alias_paths,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Updating/Retroactively Setting TTL on Artifacts
    In our hypothetical organization we've been given approval to retain a specific version of our data indefinitely. We've also been given approval to extend the retention date of an additional dataset. Below we'll:

    - Extend the TTL of an Artifact tagged with the `extended` alias
    - Remove the TTL of an Artifact tagged with the `compliant` alias
    - Programmatically check the status of these two Artifacts

    The preceding logging step creates two distinct `real-raw` versions and assigns these aliases. Retroactive TTL changes take effect when the code calls `artifact.save()`.
    """)
    return


@app.function
def update_alias_ttls(entity, project, alias_paths):
    extended_path = alias_paths["extended"]
    compliant_path = alias_paths["compliant"]
    api = wandb.Api()
    extended_current = api.artifact(extended_path)
    compliant_current = api.artifact(compliant_path)
    extended_ttl = timedelta(days=365)
    needs_extended_update = extended_current.ttl != extended_ttl
    needs_compliant_update = compliant_current.ttl is not None

    changed = []
    run_url = None
    if needs_extended_update or needs_compliant_update:
        with wandb.init(
            entity=entity,
            project=project,
            job_type="modify-ttl",
            reinit="create_new",
        ) as run:
            extended_art = run.use_artifact(extended_path)
            if needs_extended_update:
                extended_art.ttl = extended_ttl  # Delete in a year
                extended_art.save()
                changed.append("real-raw:extended")

            compliant_art = run.use_artifact(compliant_path)
            if needs_compliant_update:
                compliant_art.ttl = None
                compliant_art.save()
                changed.append("real-raw:compliant")

            print(extended_art.ttl)
            print(compliant_art.ttl)
            run_url = run.url
    return changed, run_url


@app.cell
def _(raw_alias_paths, wandb_settings):
    updated_aliases, alias_update_run_url = update_alias_ttls(
        wandb_settings["entity"],
        wandb_settings["project"],
        raw_alias_paths,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Use W&B Import/Export API to Iterate Artifact Versions and Set TTL
    Let's say we've received approval to retain all of the data within a given Artifact and we'd like to remove all TTL policies for every version of an Artifact. Below we:

    - Use the W&B API to get a list of all Runs in a project
    - Get a list of all versions of a specific Artifact (e.g. `fake-raw`)
    - Iterate over each version and remove any existing TTL policy associated with the version
    """)
    return


@app.function
def clear_fake_raw_ttls(entity, project):
    # Artifact metadata extraction
    api = wandb.Api()

    # Define entity and project
    runs = api.runs(entity + "/" + project)

    version_names = []
    for run in runs:
        for artifact in iter(run.logged_artifacts()):
            if "fake-raw" in artifact.name:
                # Can be edited to just display individual elements
                version_names.append(artifact.qualified_name)

    unique_versions = sorted(set(version_names))
    versions_to_clear = [
        version
        for version in unique_versions
        if api.artifact(version).ttl is not None
    ]
    run_url = None
    if versions_to_clear:
        with wandb.init(
            entity=entity,
            project=project,
            job_type="modify-ttl",
            reinit="create_new",
        ) as run:
            for version in versions_to_clear:
                version_art = run.use_artifact(version)
                version_art.ttl = None
                version_art.save()
                print(version_art.ttl)
            run_url = run.url
    return unique_versions, versions_to_clear, run_url


@app.cell
def _(wandb_settings):
    fake_raw_versions, cleared_versions, clear_ttls_run_url = clear_fake_raw_ttls(
        wandb_settings["entity"], wandb_settings["project"]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    > To apply a TTL policy to all artifacts within a team's projects, team admins can set default TTL policies for their team. The default will be applied to both existing and future artifacts logged to projects as long as no custom policies have been set. To learn more about configuring a team default TTL, visit [this](https://docs.wandb.ai/models/artifacts/ttl#set-default-ttl-policies-for-a-team) section of the W&B documentation.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Traverse an Artifact Graph to Set Downstream TTL
    In this last section, we'll do some preprocessing on our images and log those as downstream Artifacts. Once again we'll use the W&B Import/Export API to set a TTL policy on our downstream images for images that originated from our "real" dataset.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Preprocess and log a new Artifact
    """)
    return


@app.function
def preprocess_image(image_path):
    "Resize the image to 64x64"
    return Image.open(image_path).resize((64, 64))


@app.function
def preprocess_and_log_artifact(entity, project):
    with tempfile.TemporaryDirectory() as data_dir:
        real_prepro_dir = Path(data_dir) / "prepro" / "real"
        real_prepro_dir.mkdir(parents=True, exist_ok=True)

        with wandb.init(
            entity=entity,
            project=project,
            job_type="preprocessing",
            reinit="create_new",
        ) as run:
            real_art = run.use_artifact(f"{entity}/{project}/real-raw:latest")
            real_images = Path(
                real_art.download(root=Path(data_dir) / "real-raw")
            )

            for image_path in real_images.rglob("*.JPEG"):
                print(f"Preprocessing {image_path.name}")
                preprocessed_image = preprocess_image(image_path)
                preprocessed_image.save(real_prepro_dir / image_path.name)

            prepro_real_art = wandb.Artifact(
                "real-prepro",
                type="dataset",
                description="Preprocessed images from Imagenette",
            )

            prepro_real_art.add_dir(real_prepro_dir)
            logged_prepro_art = run.log_artifact(prepro_real_art)
            logged_prepro_art.wait()
            run_url = run.url
    return run_url


@app.cell
def _(wandb_settings):
    preprocessing_run_url = preprocess_and_log_artifact(
        wandb_settings["entity"], wandb_settings["project"]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Traverse the Artifact Graph and Set TTL
    Let's take a look at the original real dataset and traverse downstream runs and Artifacts to set a TTL policy on anything that originated from the real dataset.
    """)
    return


@app.function
def set_downstream_ttls(entity, project):
    api = wandb.Api()

    # For demo purposes we'll just do this on the latest version of the real dataset
    artifact = api.artifact(f"{entity}/{project}/real-raw:latest")
    consumer_runs = artifact.used_by()

    # Same pattern from above to get all downstream versions
    version_names = []
    for run in consumer_runs:
        for artifact in iter(run.logged_artifacts()):
            # filter for datasets only
            if artifact.type == "dataset":
                # Can be edited to just display individual elements
                version_names.append(artifact.qualified_name)

    unique_versions = sorted(set(version_names))
    changed_ttls = {}
    run_url = None
    if unique_versions:
        with wandb.init(
            entity=entity,
            project=project,
            job_type="modify-ttl",
            reinit="create_new",
        ) as run:
            for version in unique_versions:
                version_art = run.use_artifact(version)
                # set ttl to a random integer so we can see changes in the UI after we run this
                ttl_days = random.randint(1, 100)
                version_art.ttl = timedelta(days=ttl_days)
                version_art.save()
                changed_ttls[version] = ttl_days
            run_url = run.url
    return changed_ttls, run_url


@app.cell
def _(wandb_settings):
    downstream_ttls, downstream_ttl_run_url = set_downstream_ttls(
        wandb_settings["entity"], wandb_settings["project"]
    )
    return


if __name__ == "__main__":
    app.run()
