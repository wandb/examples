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

__generated_with = "0.24.0"
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

    W&B Artifacts supports setting time-to-live policies on each version of an Artifact. The following examples show the use of TTL policy in a common Artifact logging workflow. We'll cover:

    - Setting a TTL policy when creating an Artifact
    - Retroactively setting TTL for a specific Artifact alias
    - Using the W&B API to set a TTL for all versions of an Artifact

    TTL applies to user-generated Artifacts. Linking an Artifact version to W&B Registry deactivates its TTL policy, so use the examples below on versions whose lifecycle you intend to manage directly.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Setup
    Let's do a few things before we get started. Below we will:

    - Connect to W&B
    - Download and sample a dataset

    The notebook dependencies install the required libraries automatically.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Authentication

    Enter your [W&B API key](https://wandb.ai/authorize) and the team entity where you have permission to create runs and Artifacts. You can leave the key blank when this environment already has credentials, such as `WANDB_API_KEY` configured in the marimo Secrets panel. If the entity is blank, W&B uses your default entity.

    Changing these fields does nothing until you click **Connect to W&B**. Connecting authenticates and resolves the target path, but it does not create or modify any W&B object.
    """)
    return


@app.cell(hide_code=True)
def _():
    _api_key_input = mo.ui.text(
        kind="password",
        label="W&B API key (optional)",
        placeholder="Paste a key or use configured credentials",
        full_width=True,
    )
    _entity_input = mo.ui.text(
        label="W&B entity or team (optional)",
        placeholder="Leave blank to use your default entity",
        full_width=True,
    )
    _project_input = mo.ui.text(
        value="artifacts-ttl-demo",
        label="W&B project",
        full_width=True,
    )
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}")
        .batch(
            api_key=_api_key_input,
            entity=_entity_input,
            project=_project_input,
        )
        .form(submit_button_label="Connect to W&B", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(
        wandb_login_form.value is None,
        mo.callout(
            mo.md("Connect to W&B before running a remote Artifact step."),
            kind="info",
        ),
    )

    _api_key = wandb_login_form.value["api_key"].strip()
    _submitted_entity = wandb_login_form.value["entity"].strip()
    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
        )
        _login_error = None
    except wandb.errors.Error as _error:
        _login_ok = False
        _login_error = str(_error)

    mo.stop(
        not _login_ok,
        mo.callout(
            mo.md(
                "W&B authentication did not complete. Check the API key and "
                f"try again.\n\nW&B reported: `{_login_error or 'unknown error'}`"
            ),
            kind="danger",
        ),
    )

    entity = _submitted_entity or wandb.Api().default_entity
    mo.stop(
        not entity,
        mo.callout(
            mo.md(
                "W&B did not return a default entity. Enter a team entity in "
                "the form above and reconnect."
            ),
            kind="danger",
        ),
    )
    project = wandb_login_form.value["project"].strip() or "artifacts-ttl-demo"
    wandb_settings = {"entity": entity, "project": project}
    mo.callout(
        mo.md(f"Connected. Remote steps will target `{entity}/{project}`."),
        kind="success",
    )
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Image Sampling
    For the purposes of the walkthrough, we will sample from the Imagenette dataset and organize them into training and validation directories in our notebook session. The block below:

    - Creates folders for our sampled images if they don't already exist
    - Selects a random sample of images from the Imagenette dataset
    - Organizes the samples into training and validation directories

    *Note: we overwrite the files every time we execute this so we get new Artifact versions.*
    """)
    return


@app.cell(hide_code=True)
def _():
    prepare_imagenette = mo.ui.run_button(label="Download and sample Imagenette")
    mo.vstack(
        [
            mo.md(
                "This downloads and extracts Imagenette locally, then keeps a "
                "random 5% sample. It does not contact W&B."
            ),
            prepare_imagenette,
        ]
    )
    return (prepare_imagenette,)


@app.cell(hide_code=True)
def _(prepare_imagenette):
    mo.stop(
        not prepare_imagenette.value,
        mo.callout(
            mo.md("Click the button above when you are ready to prepare the dataset."),
            kind="info",
        ),
    )
    imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    imagenette_workspace = Path(
        tempfile.mkdtemp(prefix="wandb-ttl-imagenette-")
    )
    return imagenette_url, imagenette_workspace


@app.cell
def _(imagenette_url, imagenette_workspace):
    imagenette_archive = imagenette_workspace / "imagenette.tgz"
    urlretrieve(imagenette_url, imagenette_archive)
    return (imagenette_archive,)


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
    We are going to use Imagenette dataset for this example. [Imagenette](https://github.com/fastai/imagenette) is a subset of 10 easily classified classes from Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute). It was created by Jeremy Howard and is a great dataset to experiment with.
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
    ### Image Preview
    Quick block to view some of the images in the sampled dataset.
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
    ## Setting TTL on New Artifacts
    Below we create two new Artifacts for our real and fake data. Because we have internal retention policies in hypothetical organization we'd like to remove any Artifact that has real data (potentially containing personal data). Below we:

    - Create a W&B Run to track the logging of these raw data Artifacts
    - Set the ttl attribute on the real raw data
    - Log our two Artifacts

    > We will use the train dataset as our real data and the validation dataset as our fake data.
    """)
    return


@app.cell(hide_code=True)
def _(train_source_dir, val_source_dir, wandb_settings):
    log_raw_artifacts_button = mo.ui.run_button(
        label="Log TTL demo Artifact versions"
    )
    mo.vstack(
        [
            mo.md(
                f"This creates a `raw-data` run in "
                f"`{wandb_settings['entity']}/{wandb_settings['project']}`, "
                "logs two distinct `real-raw` versions with a 10-day TTL, "
                "assigns them the `extended` and `compliant` aliases, and "
                "logs `fake-raw` without a custom TTL. The files come from "
                f"`{train_source_dir}` and `{val_source_dir}`."
            ),
            log_raw_artifacts_button,
        ]
    )
    return (log_raw_artifacts_button,)


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


@app.cell(hide_code=True)
def _(
    log_raw_artifacts_button,
    train_source_dir,
    val_source_dir,
    wandb_settings,
):
    mo.stop(
        not log_raw_artifacts_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to create the Artifact versions."),
            kind="info",
        ),
    )
    raw_artifacts_run_url, raw_alias_paths = log_raw_artifacts(
        wandb_settings["entity"],
        wandb_settings["project"],
        train_source_dir,
        val_source_dir,
    )
    mo.callout(
        mo.md(f"Artifacts logged: [open the W&B run]({raw_artifacts_run_url})."),
        kind="success",
    )
    return raw_alias_paths, raw_artifacts_run_url


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


@app.cell(hide_code=True)
def _(raw_alias_paths, raw_artifacts_run_url, wandb_settings):
    update_alias_ttls_button = mo.ui.run_button(
        label="Update TTLs for real-raw:extended and :compliant"
    )
    mo.vstack(
        [
            mo.md(
                f"This reads those two aliases from "
                f"`{wandb_settings['entity']}/{wandb_settings['project']}`. "
                "It saves only TTL values that need to change. The source "
                f"Artifact upload completed in [this run]({raw_artifacts_run_url}). "
                f"The exact inputs are `{raw_alias_paths['extended']}` and "
                f"`{raw_alias_paths['compliant']}`."
            ),
            update_alias_ttls_button,
        ]
    )
    return (update_alias_ttls_button,)


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


@app.cell(hide_code=True)
def _(raw_alias_paths, update_alias_ttls_button, wandb_settings):
    mo.stop(
        not update_alias_ttls_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to update the aliases."),
            kind="info",
        ),
    )
    updated_aliases, alias_update_run_url = update_alias_ttls(
        wandb_settings["entity"],
        wandb_settings["project"],
        raw_alias_paths,
    )
    if updated_aliases:
        _alias_message = (
            f"Updated `{', '.join(updated_aliases)}`. "
            f"[Open the W&B run]({alias_update_run_url})."
        )
    else:
        _alias_message = "Both aliases already had the requested TTL values; no run or write was created."
    mo.callout(mo.md(_alias_message), kind="success")
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


@app.cell(hide_code=True)
def _(raw_artifacts_run_url, wandb_settings):
    clear_fake_raw_ttls_button = mo.ui.run_button(
        label="Clear TTL from every fake-raw version"
    )
    mo.vstack(
        [
            mo.md(
                f"This searches `{wandb_settings['entity']}/{wandb_settings['project']}` "
                "and saves only versions whose TTL is currently set. The "
                f"source upload completed in [this run]({raw_artifacts_run_url})."
            ),
            clear_fake_raw_ttls_button,
        ]
    )
    return (clear_fake_raw_ttls_button,)


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


@app.cell(hide_code=True)
def _(clear_fake_raw_ttls_button, wandb_settings):
    mo.stop(
        not clear_fake_raw_ttls_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to clear the TTL policies."),
            kind="info",
        ),
    )
    fake_raw_versions, cleared_versions, clear_ttls_run_url = clear_fake_raw_ttls(
        wandb_settings["entity"], wandb_settings["project"]
    )
    if cleared_versions:
        _clear_message = (
            f"Cleared TTL from {len(cleared_versions)} of "
            f"{len(fake_raw_versions)} discovered versions. "
            f"[Open the W&B run]({clear_ttls_run_url})."
        )
    else:
        _clear_message = (
            f"Found {len(fake_raw_versions)} `fake-raw` versions, all already "
            "without a custom TTL. No run or write was created."
        )
    mo.callout(mo.md(_clear_message), kind="success")
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


@app.cell(hide_code=True)
def _(raw_artifacts_run_url, wandb_settings):
    preprocess_artifact_button = mo.ui.run_button(
        label="Preprocess real-raw:latest and log real-prepro"
    )
    mo.vstack(
        [
            mo.md(
                f"This downloads `real-raw:latest` from "
                f"`{wandb_settings['entity']}/{wandb_settings['project']}`, "
                "resizes its images, and creates a new downstream Artifact "
                "version. Every click intentionally creates a new version. "
                f"The source upload completed in [this run]({raw_artifacts_run_url})."
            ),
            preprocess_artifact_button,
        ]
    )
    return (preprocess_artifact_button,)


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


@app.cell(hide_code=True)
def _(preprocess_artifact_button, wandb_settings):
    mo.stop(
        not preprocess_artifact_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to create the downstream Artifact."),
            kind="info",
        ),
    )
    preprocessing_run_url = preprocess_and_log_artifact(
        wandb_settings["entity"], wandb_settings["project"]
    )
    mo.callout(
        mo.md(f"Downstream Artifact logged: [open the W&B run]({preprocessing_run_url})."),
        kind="success",
    )
    return (preprocessing_run_url,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Traverse the Artifact Graph and Set TTL
    Let's take a look at the original real dataset and traverse downstream runs and Artifacts to set a TTL policy on anything that originated from the real dataset.
    """)
    return


@app.cell(hide_code=True)
def _(preprocessing_run_url, wandb_settings):
    set_downstream_ttls_button = mo.ui.run_button(
        label="Assign random TTLs to downstream dataset versions"
    )
    mo.vstack(
        [
            mo.md(
                f"This traverses consumers of `real-raw:latest` in "
                f"`{wandb_settings['entity']}/{wandb_settings['project']}` and "
                "saves a new random 1–100 day TTL on each downstream dataset "
                "version. Every click intentionally changes those policies. "
                f"The preprocessing upload completed in [this run]({preprocessing_run_url})."
            ),
            set_downstream_ttls_button,
        ]
    )
    return (set_downstream_ttls_button,)


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


@app.cell(hide_code=True)
def _(set_downstream_ttls_button, wandb_settings):
    mo.stop(
        not set_downstream_ttls_button.value,
        mo.callout(
            mo.md("Click the button above when you are ready to change downstream TTL policies."),
            kind="info",
        ),
    )
    downstream_ttls, downstream_ttl_run_url = set_downstream_ttls(
        wandb_settings["entity"], wandb_settings["project"]
    )
    if downstream_ttls:
        _downstream_message = (
            f"Updated {len(downstream_ttls)} downstream dataset versions. "
            f"[Open the W&B run]({downstream_ttl_run_url})."
        )
    else:
        _downstream_message = "No downstream dataset versions were found, so no run or write was created."
    mo.callout(mo.md(_downstream_message), kind="success")
    return


if __name__ == "__main__":
    app.run()
