# /// script
# dependencies = ["wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/WandB_Artifacts_Time_to_live_TTL_Walkthrough.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Weights & Biases Artifacts Time-to-live (TTL) Walkthrough
    W&B Artifacts now supports setting time-to-live policies on each version of an Artifact. The feature is currently available in W&B SaaS Cloud and will be released to Enterprise customers using W&B Server in version 0.42.0. The following examples show the use TTL policy in a common Artifact logging workflow. We'll cover:

    - Setting a TTL policy when creating an Artifact
    - Retroactively setting TTL for a specific Artifact aliases
    - Using the W&B API to set a TTL for all versions of an Artifact
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    Let's do a few things before we get started. Below we will:

    - Install the wandb library and download a dataset
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    log to wandb
    """)
    return


@app.cell
def _():
    import wandb
    wandb.login()
    return (wandb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Image Sampling
    For the purposes of the walkthrough, we will sample from the Imagenette dataset and organize them into training and validation directories in our Colab session. The block below:

    - Creates folders for our sampled images if they don't already exist
    - Selects a random sample of images from the Imagenette dataset
    - Organizes the samples into training and validation directories

    *Note: we overwrite the files every time we execute this so we get new Artifact versions.*
    """)
    return


@app.cell
def _(subprocess):
    imagenette_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    #! wget {imagenette_url} -O "imagenette.tgz"
    subprocess.call(['wget', str(imagenette_url), '-O', 'imagenette.tgz'])
    return


@app.cell
def _():
    def untar_file(file_path, dest_path):
        import tarfile
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(dest_path)

    untar_file("imagenette.tgz", "./")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are going to use Imagenette dataset for this example. [Imagenette](https://github.com/fastai/imagenette) is a subset of 10 easily classified classes from Imagenet (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute). It was created by Jeremy Howard and is a great dataset to experiment with.
    """)
    return


@app.cell
def _():
    import random
    from pathlib import Path

    dataset_dir = Path("imagenette2-160")

    # let's keep 5% of the images
    for image in dataset_dir.rglob("*.JPEG"):
        if random.random() > 0.05:
            image.unlink()

    # we get two image folders: train and validation 
    train_source_dir = Path("imagenette2-160/train")
    val_source_dir = Path("imagenette2-160/val")
    return Path, random, train_source_dir, val_source_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Image Preview
    Quick block to view some of the images in the sampled dataset.
    """)
    return


@app.cell
def _():
    import os
    from PIL import Image
    import matplotlib.pyplot as plt

    def show_sample_images(img_dir, num_images=5):
      images = list(img_dir.rglob("*.JPEG"))[:num_images]
      fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

      # Iterate over the images and display them
      for i, img_path in enumerate(images):
          img = Image.open(img_path)
          axes[i].imshow(img)
          axes[i].axis('off')  # Turn off axis labels

      plt.tight_layout()
      plt.show()

    return Image, show_sample_images


@app.cell
def _(show_sample_images, train_source_dir):
    show_sample_images(train_source_dir)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setting TTL on New Artifacts
    Below we create two new Artifacts for our real and fake data. Because we have internal retention policies in hypothetical organization we'd like to remove any Artifact that has real data (potentially containing personal data). Below we:

    - Create a W&B Run to track the logging of these raw data Artifacts
    - Set the ttl attribute on the real raw data
    - Log our two Artifacts

    > We will use the train dataset as our real data and the validation dataset as our fake data.
    """)
    return


@app.cell
def _(train_source_dir, val_source_dir, wandb):
    from datetime import timedelta
    with wandb.init(entity='wandb-smle', project='artifacts-ttl-demo', job_type='raw-data') as _run:
        raw_real_art = wandb.Artifact('real-raw', type='dataset', description='Raw sample train Imagenette')
        raw_real_art.add_dir(train_source_dir)
        raw_real_art.ttl = timedelta(days=10)
        _run.log_artifact(raw_real_art)
        raw_fake_art = wandb.Artifact('fake-raw', type='dataset', description='Raw sample from val Imagenette')
        raw_fake_art.add_dir(val_source_dir)
        _run.log_artifact(raw_fake_art)
        _run.finish()
    return (timedelta,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Updating/Retroactively Setting TTL on Artifacts
    In our hypothetical organization we've been given approval to retain a specific version of our data indefinitely. We've also been given approval to extend the retention date of an additional dataset. Below we'll:

    - Extend the TTL of an Artifact tagged with the `extended` alias
    - Remove the TTL of an Artifact tagged with the `compliant` alias
    - Programmatically check the status of these two Artifacts
    """)
    return


@app.cell
def _(timedelta, wandb):
    with wandb.init(entity='wandb-smle', project='artifacts-ttl-demo', job_type='modify-ttl') as _run:
        extended_art = _run.use_artifact('wandb-smle/artifacts-ttl-demo/real-raw:extended')
        extended_art.ttl = timedelta(days=365)  # Delete in a year
        extended_art.save()
        compliant_art = _run.use_artifact('wandb-smle/artifacts-ttl-demo/real-raw:compliant')
        compliant_art.ttl = None
        compliant_art.save()
        print(extended_art.ttl)
        print(compliant_art.ttl)
        _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use W&B Import/Export API to Iterate Artifact Versions and Set TTL
    Let's say we've received approval to retain all of the data within a given Artifact and we'd like to remove all TTL policies for every version of an Artifact. Below we:

    - Use the W&B API to get a list of all Runs in a project
    - Get a list of all versions of a specific Artifact (e.g. `fake-raw`)
    - Iterate over each  version and remove any existing TTL policy associated with the version
    """)
    return


@app.cell
def _(wandb):
    # Artifact metadata extraction
    _api = wandb.Api()
    entity, project = ('wandb-smle', 'artifacts-ttl-demo')
    # Define entity and project
    runs = _api.runs(entity + '/' + project)
    _version_names = []
    for _run in runs:
        for _artifact in iter(_run.logged_artifacts()):
            if 'fake-raw' in _artifact.name:
                _version_names.append(f'{_artifact.name}/{_artifact.version}')
    with wandb.init(entity='wandb-smle', project='artifacts-ttl-demo', job_type='modify-ttl') as _run:
        for _version in _version_names:
            _version_art = _run.use_artifact(f"wandb-smle/artifacts-ttl-demo/{'/'.join(_version.split('/')[:-1])}")  # Can be edited to just display individual elements
            _version_art.ttl = None
            _version_art.save()
            print(_version_art.ttl)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > To apply a TTL policy to all artifacts within a team's projects, team admins can set default TTL policies for their team. The default will be applied to both existing and future artifacts logged to projects as long as no custom policies have been set. To learn more about configuring a team default TTL, visit [this](https://docs.wandb.ai/guides/artifacts/ttl#set-default-ttl-policies-for-a-team) section of the W&B documentation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Traverse an Artifact Graph to Set Downstream TTL
    In this last section, we'll do some preprocessing on our images and log those as downstream Artifacts. Once again we'll use the W&B Import/Export API to set a TTL policy on our downstream images for images that originated from our "real" dataset.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocess and log a new Artifact
    """)
    return


@app.cell
def _(Image, Path, wandb):
    real_prepro_dir = Path('data/prepro/real')
    real_prepro_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_image(image_path):
        """Resize the image to 64x64"""
        return Image.open(image_path).resize((64, 64))
    with wandb.init(entity='wandb-smle', project='artifacts-ttl-demo', job_type='preprocessing') as _run:
        real_art = _run.use_artifact('wandb-smle/artifacts-ttl-demo/real-raw:latest')
        real_images = Path(real_art.download())
        for image_path in real_images.rglob('*.JPEG'):
            print(f'Preprocessing {image_path.name}')
            preprocessed_image = preprocess_image(image_path)
            preprocessed_image.save(real_prepro_dir / image_path.name)
        prepro_real_art = wandb.Artifact('real-prepro', type='dataset', description='Preprocessed images from CIFAR')
        prepro_real_art.add_dir(real_prepro_dir)
        _run.log_artifact(prepro_real_art)
        _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Traverse the Artifact Graph and Set TTL
    Let's take a look at the original real dataset and traverse downstream runs and Artifacts to set a TTL policy on anything that originated from the real dataset.
    """)
    return


@app.cell
def _(random, timedelta, wandb):
    _api = wandb.Api()
    _artifact = _api.artifact('wandb-smle/artifacts-ttl-demo/real-raw:latest')
    # For demo purposes we'll just do this on the latest version of the real dataset
    consumer_runs = _artifact.used_by()
    _version_names = []
    for _run in consumer_runs:
    # Same pattern from above to get all downstream versions
        for _artifact in iter(_run.logged_artifacts()):
            if _artifact.type == 'dataset':
                _version_names.append(f'{_artifact.name}/{_artifact.version}')
    with wandb.init(entity='wandb-smle', project='artifacts-ttl-demo', job_type='modify-ttl') as _run:  # filter for datasets only
        for _version in _version_names:
            _version_art = _run.use_artifact(f"wandb-smle/artifacts-ttl-demo/{'/'.join(_version.split('/')[:-1])}")  # Can be edited to just display individual elements
            _version_art.ttl = timedelta(days=random.randint(1, 100))
            _version_art.save()
            _run.finish()  # set ttl to a random integer so we can see changes in the UI after we run this
    return


if __name__ == "__main__":
    app.run()
