# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "kagglehub==0.4.0",
#     "marimo>=0.24.0",
#     "pyarrow<21",
#     "torch==2.6.0",
#     "torchao==0.10.0",
#     "torchtune==0.6.1",
#     "wandb>=0.18",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="torchtune and W&B")


@app.cell
def _():
    import os
    import subprocess
    import sys
    import uuid
    from pathlib import Path

    import marimo as mo
    import torch
    import wandb

    return Path, mo, os, subprocess, sys, torch, uuid, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/torchtune-and-wandb/torchtune_and_wandb.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    # Getting Started with torchtune and Weights & Biases

    In this notebook you will learn how to use
    [torchtune](https://meta-pytorch.org/torchtune/stable/) with
    [Weights & Biases](https://wandb.ai) to monitor a Mistral 7B LoRA
    fine-tuning run.

    The notebook uses torchtune 0.6.1, its final stable release. The project is
    no longer actively maintained, so the version is pinned to keep its recipe
    names and configuration schema reproducible.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisites

    Attach a BF16-capable CUDA GPU before continuing; an L40 or A100 is a
    suitable choice for this 7B-parameter model. In molab, use **Configure
    compute**, attach a compatible GPU, then save and restart the notebook.

    You also need access to
    [`mistralai/Mistral-7B-v0.1`](https://huggingface.co/mistralai/Mistral-7B-v0.1)
    and a Hugging Face token that can download it. The model download is large,
    and training can take substantial time and GPU memory. Neither begins just
    by opening this notebook.

    The former clone, `cd`, and editable-install cells are unnecessary here:
    the notebook's script metadata installs the stable torchtune package in the
    same Python environment as this kernel.
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    cuda_available = torch.cuda.is_available()
    bf16_supported = cuda_available and torch.cuda.is_bf16_supported()
    torchtune_runtime_ready = cuda_available and bf16_supported
    _device_message = (
        mo.callout(
            mo.md(
                f"BF16 CUDA training is ready: "
                f"**{torch.cuda.get_device_name(0)}**."
            ),
            kind="success",
        )
        if torchtune_runtime_ready
        else mo.callout(
            mo.md(
                (
                    f"CUDA device **{torch.cuda.get_device_name(0)}** is attached, "
                    "but it does not support the recipe's BF16 dtype. Attach a "
                    "BF16-capable GPU and restart."
                    if cuda_available
                    else "No CUDA GPU is attached. Attach a BF16-capable GPU "
                    "and restart before submitting the download or training forms."
                )
            ),
            kind="warn",
            title="BF16-capable CUDA GPU required",
        )
    )
    _device_message
    return (torchtune_runtime_ready,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Authentication

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to
    use `WANDB_API_KEY` from the molab Secrets panel or credentials already
    stored in this runtime. A fresh molab session does not inherit credentials
    from your local computer.

    The entity is the team name in a W&B project URL:
    `wandb.ai/<entity>/<project>`. Leave it blank to use your default entity.
    Editing these fields is inert until you click **Connect to W&B**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}")
        .batch(
            api_key=mo.ui.text(
                kind="password",
                label="W&B API key (optional)",
                placeholder="Paste a key or use configured credentials",
                full_width=True,
            ),
            entity=mo.ui.text(
                label="W&B entity or team (optional)",
                placeholder="Leave blank to use your default entity",
                full_width=True,
            ),
            project=mo.ui.text(
                value="mistral-lora",
                label="W&B project",
                full_width=True,
            ),
        )
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
    _project = _submitted_login["project"].strip() or "mistral-lora"
    try:
        _login_ok = wandb.login(
            key=_api_key or None,
            relogin=bool(_api_key),
        )
        _resolved_entity = _requested_entity or wandb.Api().default_entity
        _login_error = None
    except (wandb.errors.Error, ValueError) as _error:
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

    # Keep a key entered in the password field available only in kernel state
    # so the later child process can authenticate without exposing it in its
    # command line. Blank input continues to use the normal W&B resolution.
    wandb_settings = {
        "api_key": _api_key,
        "entity": _resolved_entity,
        "project": _project,
    }
    mo.callout(
        mo.md(
            f"Connected to W&B. Training will target "
            f"`{_resolved_entity}/{_project}`."
        ),
        kind="success",
    )
    return (wandb_settings,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download a model

    We will download a model checkpoint from the Hugging Face Hub with
    torchtune's `tune download` command. Enter a token below, or leave it blank
    to use `HF_TOKEN` or credentials already stored in this runtime. A token
    entered here is passed to the child process through its environment, not
    printed or added to the command line.

    The current Mistral recipe expects the original PyTorch `.bin` shards, so
    the download excludes duplicate `safetensors` weights. Clicking the button
    below is the only action that starts this large download.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    model_download_form = (
        mo.md("{hf_token}\n\n{model_dir}")
        .batch(
            hf_token=mo.ui.text(
                kind="password",
                label="Hugging Face token (optional)",
                placeholder="Paste a token or use configured credentials",
                full_width=True,
            ),
            model_dir=mo.ui.text(
                value="/tmp/Mistral-7B-v0.1",
                label="Checkpoint directory",
                full_width=True,
            ),
        )
        .form(submit_button_label="Download Mistral 7B", bordered=True)
    )
    model_download_form
    return (model_download_form,)


@app.cell
def _(
    Path,
    torchtune_runtime_ready,
    mo,
    model_download_form,
    os,
    subprocess,
    sys,
):
    mo.stop(
        model_download_form.value is None,
        mo.callout(
            mo.md(
                "Click **Download Mistral 7B** when you are ready for the "
                "large checkpoint download."
            ),
            kind="info",
        ),
    )
    mo.stop(
        not torchtune_runtime_ready,
        mo.callout(
            mo.md(
                "Attach a BF16-capable CUDA GPU, restart, and submit the form again."
            ),
            kind="warn",
        ),
    )

    _download_values = model_download_form.value
    _hf_token = _download_values["hf_token"].strip()
    _model_dir = Path(_download_values["model_dir"].strip()).expanduser()
    _model_dir.mkdir(parents=True, exist_ok=True)
    _download_environment = os.environ.copy()
    if _hf_token:
        _download_environment["HF_TOKEN"] = _hf_token
    _tune_entrypoint = (
        "from torchtune._cli.tune import main; raise SystemExit(main())"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            _tune_entrypoint,
            "download",
            "mistralai/Mistral-7B-v0.1",
            "--output-dir",
            str(_model_dir),
            "--ignore-patterns",
            "*.safetensors",
        ],
        check=True,
        env=_download_environment,
    )
    model_checkpoint_dir = str(_model_dir)
    mo.callout(
        mo.md(f"Mistral 7B is available at `{model_checkpoint_dir}`."),
        kind="success",
    )
    return (model_checkpoint_dir,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure W&B logging

    torchtune recipes are configured with YAML plus command-line overrides.
    The current logger component is:

    ```yaml
    metric_logger:
      _component_: torchtune.training.metric_logging.WandBLogger
      project: mistral-lora
      entity: your-team
    log_every_n_steps: 1
    ```

    The form below applies those same settings as explicit overrides to the
    packaged `mistral/7B_lora_single_device` recipe. It also limits the demo to
    a configurable number of steps. Submitting it creates exactly one W&B run
    and starts GPU training.
    """)
    return


@app.cell(hide_code=True)
def _(mo, model_checkpoint_dir, wandb_settings):
    _training_note = mo.md(
        f"Checkpoint: `{model_checkpoint_dir}`  \n"
        f"W&B target: `{wandb_settings['entity']}/{wandb_settings['project']}`"
    )
    torchtune_training_form = (
        mo.md("{run_name}\n\n{max_steps}\n\n{log_every}\n\n{output_dir}")
        .batch(
            run_name=mo.ui.text(
                value="mistral-lora-demo",
                label="W&B run name",
                full_width=True,
            ),
            max_steps=mo.ui.number(
                start=1,
                stop=1000,
                step=1,
                value=100,
                label="Maximum training steps",
            ),
            log_every=mo.ui.number(
                start=1,
                stop=100,
                step=1,
                value=1,
                label="Log every N steps",
            ),
            output_dir=mo.ui.text(
                value="/tmp/torchtune/mistral_7B/lora_wandb",
                label="Training output directory",
                full_width=True,
            ),
        )
        .form(submit_button_label="Fine-tune with W&B", bordered=True)
    )
    mo.vstack([_training_note, torchtune_training_form])
    return (torchtune_training_form,)


@app.cell(hide_code=True)
def _(
    torchtune_runtime_ready,
    mo,
    model_checkpoint_dir,
    os,
    torchtune_training_form,
    uuid,
    wandb,
    wandb_settings,
):
    mo.stop(
        torchtune_training_form.value is None,
        mo.callout(
            mo.md(
                "Review the settings, then submit the form when you are ready "
                "to create a W&B run and start fine-tuning."
            ),
            kind="info",
        ),
    )
    mo.stop(
        not torchtune_runtime_ready,
        mo.callout(
            mo.md(
                "Attach a BF16-capable CUDA GPU, restart, and submit the form again."
            ),
            kind="warn",
        ),
    )

    _training_values = torchtune_training_form.value
    _run_id = uuid.uuid4().hex
    _run_name = _training_values["run_name"].strip() or "mistral-lora-demo"
    _output_dir = _training_values["output_dir"].strip()
    mo.stop(
        not _output_dir,
        mo.callout(mo.md("Choose a training output directory."), kind="warn"),
    )

    torchtune_run_environment = os.environ.copy()
    torchtune_run_environment.update(
        {
            "WANDB_ENTITY": wandb_settings["entity"],
            "WANDB_PROJECT": wandb_settings["project"],
            "WANDB_RUN_ID": _run_id,
            "WANDB_NAME": _run_name,
        }
    )
    if wandb_settings["api_key"]:
        torchtune_run_environment["WANDB_API_KEY"] = wandb_settings["api_key"]
    torchtune_run_url = wandb.Settings(
        entity=wandb_settings["entity"],
        project=wandb_settings["project"],
        run_id=_run_id,
    ).run_url
    torchtune_training_request = {
        "checkpoint_dir": model_checkpoint_dir,
        "entity": wandb_settings["entity"],
        "log_every": int(_training_values["log_every"]),
        "max_steps": int(_training_values["max_steps"]),
        "output_dir": _output_dir,
        "project": wandb_settings["project"],
        "run_id": _run_id,
        "run_name": _run_name,
    }
    return (
        torchtune_run_environment,
        torchtune_run_url,
        torchtune_training_request,
    )


@app.cell(hide_code=True)
def _(mo, torchtune_run_url):
    mo.callout(
        mo.md(
            f"[Open the live torchtune run in W&B]({torchtune_run_url}) while "
            "the training command runs."
        ),
        kind="info",
    )
    return


@app.cell
def _(
    subprocess,
    sys,
    torchtune_run_environment,
    torchtune_training_request,
):
    _request = torchtune_training_request
    _tune_entrypoint = (
        "from torchtune._cli.tune import main; raise SystemExit(main())"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            _tune_entrypoint,
            "run",
            "lora_finetune_single_device",
            "--config",
            "mistral/7B_lora_single_device",
            f"checkpointer.checkpoint_dir={_request['checkpoint_dir']}",
            f"tokenizer.path={_request['checkpoint_dir']}/tokenizer.model",
            f"output_dir={_request['output_dir']}",
            f"max_steps_per_epoch={_request['max_steps']}",
            "metric_logger._component_="
            "torchtune.training.metric_logging.WandBLogger",
            f"metric_logger.project={_request['project']}",
            f"metric_logger.entity={_request['entity']}",
            f"metric_logger.name={_request['run_name']}",
            f"metric_logger.id={_request['run_id']}",
            f"log_every_n_steps={_request['log_every']}",
        ],
        check=True,
        env=torchtune_run_environment,
    )
    torchtune_training_complete = True
    return (torchtune_training_complete,)


@app.cell(hide_code=True)
def _(mo, torchtune_run_url, torchtune_training_complete):
    mo.stop(not torchtune_training_complete)
    mo.callout(
        mo.md(
            "Fine-tuning finished. Open the "
            f"[W&B run]({torchtune_run_url}) and inspect the loss, learning "
            "rate, throughput, resolved recipe configuration, and output files."
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
