# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "accelerate>=1.15,<2",
#     "datasets==5.0.1",
#     "huggingface-hub==1.32.0",
#     "ipython>=9,<10",
#     "marimo>=0.24.2",
#     "peft==0.21.0",
#     "torch==2.6.0",
#     "transformers==5.17.0",
#     "trl==1.13.0",
#     "wandb==0.30.0",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="torchtune and W&B")


@app.cell
def _():
    import os
    import uuid
    from pathlib import Path

    import marimo as mo
    import torch
    import wandb
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    return (
        AutoModelForCausalLM,
        AutoTokenizer,
        HfApi,
        LoraConfig,
        Path,
        SFTConfig,
        SFTTrainer,
        TaskType,
        load_dataset,
        mo,
        os,
        torch,
        uuid,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/torchtune-and-wandb/torchtune_and_wandb.py/server)

    <img src="https://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    # Fine-tune SmolLM2 with LoRA and Weights & Biases

    This notebook uses the maintained Hugging Face training stack:
    [Transformers](https://huggingface.co/docs/transformers),
    [TRL](https://huggingface.co/docs/trl), and
    [PEFT](https://huggingface.co/docs/peft). You will stream a conversational
    dataset, train a compact LoRA adapter, and inspect the live metrics and saved
    adapter in W&B.

    The default model is the Apache-2.0 licensed
    [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
    whose model weights are roughly 269 MB.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prerequisites

    Storage use is deliberately bounded: the dataset is streamed instead of fully
    cached, the base model is much smaller than the former 7B checkpoint, and PEFT
    saves only the trained LoRA adapter. Opening the notebook does not download a
    model, start training, or create a W&B run.
    """)
    return


@app.cell(hide_code=True)
def _(torch):
    _cuda_available = torch.cuda.is_available()
    _mps_available = (
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    )

    if _cuda_available:
        training_device = "cuda"
        model_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    elif _mps_available:
        training_device = "mps"
        model_dtype = torch.float32
    else:
        training_device = "cpu"
        model_dtype = torch.float32
    return model_dtype, training_device


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Authentication and remote storage

    Enter your [W&B API key](https://wandb.ai/authorize), or leave it blank to use
    `WANDB_API_KEY` from molab Secrets or existing credentials. The entity is the
    team name in `wandb.ai/<entity>/<project>`.

    The default Hugging Face model and dataset are public, so the Hugging Face token
    is optional. Creating the `HfApi` client also makes the Hub available in
    marimo's **Files → Remote Storage** panel. The client is read-only in this
    notebook.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    wandb_login_form = (
        mo.md("{api_key}\n\n{entity}\n\n{project}\n\n{hf_token}")
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
                value="smollm2-lora",
                label="W&B project",
                full_width=True,
            ),
            hf_token=mo.ui.text(
                kind="password",
                label="Hugging Face token (optional)",
                placeholder="Only needed for private or gated repositories",
                full_width=True,
            ),
        )
        .form(submit_button_label="Connect", bordered=True)
    )
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(HfApi, mo, wandb, wandb_login_form):
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
    _project = _submitted_login["project"].strip() or "smollm2-lora"
    _hf_token = _submitted_login["hf_token"].strip()
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

    wandb_settings = {
        "api_key": _api_key,
        "entity": _resolved_entity,
        "project": _project,
    }
    hf_settings = {"token": _hf_token}
    hf = HfApi(token=_hf_token or None)
    mo.callout(
        mo.md(
            f"Connected to W&B. Training will target "
            f"`{_resolved_entity}/{_project}`."
        ),
        kind="success",
    )
    return hf, hf_settings, wandb_settings


@app.cell(hide_code=True)
def _(hf, mo):
    _hf_endpoint = hf.endpoint
    mo.md(f"""
    ## Choose the training job

    The public `hf` client connected to `{_hf_endpoint}` lets you browse model and
    dataset repositories in marimo's Remote Storage panel. PyTorch still needs
    model tensors locally while training, so the notebook minimizes that footprint
    by using a 135M-parameter model. The Capybara dataset is loaded with
    `streaming=True`, which avoids a full dataset download and Arrow cache.

    Submitting the form below is the only action that downloads the model, creates
    a W&B run, or starts training.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    lora_training_form = (
        mo.md(
            "{model_id}\n\n{dataset_id}\n\n{sample_count}\n\n"
            "{max_steps}\n\n{run_name}\n\n{output_dir}"
        )
        .batch(
            model_id=mo.ui.text(
                value="HuggingFaceTB/SmolLM2-135M-Instruct",
                label="Hugging Face model",
                full_width=True,
            ),
            dataset_id=mo.ui.text(
                value="trl-lib/Capybara",
                label="Streaming dataset",
                full_width=True,
            ),
            sample_count=mo.ui.number(
                start=32,
                stop=2000,
                step=32,
                value=128,
                label="Training examples",
            ),
            max_steps=mo.ui.number(
                start=1,
                stop=200,
                step=1,
                value=10,
                label="Maximum training steps",
            ),
            run_name=mo.ui.text(
                value="smollm2-lora-demo",
                label="W&B run name",
                full_width=True,
            ),
            output_dir=mo.ui.text(
                value="/tmp/smollm2-lora-wandb",
                label="Adapter output directory",
                full_width=True,
            ),
        )
        .form(submit_button_label="Fine-tune with W&B", bordered=True)
    )
    lora_training_form
    return (lora_training_form,)


@app.cell
def _(Path, lora_training_form, mo, model_dtype, training_device, uuid):
    mo.stop(
        lora_training_form.value is None,
        mo.callout(
            mo.md(
                "Review the settings, then click **Fine-tune with W&B** to start."
            ),
            kind="info",
        ),
    )

    _values = lora_training_form.value
    _model_id = _values["model_id"].strip()
    _dataset_id = _values["dataset_id"].strip()
    _output_dir = _values["output_dir"].strip()
    mo.stop(
        not _model_id or not _dataset_id or not _output_dir,
        mo.callout(
            mo.md("Provide a model, dataset, and adapter output directory."),
            kind="warn",
        ),
    )

    _batch_size = 2 if training_device == "cuda" else 1
    _max_steps = int(_values["max_steps"])
    _sample_count = int(_values["sample_count"])
    mo.stop(
        _sample_count < _max_steps * _batch_size,
        mo.callout(
            mo.md("Choose at least one streamed example per training batch."),
            kind="warn",
        ),
    )

    lora_training_request = {
        "batch_size": _batch_size,
        "cache_dir": "/tmp/huggingface",
        "dataset_id": _dataset_id,
        "device": training_device,
        "dtype": model_dtype,
        "max_steps": _max_steps,
        "model_id": _model_id,
        "output_dir": str(Path(_output_dir).expanduser()),
        "run_id": uuid.uuid4().hex,
        "run_name": _values["run_name"].strip() or "smollm2-lora-demo",
        "sample_count": _sample_count,
    }
    return (lora_training_request,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configure LoRA and W&B logging

    LoRA freezes the base model and adds small trainable matrices to selected
    attention projections. TRL's `SFTTrainer` accepts this PEFT configuration and
    reports metrics through the standard Transformers W&B integration.

    Only the adapter is saved and uploaded as a W&B Artifact; the 269 MB base model
    is not duplicated in the output directory.
    """)
    return


@app.cell(hide_code=True)
def _(LoraConfig, TaskType, lora_training_request, mo, wandb_settings):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    mo.md(
        f"**Model:** `{lora_training_request['model_id']}`  \n"
        f"**Dataset:** `{lora_training_request['dataset_id']}` (streamed)  \n"
        f"**Steps:** `{lora_training_request['max_steps']}`  \n"
        f"**W&B target:** `{wandb_settings['entity']}/{wandb_settings['project']}`"
    )
    return (lora_config,)


@app.cell(hide_code=True)
def _(Path, lora_training_request, wandb, wandb_settings):
    _request = lora_training_request
    _wandb_dir = Path("/tmp/wandb")
    _hf_cache_dir = Path(_request["cache_dir"])
    _wandb_dir.mkdir(parents=True, exist_ok=True)
    _hf_cache_dir.mkdir(parents=True, exist_ok=True)

    lora_run_environment = {
        "WANDB_ENTITY": wandb_settings["entity"],
        "WANDB_PROJECT": wandb_settings["project"],
        "WANDB_RUN_ID": _request["run_id"],
        "WANDB_NAME": _request["run_name"],
        "WANDB_DIR": str(_wandb_dir),
        "HF_HOME": str(_hf_cache_dir),
    }
    if wandb_settings["api_key"]:
        lora_run_environment["WANDB_API_KEY"] = wandb_settings["api_key"]

    lora_run_url = wandb.Settings(
        entity=wandb_settings["entity"],
        project=wandb_settings["project"],
        run_id=_request["run_id"],
    ).run_url
    return lora_run_environment, lora_run_url


@app.cell(hide_code=True)
def _(lora_run_url, mo):
    mo.callout(
        mo.md(
            f"[Open the live training run in W&B]({lora_run_url}) while the "
            "training cell runs."
        ),
        kind="info",
    )
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    SFTConfig,
    SFTTrainer,
    hf_settings,
    load_dataset,
    lora_config,
    lora_run_environment,
    lora_training_request,
    os,
    torch,
    wandb,
    wandb_settings,
):
    _request = lora_training_request
    _hf_token = hf_settings["token"] or None
    _previous_environment = {
        _key: os.environ.get(_key) for _key in lora_run_environment
    }
    os.environ.update(lora_run_environment)

    try:
        _dataset = load_dataset(
            _request["dataset_id"],
            split="train",
            streaming=True,
            token=_hf_token,
            cache_dir=_request["cache_dir"],
        )
        _dataset = _dataset.shuffle(seed=42, buffer_size=1000).take(
            _request["sample_count"]
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            _request["model_id"],
            token=_hf_token,
            cache_dir=_request["cache_dir"],
        )
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token

        _model = AutoModelForCausalLM.from_pretrained(
            _request["model_id"],
            dtype=_request["dtype"],
            token=_hf_token,
            cache_dir=_request["cache_dir"],
        )
        _model.config.use_cache = False

        _training_args = SFTConfig(
            output_dir=_request["output_dir"],
            max_steps=_request["max_steps"],
            per_device_train_batch_size=_request["batch_size"],
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            logging_steps=1,
            save_strategy="no",
            report_to="wandb",
            run_name=_request["run_name"],
            max_length=256,
            packing=False,
            bf16=(
                _request["device"] == "cuda"
                and _request["dtype"] == torch.bfloat16
            ),
            fp16=(
                _request["device"] == "cuda"
                and _request["dtype"] == torch.float16
            ),
            optim="adamw_torch",
            dataloader_pin_memory=_request["device"] == "cuda",
            use_cpu=_request["device"] == "cpu",
            seed=42,
        )

        with wandb.init(
            entity=wandb_settings["entity"],
            project=wandb_settings["project"],
            id=_request["run_id"],
            name=_request["run_name"],
            config={
                "model_id": _request["model_id"],
                "dataset_id": _request["dataset_id"],
                "sample_count": _request["sample_count"],
                "max_steps": _request["max_steps"],
                "device": _request["device"],
                "lora_r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "lora_dropout": lora_config.lora_dropout,
            },
        ) as _run:
            _trainer = SFTTrainer(
                model=_model,
                args=_training_args,
                train_dataset=_dataset,
                processing_class=_tokenizer,
                peft_config=lora_config,
            )
            _trainable_parameters, _total_parameters = (
                _trainer.model.get_nb_trainable_parameters()
            )
            _result = _trainer.train()
            _trainer.save_model(_request["output_dir"])
            _tokenizer.save_pretrained(_request["output_dir"])

            _artifact = wandb.Artifact(
                name=f"{_run.id}-lora-adapter",
                type="model",
                metadata={
                    "base_model": _request["model_id"],
                    "dataset": _request["dataset_id"],
                    "trainable_parameters": _trainable_parameters,
                    "total_parameters": _total_parameters,
                },
            )
            _artifact.add_dir(_request["output_dir"])
            _run.log_artifact(_artifact)
            _run.summary["trainable_parameters"] = _trainable_parameters
            _run.summary["trainable_percent"] = (
                100 * _trainable_parameters / _total_parameters
            )
            _run.summary["adapter_output_dir"] = _request["output_dir"]

            lora_training_summary = {
                "adapter_artifact": _artifact.name,
                "output_dir": _request["output_dir"],
                "run_url": _run.url,
                "train_loss": _result.metrics.get("train_loss"),
                "trainable_parameters": _trainable_parameters,
                "total_parameters": _total_parameters,
            }
    finally:
        for _key, _previous_value in _previous_environment.items():
            if _previous_value is None:
                os.environ.pop(_key, None)
            else:
                os.environ[_key] = _previous_value
    return (lora_training_summary,)


@app.cell(hide_code=True)
def _(lora_training_summary, mo):
    mo.callout(
        mo.md(
            "Fine-tuning finished. Open the "
            f"[W&B run]({lora_training_summary['run_url']}) to inspect the loss "
            "and system metrics, then open the model Artifact to find the LoRA "
            f"adapter. The local adapter is in "
            f"`{lora_training_summary['output_dir']}`."
        ),
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
