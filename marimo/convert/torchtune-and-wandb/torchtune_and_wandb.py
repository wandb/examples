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
#     "wandb==0.30.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="LoRA fine-tuning with Weights & Biases",
)

with app.setup:
    import tempfile
    from pathlib import Path

    import marimo as mo
    import torch
    import wandb
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Trainer, TrainingArguments,
    )


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

    # Fine-tune SmolLM2 with LoRA and Weights & Biases

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/torchtune-and-wandb/torchtune_and_wandb.py/server)

    This lesson uses [Transformers](https://huggingface.co/docs/transformers) and
    [PEFT](https://huggingface.co/docs/peft) to fine-tune a small language model
    with LoRA. You will track the experiment in Weights & Biases, inspect training metrics,
    and save the trained adapter as a model Artifact.

    The public model `HuggingFaceTB/SmolLM2-135M-Instruct` has 135 million
    parameters. We stream a small sample of conversational data and train for
    ten steps to demonstrate the integration. This is a short workflow example,
    not an evaluation of model quality.

    The default model and dataset are public. For a private or gated repository, configure `HF_TOKEN` as well.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Authenticate with Weights & Biases

    Paste your [Forge API key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys) into the password field below. Keep your key out of the notebook’s source code.

    If this runtime already has credentials in `WANDB_API_KEY` or your `.netrc` file, leave the API key field blank.

    ## Set your team and project

    Enter your Weights & Biases team name and choose a project for this lesson. The default project is `smollm2-lora`. Then select **Connect to Weights & Biases**.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_setup_form = mo.md("{api_key}\n\n{entity}\n\n{project}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key", full_width=True),
        entity=mo.ui.text(label="Team", placeholder="Your team name", full_width=True),
        project=mo.ui.text(value="smollm2-lora", label="Project", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_setup_form
    return (wandb_setup_form,)


@app.cell(hide_code=True)
def _(wandb_setup_form):
    mo.stop(wandb_setup_form.value is None)

    _submission = wandb_setup_form.value
    _api_key = _submission["api_key"].strip()
    _entity = _submission["entity"].strip()
    _project = _submission["project"].strip()
    mo.stop(
        not _entity or not _project,
        mo.callout("Enter your team and project, then connect again.", kind="warn"),
    )

    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False

    mo.stop(
        not _login_ok,
        mo.callout("Check your API key or saved credentials, then connect again.", kind="danger"),
    )

    entity = _entity
    project = _project
    mo.md("✓ Connected to Weights & Biases.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Choose the Device and Training Settings

    The next cell selects CUDA, Apple MPS, or CPU and sets a suitable model
    dtype. CUDA uses bfloat16 when supported and float16 otherwise. MPS and CPU
    use float32.

    Edit the model, dataset, sample count, and step limit here. A small batch size
    and a 256-token limit keep this demo modest. A GPU is faster, but the default
    model can also train on CPU. Each execution saves to a new folder under
    `output_root` so artifacts contain only that experiment's files.
    """)
    return


@app.cell
def _():
    if torch.cuda.is_available():
        training_device = "cuda"
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        training_device = "mps"
        model_dtype = torch.float32
    else:
        training_device = "cpu"
        model_dtype = torch.float32

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset_id = "trl-lib/Capybara"
    sample_count = 128
    max_steps = 10
    batch_size = 2 if training_device == "cuda" else 1
    learning_rate = 2e-4
    max_length = 256
    run_name = "smollm2-lora-demo"
    cache_dir = "/tmp/huggingface"
    output_root = Path("/tmp/smollm2-lora-wandb")
    print(f"Training device: {training_device}; dtype: {model_dtype}")
    return (
        batch_size,
        cache_dir,
        dataset_id,
        learning_rate,
        max_length,
        max_steps,
        model_dtype,
        model_id,
        output_root,
        run_name,
        sample_count,
        training_device,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Stream Conversational Training Data

    The next cell loads `trl-lib/Capybara` with `streaming=True`, shuffles
    with a fixed seed, and takes a small sample. Streaming avoids downloading
    and caching the entire dataset. Examples contain conversations that
    we format with the model's chat template before training.

    Because this dataset is iterable, we set `max_steps` explicitly in the
    training configuration rather than relying on a known dataset length.
    """)
    return


@app.cell
def _(cache_dir, dataset_id, sample_count):
    train_dataset = load_dataset(
        dataset_id,
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )
    train_dataset = train_dataset.shuffle(seed=42, buffer_size=1000).take(sample_count)
    return (train_dataset,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configure LoRA

    The next cell creates a PEFT configuration for the model's attention
    projections. LoRA freezes the base weights and trains small added matrices.

    `r` controls their rank, `lora_alpha` controls their scaling, and
    `lora_dropout` applies dropout on the adapter path. We pass this
    configuration to `get_peft_model`, which constructs the trainable adapter.
    The parameter counts logged later show how much of the model is trainable.
    """)
    return


@app.cell
def _():
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    return (lora_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fine-tune and Track the Experiment

    The next cell loads a fresh model and tokenizer, then starts
    `fine_tuning_run` with `wandb.init()`. The run configuration records the
    dataset, training settings, and LoRA hyperparameters for comparison later.

    In `TrainingArguments`, `report_to="wandb"` enables the Transformers Weights & Biases
    integration. `logging_steps=1` reports metrics at every training step.
    The trainer uses the active Weights & Biases run, so its metrics and our model Artifact
    belong to the same experiment.

    We use the standard Transformers `Trainer` so training does not require
    TRL's fused Triton loss kernels. Each conversation is tokenized with the
    chat template and truncated to `max_length`. The model learns to predict
    the next token throughout the conversation; padding is excluded from the loss.

    After training, `trainer.save_model()` saves the PEFT adapter and
    `tokenizer.save_pretrained()` saves the tokenizer. We add that directory
    to a `wandb.Artifact(type="model")` and upload it with
    `fine_tuning_run.log_artifact()`. The saved files contain the adapter rather
    than a full copy of the base model. The Artifact metadata identifies the
    base model needed to load the adapter.

    Finally, we add the trainable parameter count and percentage to
    `fine_tuning_run.summary`. The context manager finishes the run once
    training and artifact logging complete.
    """)
    return


@app.cell
def _(
    batch_size,
    cache_dir,
    dataset_id,
    entity,
    learning_rate,
    lora_config,
    max_length,
    max_steps,
    model_dtype,
    model_id,
    output_root,
    project,
    run_name,
    sample_count,
    train_dataset,
    training_device,
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
        cache_dir=cache_dir,
    )
    model.config.use_cache = False
    model = get_peft_model(model, lora_config)

    def _tokenize_conversation(example):
        encoded = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_length,
            return_dict=True,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized_dataset = train_dataset.map(_tokenize_conversation)
    # Pad labels with -100, preserving real EOS tokens even when EOS is also PAD.
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

    output_root.mkdir(parents=True, exist_ok=True)
    adapter_output_dir = tempfile.mkdtemp(prefix="run-", dir=output_root)
    training_args = TrainingArguments(
        output_dir=adapter_output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        bf16=training_device == "cuda" and model_dtype == torch.bfloat16,
        fp16=training_device == "cuda" and model_dtype == torch.float16,
        optim="adamw_torch",
        dataloader_pin_memory=training_device == "cuda",
        use_cpu=training_device == "cpu",
        seed=42,
    )

    with wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        config={
            "model_id": model_id,
            "dataset_id": dataset_id,
            "sample_count": sample_count,
            "max_steps": max_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_length": max_length,
            "device": training_device,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
        },
    ) as fine_tuning_run:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        trainable_parameters, total_parameters = trainer.model.get_nb_trainable_parameters()
        training_result = trainer.train()
        trainer.save_model(adapter_output_dir)
        tokenizer.save_pretrained(adapter_output_dir)

        adapter_artifact = wandb.Artifact(
            name=f"{fine_tuning_run.id}-lora-adapter",
            type="model",
            metadata={
                "base_model": model_id,
                "dataset": dataset_id,
                "trainable_parameters": trainable_parameters,
                "total_parameters": total_parameters,
            },
        )
        adapter_artifact.add_dir(adapter_output_dir)
        fine_tuning_run.log_artifact(adapter_artifact)

        fine_tuning_run.summary["trainable_parameters"] = trainable_parameters
        fine_tuning_run.summary["trainable_percent"] = 100 * trainable_parameters / total_parameters
        fine_tuning_run.summary["adapter_output_dir"] = adapter_output_dir
        run_url = fine_tuning_run.url

    print(f"Training loss: {training_result.metrics.get('train_loss')}")
    print(f"Trainable parameters: {trainable_parameters:,} / {total_parameters:,}")
    print(f"Adapter saved to: {adapter_output_dir}")
    print(f"View the Weights & Biases run: {run_url}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Inspect Metrics and the Saved Adapter

    Open your Weights & Biases run to inspect training loss, learning rate,
    and system metrics. The configuration records the settings you chose, and
    the run summary includes the percentage of trainable parameters.

    In **Artifacts**, open the logged model Artifact to browse the adapter
    weights, tokenizer files, and base-model metadata. To use the adapter later, load it alongside the base model named in its metadata.

    Try changing the learning rate or LoRA rank, rerun the training cell, and
    compare the experiments in your Weights & Biases project. The ten-step demo exercises the logging workflow. Assess model quality
    with a suitable training budget and held-out evaluation data.
    """)
    return


if __name__ == "__main__":
    app.run()
