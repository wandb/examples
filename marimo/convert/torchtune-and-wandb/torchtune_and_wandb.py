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
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer


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

    This lesson uses [Transformers](https://huggingface.co/docs/transformers),
    [TRL](https://huggingface.co/docs/trl), and
    [PEFT](https://huggingface.co/docs/peft) to fine-tune a small language model
    with LoRA. You will track the experiment in Weights & Biases, inspect training metrics,
    and save the trained adapter as a model Artifact.

    The public model `HuggingFaceTB/SmolLM2-135M-Instruct` has 135 million
    parameters. We stream a small sample of conversational data and train for
    ten steps to demonstrate the integration. This is a short workflow example,
    not an evaluation of model quality.

    ## Environment Setup

    Open this notebook with its declared Python and package versions:

    ```bash
    uvx marimo edit --sandbox torchtune_and_wandb.py
    ```

    Run this command from the notebook's directory. The sandbox reads the
    dependency metadata at the top of the file. Opening it in an existing
    Python environment does not automatically apply those requirements.
    This lesson uses TRL 1.13; TRL 1.15's fused loss path requires a kernel
    that is unavailable on macOS.

    ## 1. Authenticate and Choose a Project

    The next cell calls `wandb.login()` and sets the experiment's destination.
    Set `entity` to your team name, or use `None` for your default entity.

    Configure `WANDB_API_KEY` in your environment or the molab Secrets panel;
    get a key from [Weights & Biases](https://wandb.ai/authorize). In a terminal-backed
    session, `wandb.login()` can also prompt for a key. The default model and
    dataset are public. If you select a private or gated repository, configure
    `HF_TOKEN` as well.
    """)
    return


@app.cell
def _():
    entity = "your-entity"
    project = "smollm2-lora"
    authenticated = wandb.login()
    if not authenticated:
        raise RuntimeError("Configure Weights & Biases credentials before running the lesson.")
    return authenticated, entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2. Choose the Device and Training Settings

    The next cell selects CUDA, Apple MPS, or CPU and sets a suitable model
    dtype. CUDA uses bfloat16 when supported, otherwise float16; MPS and CPU use
    float32.

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
    ## 3. Stream Conversational Training Data

    The next cell loads `trl-lib/Capybara` with `streaming=True`, shuffles
    with a fixed seed, and takes a small sample. Streaming avoids downloading
    and caching the entire dataset. Examples contain conversations that
    `SFTTrainer` formats with the model's chat template.

    Because this dataset is iterable, we set `max_steps` explicitly in the
    training configuration rather than relying on a known dataset length.
    """)
    return


@app.cell
def _(authenticated, cache_dir, dataset_id, sample_count):
    if not authenticated:
        raise RuntimeError("Log in to Weights & Biases first.")
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
    ## 4. Configure LoRA

    The next cell creates a PEFT configuration for the model's attention
    projections. LoRA freezes the base weights and trains small added matrices.

    `r` controls their rank, `lora_alpha` controls their scaling, and
    `lora_dropout` applies dropout on the adapter path. We pass this
    configuration to `SFTTrainer`, which constructs the trainable adapter.
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
    ## 5. Fine-tune and Track the Experiment

    The next cell loads a fresh model and tokenizer, then starts
    `fine_tuning_run` with `wandb.init()`. The run configuration records the
    dataset, training settings, and LoRA hyperparameters for comparison later.

    In `SFTConfig`, `report_to="wandb"` enables the Transformers Weights & Biases
    integration. `logging_steps=1` reports metrics at every training step.
    The trainer uses the active Weights & Biases run, so its metrics and our model Artifact
    belong to the same experiment.

    After training, `trainer.save_model()` saves the PEFT adapter and
    `tokenizer.save_pretrained()` saves the tokenizer. We add that directory
    to a `wandb.Artifact(type="model")` and upload it with
    `fine_tuning_run.log_artifact()`. The base model weights are not included;
    the Artifact metadata records which base model the adapter needs.

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

    output_root.mkdir(parents=True, exist_ok=True)
    adapter_output_dir = tempfile.mkdtemp(prefix="run-", dir=output_root)
    training_args = SFTConfig(
        output_dir=adapter_output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to="wandb",
        run_name=run_name,
        max_length=max_length,
        packing=False,
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
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
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
    return adapter_output_dir, run_url


@app.cell(hide_code=True)
def _(adapter_output_dir, run_url):
    mo.md(f"""
    ## 6. Inspect Metrics and the Saved Adapter

    [Open your Weights & Biases run]({run_url}) to inspect training loss, learning rate,
    and system metrics. The configuration records the settings you chose, and
    the run summary includes the percentage of trainable parameters.

    In **Artifacts**, open the logged model Artifact to browse the adapter
    weights, tokenizer files, and base-model metadata. The local copy is in
    `{adapter_output_dir}`. To use the adapter later, load it alongside the
    base model named in its metadata.

    Try changing the learning rate or LoRA rank, rerun the training cell, and
    compare the experiments in your Weights & Biases project. The ten-step demo verifies the
    workflow; assessing model quality requires a longer run and held-out
    evaluation data.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
