# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.24.0",
#     "wandb>=0.18",
#     "torch>=2.6",
#     "transformers>=4.56,<5",
#     "accelerate>=1.0",
#     "datasets>=3.0",
#     "ipython>=8.0",
#     "pandas>=2.0",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", app_title="Visualize LLM training progress")

with app.setup:
    import os
    import tempfile
    from itertools import chain

    # This tutorial uses PyTorch; disable Transformers' optional TensorFlow backend.
    os.environ["USE_TF"] = "0"

    import marimo as mo
    import pandas as pd
    import torch
    import wandb
    from datasets import DatasetDict, load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
    )
    from transformers.integrations import WandbCallback


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

    # Visualize LLM training progress with Weights & Biases Tables

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/custom-progress-callback/custom_progress_callback.py/server)

    In this example we will see how to instrument a custom [callback](https://huggingface.co/docs/transformers/main_classes/callback) for the Hugging Face [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) to periodically visualize model predictions using Weights & Biases [Tables](https://wandb.ai/site/tables)

    > Note: The default GPT-2 model benefits from a GPU. CPU training is supported but slower. We recommend that you start with one epoch and fewer training texts.

    ## Authentication

    If prompted, specify your [Forge API Key](https://docs.coreweave.com/products/wandb/platform/hosting/iam/api-keys). If you run the notebook locally, you can set `WANDB_API_KEY` in your environment or molab Secrets.
    """)
    return


@app.cell(hide_code=True)
def _():
    wandb_login_form = mo.md("{api_key}\n\n{entity}").batch(
        api_key=mo.ui.text(kind="password", label="Forge API key (optional)", full_width=True),
        entity=mo.ui.text(label="Weights & Biases entity or team (optional)", full_width=True),
    ).form(submit_button_label="Connect to Weights & Biases", bordered=True)
    wandb_login_form
    return (wandb_login_form,)


@app.cell(hide_code=True)
def _(wandb_login_form):
    mo.stop(wandb_login_form.value is None, mo.md("Connect to Weights & Biases above before training."))
    _api_key = wandb_login_form.value["api_key"].strip()
    try:
        _login_ok = wandb.login(key=_api_key or None, relogin=bool(_api_key))
    except (wandb.errors.Error, ValueError):
        _login_ok = False
    mo.stop(
        not _login_ok,
        mo.callout("Authentication did not complete. Check your API key or runtime credentials and reconnect.", kind="danger"),
    )
    wandb_settings = {"entity": wandb_login_form.value["entity"].strip() or None}
    mo.callout("Connected to Weights & Biases.", kind="success")
    return (wandb_settings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train a language model

    In this notebook, we'll see how to train a [Transformers](https://github.com/huggingface/transformers) model on a causal language modeling task: predict the next token in a sentence. Labels copy the input tokens; the model shifts them internally to compute next-token loss. Its causal attention mask prevents access to future tokens.

    We will see how to load and preprocess the dataset for the task and train a model on it using the `Trainer` API.
    We will be building a custom trainer callback to visualize the model predictions using Weights & Biases Tables by periodically logging the predictions to the table.
    """)
    return


@app.cell(hide_code=True)
def _(wandb_login_form):
    # Reconnecting resets training consent without hiding the configuration form.
    _connection_submission = wandb_login_form.value
    training_form = mo.md(
        "{project}\n\n{run_name}\n\n{epochs}\n\n{train_texts}\n\n{validation_texts}\n\n{frequency}"
    ).batch(
        project=mo.ui.text(value="huggingface", label="Weights & Biases project"),
        run_name=mo.ui.text(value="gpt2-wikitext2-prediction-progress", label="Run name"),
        epochs=mo.ui.number(start=1, stop=10, value=3, step=1, label="Epochs"),
        train_texts=mo.ui.number(start=0, stop=36718, value=2000, step=100, label="Training texts (0 = all)"),
        validation_texts=mo.ui.number(start=0, stop=3760, value=500, step=100, label="Validation texts (0 = all)"),
        frequency=mo.ui.number(start=1, stop=10, value=1, step=1, label="Log predictions every N epochs"),
    ).form(submit_button_label="Train GPT-2 and log predictions", bordered=True)
    mo.vstack([
        mo.md("Submitting downloads WikiText-2 and GPT-2, trains the model, and creates one Weights & Biases run with metrics, prediction Tables, and model checkpoints."),
        training_form,
    ])
    return (training_form,)


@app.cell(hide_code=True)
def _(BLOCK_SIZE, MODEL_NAME, training_form, wandb_settings):
    mo.stop(training_form.value is None, mo.md("Submit the training form to prepare the dataset and train."))
    mo.stop(not training_form.value["project"].strip(), mo.callout("Enter a Weights & Biases project name.", kind="warn"))
    mo.stop(training_form.value["frequency"] > training_form.value["epochs"], mo.callout("Set the prediction frequency no higher than the number of epochs.", kind="warn"))
    training_config = dict(training_form.value)
    training_config["project"] = training_config["project"].strip()
    training_config["run_name"] = training_config["run_name"].strip() or None
    training_config["entity"] = wandb_settings["entity"]
    training_config["model"] = MODEL_NAME
    training_config["block_size"] = BLOCK_SIZE
    return (training_config,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Prepare the dataset

    We use the [WikiText-2 dataset](https://huggingface.co/datasets/Salesforce/wikitext)
    for this task. Text limits make it practical to explore the callback first.
    """)
    return


@app.cell
def _(training_config):
    datasets = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    datasets = DatasetDict({
        split: datasets[split].select(range(min(limit or len(datasets[split]), len(datasets[split]))))
        for split, limit in [
            ("train", int(training_config["train_texts"])),
            ("validation", int(training_config["validation_texts"])),
        ]
    })
    # look at a sample from the train dataset
    datasets["train"][0]
    return (datasets,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    For causal language modeling (CLM) task we are going to take all the texts in our dataset and concatenate them after they are tokenized.
    Then we split them in examples of a certain sequence length. This way the model will receive chunks of contiguous text that may look like:
    ```
    part of text 1
    ```
    or
    ```
    end of text 1 [BOS_TOKEN] beginning of text 2
    ```
    depending on whether they span over several of the original texts in the dataset or not. We copy the inputs to the labels; the model shifts them internally when computing next-token loss.

    We will use the [`gpt2`](https://huggingface.co/gpt2) architecture for this example. You can pick any of the checkpoints listed [here](https://huggingface.co/models?filter=causal-lm) instead.
    """)
    return


@app.cell
def _():
    MODEL_NAME = "gpt2"
    return (MODEL_NAME,)


@app.cell
def _(training_config):
    tokenizer = AutoTokenizer.from_pretrained(training_config["model"])
    tokenizer.pad_token = tokenizer.eos_token
    return (tokenizer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can now call the tokenizer on all our texts using the [`map`](https://huggingface.co/docs/datasets/process#map) method from the Datasets library.
    """)
    return


@app.cell
def _(datasets, tokenizer):
    tokenized_datasets = datasets.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=["text"],
    )
    return (tokenized_datasets,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Next, we need to concatenate all our texts together then split the result in small chunks of a certain `block_size`. Here we will use a block_size of `128`.
    """)
    return


@app.cell
def _():
    BLOCK_SIZE = 128
    return (BLOCK_SIZE,)


@app.cell
def _(BLOCK_SIZE, tokenized_datasets, tokenizer):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        fn_kwargs={"block_size": BLOCK_SIZE},
        batched=True,
        batch_size=1000,
    )
    if not len(lm_datasets["train"]) or not len(lm_datasets["validation"]):
        raise ValueError(f"Too few tokens for a {BLOCK_SIZE}-token block. Increase the text limits and resubmit.")
    # look at a sample from the preprocessed dataset
    tokenizer.decode(lm_datasets["train"][0]["input_ids"])
    return (lm_datasets,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Model & TrainingArguments

    Load the pretrained GPT-2 model, evaluate once per epoch, and use the submitted
    epoch count. `fp16` is enabled only for CUDA. The training function below sets
    `report_to="wandb"` explicitly and gives the Weights & Biases run a descriptive name.

    ### WandbPredictionProgressCallback

    Subclass [`WandbCallback`](https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.WandbCallback)
    to retain the built-in metric and checkpoint logging while adding a
    `sample_predictions` Table. The same validation examples appear at each selected
    epoch, making it easier to inspect changes during training.

    These are **teacher-forced next-token predictions**, not free-running generated
    text. Each prediction is compared with the following label token. We reduce
    logits to token IDs before collecting them, so the callback does not retain a
    full vocabulary-sized tensor for every sample.
    """)
    return


@app.class_definition
class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each selected evaluation during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 10.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=10, freq=1):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 10.
            freq (int, optional): Control the frequency of logging. Defaults to 1.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(min(num_samples, len(val_dataset))))
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions every `freq` epochs
        epoch = round(state.epoch or 0)
        if epoch > 0 and epoch % self.freq == 0:
            # generate predictions
            predictions = self.trainer.predict(self.sample_dataset)
            # decode predictions and labels
            predictions = decode_predictions(self.tokenizer, predictions)
            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            if state.is_world_process_zero:
                records_table = wandb.Table(dataframe=predictions_df)
                # log the table to wandb
                wandb.run.log({"sample_predictions": records_table})


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Trainer and training

    First instantiate `Trainer` with the model, training arguments, and train and
    evaluation datasets. Our callback needs the trainer itself, so add it afterward
    with `add_callback`. Remove the automatically installed `WandbCallback` first;
    the custom subclass already provides that integration.

    The complete run lifecycle is inside `train_model`. Each submission creates a
    fresh model and closes the run when training completes or fails. Checkpoints
    are uploaded as Weights & Biases Artifacts; temporary local checkpoints are removed afterward.
    """)
    return


@app.function
def train_model(training_config, lm_datasets, tokenizer, callback_class):
    if wandb.run is not None:
        wandb.finish()
    previous_log_model = os.environ.get("WANDB_LOG_MODEL")
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    try:
        with tempfile.TemporaryDirectory(prefix="wandb-gpt2-") as output_dir:
            with wandb.init(
                project=training_config["project"],
                entity=training_config["entity"],
                name=training_config["run_name"],
                config={
                    key: value for key, value in training_config.items()
                    if key not in {"project", "entity", "run_name"}
                },
            ) as run:
                model = AutoModelForCausalLM.from_pretrained(training_config["model"])
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    eval_strategy="epoch",
                    save_strategy="epoch",
                    save_total_limit=1,
                    num_train_epochs=training_config["epochs"],
                    learning_rate=2e-5,
                    weight_decay=0.01,
                    logging_strategy="epoch",
                    fp16=torch.cuda.is_available(),
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    dataloader_num_workers=0,
                    report_to="wandb",
                    seed=42,
                )
                # Weights & Biases already owns the display name. Use its unique run ID for
                # checkpoint Artifacts, even when the display name has spaces.
                training_args.run_name = None
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=lm_datasets["train"],
                    eval_dataset=lm_datasets["validation"],
                    processing_class=tokenizer,
                    data_collator=default_data_collator,
                    preprocess_logits_for_metrics=prediction_token_ids,
                )
                trainer.remove_callback(WandbCallback)
                progress_callback = callback_class(
                    trainer, tokenizer, lm_datasets["validation"],
                    num_samples=10, freq=int(training_config["frequency"]),
                )
                trainer.add_callback(progress_callback)
                training_result = trainer.train()
                result = {"url": run.url, "name": run.name, "metrics": training_result.metrics}
        return result
    finally:
        if previous_log_model is None:
            os.environ.pop("WANDB_LOG_MODEL", None)
        else:
            os.environ["WANDB_LOG_MODEL"] = previous_log_model


@app.cell
def _(lm_datasets, tokenizer, training_config):
    training_result = train_model(
        training_config, lm_datasets, tokenizer, WandbPredictionProgressCallback
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Visualize results in dashboard

    Open the run link above. In **Workspace**, inspect `train/loss` and
    `eval/loss`, then open the `sample_predictions` Table and compare its versions
    across epochs. The `labels`, `predictions`, and `epoch` columns show how the same
    validation examples change during training. In **Artifacts**, inspect the model
    checkpoints logged by the inherited `WandbCallback`.

    ## Learn more
    - [Documentation](https://docs.wandb.ai/models/integrations/huggingface): docs on the Weights & Biases and Hugging Face integration
    - [Videos](https://wandb.me/youtube): tutorials, interviews with practitioners, and more on our YouTube channel
    - Contact: Message us at contact@wandb.com with questions

    ## Helper functions

    These helpers tokenize and group the text, reduce evaluation memory, and align
    predictions with the causal language-model labels.
    """)
    return


@app.function
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


@app.function
def group_texts(examples, block_size):
    # Concatenate all texts.
    concatenated_examples = {
        key: list(chain.from_iterable(value)) for key, value in examples.items()
    }
    total_length = len(concatenated_examples["input_ids"])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        key: [tokens[i:i + block_size] for i in range(0, total_length, block_size)]
        for key, tokens in concatenated_examples.items()
    }
    result["labels"] = [tokens.copy() for tokens in result["input_ids"]]
    return result


@app.function
def prediction_token_ids(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


@app.function
def decode_predictions(tokenizer, predictions):
    labels = predictions.label_ids[:, 1:]
    predicted_ids = predictions.predictions[:, :-1]
    valid_tokens = labels != -100
    labels = [row[mask].tolist() for row, mask in zip(labels, valid_tokens)]
    predicted_ids = [row[mask].tolist() for row, mask in zip(predicted_ids, valid_tokens)]
    return {
        "labels": tokenizer.batch_decode(labels, skip_special_tokens=True),
        "predictions": tokenizer.batch_decode(predicted_ids, skip_special_tokens=True),
    }


if __name__ == "__main__":
    app.run()
