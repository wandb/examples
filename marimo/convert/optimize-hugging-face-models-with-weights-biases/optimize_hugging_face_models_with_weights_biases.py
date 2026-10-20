# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "accelerate>=1.10",
#     "datasets>=3.0",
#     "evaluate>=0.4",
#     "fsspec[http]>=2025.7.0",
#     "ipython>=9,<10",
#     "marimo>=0.24.0",
#     "numpy>=1.26",
#     "protobuf>=5",
#     "scikit-learn>=1.5",
#     "scipy>=1.13",
#     "sentencepiece>=0.2,!=0.1.92",
#     "torch>=2.6",
#     "transformers @ git+https://github.com/huggingface/transformers.git@02d8fb9784e8f14a1251e4c992cd82a5762417c6",
#     "wandb>=0.18",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    app_title="Optimize Hugging Face Models with Weights & Biases",
)

with app.setup:
    import os
    import subprocess
    import sys
    from pathlib import Path

    # This lesson uses PyTorch; avoid optional TensorFlow/Keras imports.
    os.environ["USE_TF"] = "0"

    import fsspec
    import marimo as mo
    import numpy as np
    import torch
    import wandb
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
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

    [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/wandb/examples/blob/main/marimo/convert/optimize-hugging-face-models-with-weights-biases/optimize_hugging_face_models_with_weights_biases.py/server)

    # Optimize Hugging Face models with Weights & Biases

    [Hugging Face](https://huggingface.co/) provides tools to quickly train neural networks for NLP (Natural Language Processing) on any task (classification, translation, question answering, etc) and any dataset with PyTorch and TensorFlow 2.0.

    Coupled with [Weights & Biases integration](https://docs.wandb.ai/models/integrations/huggingface_transformers), you can quickly train and monitor models for full traceability and reproducibility without any extra line of code! The notebook metadata provides the libraries; configure Weights & Biases credentials below before starting an experiment.

    **Note**: To enable logging to Weights & Biases, set `report_to` to `wandb` in your `TrainingArguments` or script.

    Weights & Biases integration with 🤗 Hugging Face can automatically:
    * log your configuration parameters
    * log your losses and metrics
    * log gradients and parameter distributions
    * log your model
    * keep track of your code
    * log your system metrics (GPU, CPU, memory, temperature, etc)
    ## Installation and set-up

    We need the following 🤗 Hugging Face libraries:
    * [transformers](https://huggingface.co/transformers/) contains an API for training models and many pre-trained models
    * [tokenizers](https://huggingface.co/docs/tokenizers) is automatically installed by transformers and "tokenize" our data (ie it converts text to sequence of numbers)
    * [datasets](https://huggingface.co/docs/datasets/) contains a rich source of data and common metrics, perfect for prototyping

    The notebook dependency metadata includes these libraries and `wandb`.
    Transformers and `run_glue.py` use the same pinned source revision so the
    script and installed APIs stay compatible. Keep the IPython dependency:
    Transformers uses its notebook progress callback when marimo is loaded.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Configure Weights & Biases and authenticate

    Set `WANDB_API_KEY` in your environment or molab Secrets, or use credentials
    already stored in this runtime. In a terminal-backed session, `wandb.login()`
    can also prompt for your [API key](https://wandb.ai/authorize). A fresh molab
    session does not inherit credentials from your computer.

    Edit the project and optional team entity below. `WANDB_LOG_MODEL="end"`
    tells the Transformers callback to log the trained model as a Weights & Biases Artifact.
    `WANDB_WATCH="all"` enables gradient and parameter histograms. Both examples
    enable metrics with `report_to="wandb"` and give the run a descriptive name.

    > Note: Running this notebook downloads BERT and MRPC, then starts a 300-step
    training run. The custom example downloads Yahoo! Answers and DistilBERT.
    """)
    return


@app.cell
def _():
    project = "huggingface-demo"
    entity = None  # Set this to your Weights & Biases team.
    os.environ["WANDB_LOG_MODEL"] = "end"
    os.environ["WANDB_WATCH"] = "all"
    if not wandb.login():
        raise RuntimeError("Configure Weights & Biases credentials before running this lesson.")
    return entity, project


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Train with an existing Hugging Face script

    The [task guides](https://huggingface.co/docs/transformers/tasks) provide
    ready-made training workflows. Here we use `run_glue.py` to fine-tune BERT on
    MRPC, a sentence-pair classification task.

    The next cells download the script from the same source revision as the
    installed Transformers library, then run it with this Python environment.
    `--report_to wandb` activates the integration; Weights & Biases prints the run link in
    the subprocess output. The script finishes its run when the process exits.

    We evaluate and save every 30 steps, retain two checkpoints, and reload the
    checkpoint with the best accuracy. `WANDB_LOG_MODEL="end"` then logs that
    model as an Artifact. Files remain in `hf_assets/` and `hf_outputs/` so you
    can inspect and reuse them. This example can take several minutes.
    """)
    return


@app.cell
def _():
    transformers_revision = "02d8fb9784e8f14a1251e4c992cd82a5762417c6"
    quick_run_name = "bert-mrpc-300-steps"
    quick_max_steps = 300
    quick_output_dir = Path("hf_outputs/mrpc")
    return (
        quick_max_steps,
        quick_output_dir,
        quick_run_name,
        transformers_revision,
    )


@app.cell
def _(transformers_revision):
    run_glue_url = (
        "https://raw.githubusercontent.com/huggingface/transformers/"
        f"{transformers_revision}/examples/pytorch/text-classification/run_glue.py"
    )
    run_glue_path = Path("hf_assets") / f"run_glue-{transformers_revision[:12]}.py"
    run_glue_path.parent.mkdir(parents=True, exist_ok=True)
    with fsspec.open(run_glue_url, "rb") as glue_source:
        run_glue_path.write_bytes(glue_source.read())
    return (run_glue_path,)


@app.cell
def _(
    entity,
    project,
    quick_max_steps,
    quick_output_dir,
    quick_run_name,
    run_glue_path,
):
    # Pass the project and entity to the separate Python process.
    quick_environment = os.environ.copy()
    quick_environment["WANDB_PROJECT"] = project
    if entity:
        quick_environment["WANDB_ENTITY"] = entity
    else:
        quick_environment.pop("WANDB_ENTITY", None)

    quick_training_process = subprocess.run(
        [
            sys.executable,
            str(run_glue_path),
            "--report_to", "wandb",
            "--model_name_or_path", "bert-base-uncased",
            "--task_name", "mrpc",
            "--learning_rate", "1e-4",
            "--do_train",
            "--do_eval",
            "--max_steps", str(quick_max_steps),
            "--logging_steps", "30",
            "--eval_strategy", "steps",
            "--eval_steps", "30",
            "--save_steps", "30",
            "--save_total_limit", "2",
            "--load_best_model_at_end", "true",
            "--metric_for_best_model", "accuracy",
            "--output_dir", str(quick_output_dir),
            "--run_name", quick_run_name,
        ],
        env=quick_environment,
        check=True,
    )
    quick_training_complete = quick_training_process.returncode == 0
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Open the Weights & Biases run link printed by the script to inspect training loss, MRPC
    evaluation metrics, and system telemetry. Open the run's **Artifacts** tab
    to find the saved model and its evaluation metadata.

    ## Advanced usage: customize a Trainer workflow

    Now build a topic classifier with DistilBERT and Yahoo! Answers. This path
    makes dataset preparation, tokenization, model loading, metrics, and the Weights & Biases
    run lifecycle explicit.

    A sequence classifier maps text to a class. For example:

    | Input | Class |
    | --- | --- |
    | The team scored a goal in the last seconds | sports |
    | The debate was heated between the two parties | politics |
    | I've never tasted croissants so delicious! | food |

    The [Datasets Hub](https://huggingface.co/datasets) provides many starting
    points. We use [Yahoo! Answers Topic Classification](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics),
    which has ten topics such as Society & Culture, Science & Mathematics, and
    Health. We use `question_title` as the input and `topic` as the target.
    The following cells download and preprocess the full dataset, which can
    take time and disk space.
    """)
    return


@app.cell
def _():
    model_checkpoint = "distilbert-base-uncased"
    custom_run_name = "distilbert-yahoo-topics"
    custom_max_steps = 30_000
    custom_output_dir = Path("hf_outputs/topic_classification")
    return (
        custom_max_steps,
        custom_output_dir,
        custom_run_name,
        model_checkpoint,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load and inspect the dataset

    `load_dataset()` downloads data when it is not cached. The resulting
    `DatasetDict` shows the splits, features, and row counts. Inspect one row,
    then use the `ClassLabel` feature's `int2str()` and `str2int()` methods to
    translate between topic IDs and names.
    """)
    return


@app.cell
def _():
    dataset = load_dataset("community-datasets/yahoo_answers_topics")
    dataset
    return (dataset,)


@app.cell
def _(dataset):
    dataset["train"][0]
    return


@app.cell
def _(dataset):
    topic_feature = dataset["train"].features["topic"]
    label_names = topic_feature.names
    num_labels = topic_feature.num_classes
    print(f"{num_labels} topics: {label_names}")
    topic_feature.int2str(4)
    return label_names, num_labels


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Rename `topic` to `labels`, the target column expected by `Trainer`. Keep the
    original dataset available for inspection and give each preparation stage
    its own name so marimo can track the dependencies.
    """)
    return


@app.cell
def _(dataset):
    labeled_dataset = dataset.rename_column("topic", "labels")
    return (labeled_dataset,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tokenize the text

    A tokenizer splits text into tokens and maps them to integer IDs.
    `AutoTokenizer` loads the tokenizer associated with the model checkpoint.
    Inspect a single title and its tokenization before applying it in batches
    with `Dataset.map()`, which caches the processed data locally.

    Truncation limits inputs to the model's supported length. During training,
    passing the tokenizer as `processing_class` lets `Trainer` dynamically pad
    each batch. The tokenized rows include `input_ids` and `attention_mask`.
    """)
    return


@app.cell
def _(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return (tokenizer,)


@app.cell
def _(labeled_dataset, tokenizer):
    sample_input = labeled_dataset["train"][0]["question_title"]
    print(sample_input)
    tokenizer(sample_input)
    return


@app.cell
def _(labeled_dataset, tokenizer):
    tokenized_dataset = labeled_dataset.map(
        lambda batch: tokenizer(batch["question_title"], truncation=True),
        batched=True,
    )
    tokenized_dataset["train"][0]
    return (tokenized_dataset,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Load a model and inspect an initial prediction

    Explore supported architectures on the [Model Hub](https://huggingface.co/models).
    `AutoModelForSequenceClassification` loads a pretrained encoder and adds a
    classification head for our ten topics. The new head has not learned the
    topic labels yet, so its initial predictions are a baseline.

    `get_topic()` tokenizes a sentence, moves its tensors to the model's device,
    and selects the largest output logit. Evaluation mode disables dropout;
    `torch.no_grad()` avoids building a gradient graph for inference.
    """)
    return


@app.cell
def _(label_names, model_checkpoint, num_labels):
    set_seed(42)
    untrained_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=dict(enumerate(label_names)),
        label2id={label: index for index, label in enumerate(label_names)},
    )
    return (untrained_model,)


@app.function
def get_topic(sentence, model, tokenizer):
    model.eval()
    model_device = next(model.parameters()).device
    inputs = tokenizer(sentence, truncation=True, return_tensors="pt")
    inputs = {name: tensor.to(model_device) for name, tensor in inputs.items()}
    with torch.no_grad():
        predicted_id = model(**inputs).logits.argmax(dim=-1).item()
    return model.config.id2label[predicted_id]


@app.cell
def _(tokenizer, untrained_model):
    sample_sentence = "Why is cheese so much better with wine?"
    initial_topic_prediction = get_topic(sample_sentence, untrained_model, tokenizer)
    initial_topic_prediction
    return initial_topic_prediction, sample_sentence


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Define an evaluation metric

    `Trainer` passes model logits and reference labels to `compute_metrics()`.
    Convert logits to the most likely class with `argmax`, then compute accuracy
    with NumPy. You can use metrics from the [Evaluate library](https://huggingface.co/docs/evaluate)
    here for tasks that need other scores.
    """)
    return


@app.function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": float(np.mean(predictions == labels))}


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Configure training and Weights & Biases logging

    The next cell configures `Trainer` for 30,000 steps. `report_to="wandb"`
    installs the Transformers Weights & Biases callback, which logs the model configuration,
    training loss, evaluation accuracy, and other metrics automatically.

    Evaluate and save every 5,000 steps so each evaluated model has a checkpoint.
    At the end, reload the checkpoint with the best accuracy and log it as a
    model Artifact.
    To try a shorter run, reduce `custom_max_steps` and the evaluation and save
    intervals together. Local checkpoints remain in `custom_output_dir`.
    """)
    return


@app.cell
def _(custom_max_steps, custom_output_dir, custom_run_name):
    training_args = TrainingArguments(
        report_to="wandb",
        output_dir=str(custom_output_dir),
        eval_strategy="steps",
        learning_rate=5e-5,
        max_steps=custom_max_steps,
        logging_steps=100,
        eval_steps=5_000,
        save_steps=5_000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        run_name=custom_run_name,
    )
    return (training_args,)


@app.cell(hide_code=True)
def _(custom_max_steps):
    mo.md(f"""
    ### Evaluate the baseline and train

    The next action runs baseline evaluation on Yahoo! Answers and starts
    **{custom_max_steps:,} training steps**. It creates a Weights & Biases run and uploads
    metrics and the trained model. This can take substantial time and compute.

    We load a fresh copy of the pretrained checkpoint for each run. The explicit
    `with wandb.init(...) as training_run` block connects the callback to that
    run and finishes it when training exits, including after an error.
    """)
    return


@app.cell
def _(
    custom_run_name,
    entity,
    initial_topic_prediction,
    label_names,
    model_checkpoint,
    num_labels,
    project,
    tokenized_dataset,
    tokenizer,
    training_args,
):
    set_seed(training_args.seed)

    training_model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=dict(enumerate(label_names)),
        label2id={label: index for index, label in enumerate(label_names)},
    )
    with wandb.init(project=project, entity=entity, name=custom_run_name) as training_run:
        print(f"Initial sample prediction: {initial_topic_prediction}")
        trainer = Trainer(
            model=training_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            processing_class=tokenizer,
            compute_metrics=compute_metrics,
        )
        initial_metrics = trainer.evaluate()
        print(f"Baseline metrics: {initial_metrics}")
        training_result = trainer.train()
        trained_model = trainer.model
        training_run_url = training_run.url
    return (trained_model,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Predict with the trained model

    The next cell consumes `trained_model`, which is returned only after
    training and Weights & Biases cleanup finish. It classifies the same sentence as the
    baseline so you can compare the predictions.
    """)
    return


@app.cell
def _(sample_sentence, tokenizer, trained_model):
    trained_topic_prediction = get_topic(sample_sentence, trained_model, tokenizer)
    trained_topic_prediction
    return


@app.cell(hide_code=True)
def _():
    mo.md(f"""
    ## Verify and next steps

    Inspect `train/loss`, `eval/loss`, and `eval/accuracy`, the recorded configuration,
    and gradient/parameter histograms. The **Artifacts** tab contains the
    trained model with its performance metadata.

    Once you are happy with a model, you can
    [share it on the Hugging Face Hub](https://huggingface.co/docs/transformers/model_sharing).

    ## Resources

    * [Hugging Face and Weights & Biases integration documentation](https://docs.wandb.ai/models/integrations/huggingface) contains tips for making the most of Weights & Biases
    * [🤗 Transformers documentation](https://huggingface.co/docs/transformers/) is extremely thorough and full of examples
    """)
    return


if __name__ == "__main__":
    app.run()
