# /// script
# dependencies = ["//", "accelerate", "datasets", "evaluate", "github-com/huggingface/transformers", "package @ git+https:", "qqq", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/wandb_hf_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{huggingface_wandb} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🏃‍♀️ Introduction
    [Hugging Face](https://huggingface.co/) provides tools to quickly train neural networks for NLP (Natural Language Processing) on any task (classification, translation, question answering, etc) and any dataset with PyTorch and TensorFlow 2.0.

    ## 🤔 Why should I use W&B?

    <img src="https://wandb.me/mini-diagram" width="650">

    - **Unified dashboard**: Central repository for all your model metrics and predictions
    - **Lightweight**: No code changes required to integrate with Hugging Face
    - **Accessible**: Free for individuals and academic teams
    - **Secure**: All projects are private by default
    - **Trusted**: Used by machine learning teams at OpenAI, Toyota, Lyft and more

    Think of Weights & Biases like GitHub for machine learning models — save machine learning experiments to your private, hosted dashboard. Experiment quickly with the confidence that all the versions of your models are saved for you, no matter where you're running your scripts.

    W&B lightweight integrations works with any Python script, and all you need to do is sign up for a free W&B account to start tracking and visualizing your models.

    In the HuggingFace Transformers repo, we've instrumented the Trainer to automatically log training and evaluation metrics to W&B at each logging step.

    Here's an in depth look at how the integration works: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌴 Installation and Setup

    First, let us install the latest version of Weights and Biases. We will then setup a few environment variables to enable Weights & Biases logging and finally authenticate this colab instance to use W&B.

    **Note**: To enable logging to W&B, you will also need to set the `report_to` argument in your `TrainingArguments` or script to `wandb`.
    """)
    return


@app.cell
def _():
    # Install required transformer libraries along with wandb
    # packages added via marimo's package management: qqq evaluate datasets wandb accelerate git+https: // github.com/huggingface/transformers !pip install - qqq evaluate datasets wandb accelerate git+https: // github.com/huggingface/transformers
    return


@app.cell
def _():
    # Setup enviroment variables to enable logging to Weights & Biases

    import os
    # can be "end", "checkpoint" or "false"
    os.environ['WANDB_LOG_MODEL'] = "checkpoint"
    # the name of the wandb project defaults to `huggingface`
    os.environ['WANDB_PROJECT'] = "hf_transformers"
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🖊️ Sign-up/login
    If this is your first time using Weights & Biases or you are not logged in, the link that appears after running `wandb.login()` in the following code cell will take you to sign-up/login page. Signing up for a [free account](https://wandb.ai/signup) is as easy as a few clicks.

    ## 🔑 Authentication
    Once you've signed up, run the next cell. You'll be prompted to create a new API key at [wandb.ai/settings](https://wandb.ai/settings) if you haven't already. Store your API key securely. It can only be viewed once when created.
    """)
    return


@app.cell
def _():
    # Login and authenticate Weights & Biases
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task

    Text classification is a common NLP task that assigns a label or class to text. Some of the largest companies run text classification in production for a wide range of practical applications. In this example we will use the [TweetEval](https://arxiv.org/abs/2010.12421) dataset to classify tweets into identify the emotions evoked by a tweet. The dataset is used as a benchmark to train models for tweet classification tasks. We will use then use a distilled verison of RoBERTa model - [distilroberta-base](https://huggingface.co/distilroberta-base) to recoganize the emotions evoked by the tweets.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading the data
    Start by loading the tweet_eval dataset from the 🤗 Datasets library:
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    dataset = load_dataset("tweet_eval", "emotion")
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Understanding the dataset
    """)
    return


@app.cell
def _(dataset):
    # What does the dataset look like ?
    print(dataset)

    # look at an example record
    print("\nSample Record:", end="\t")
    print(dataset["validation"][0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are two fields in this dataset:

    - `text`: The text of the tweet.
    - `label`: The integer label of the emotion corresponding to the tweet
    """)
    return


@app.cell
def _(dataset):
    # What do the labels mean ?
    idx2label = dict(enumerate(dataset["train"].features["label"].names))
    label2idx = {v: k for k, v in idx2label.items()}

    print(idx2label)
    return idx2label, label2idx


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We need to convert the `text` to integer tokens so that they can be passed into the model as inputs. To do this we will use the  `distilroberta` tokenizer to preprocess the `text` field in the dataset.
    """)
    return


@app.cell
def _():
    from transformers import AutoTokenizer
    MODEL_NAME = "distilroberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return MODEL_NAME, tokenizer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a preprocessing function to tokenize `text` and truncate sequences to be no longer than distilroberta's maximum input length:
    """)
    return


@app.cell
def _(tokenizer):
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    return (preprocess_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map) function. You can speed up `map` by setting `batched=True` to process multiple elements of the dataset at once:
    """)
    return


@app.cell
def _(dataset, preprocess_function):
    tokenized_ds = dataset.map(preprocess_function, batched=True,)
    tokenized_ds
    return (tokenized_ds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The above step added two new columns to our dataset. `input_ids` and `attention_mask`. These are the inputs we will be passing to our model.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since all our examples are of different lengths and the model expects a batch of tokens with the same length we will need to pad our inputs. We can use the `DataCollatorWithPadding` utility to do this. To further speed up training we will pre-compute the length of texts in the tokenized dataset and sort the dataset by this column. This ensures that the batches of data have as minimal padding as possible.
    """)
    return


@app.cell
def _(tokenized_ds):
    def length_function(examples):
        return {'length': [len(example) for example in examples['input_ids']]}
    tokenized_ds_1 = tokenized_ds.map(length_function, batched=True)
    tokenized_ds_1 = tokenized_ds_1.sort('length')
    return (tokenized_ds_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now create a batch of examples using [DataCollatorWithPadding](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithPadding). It's more efficient to *dynamically pad* the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximium length.
    """)
    return


@app.cell
def _(tokenizer):
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return (data_collator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Evaluation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Including a metric during training is often helpful for evaluating your model's performance. You can quickly load a evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [f1-score](https://huggingface.co/spaces/evaluate-metric/f1) metric. This is the metric used in the TweetEval benchmark.
    You will notice that this metric get logged automatically to your weights & biases run while training.
    """)
    return


@app.cell
def _():
    import evaluate

    f1_score = evaluate.load("f1")
    return (f1_score,)


@app.cell
def _(f1_score):
    import numpy as np


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return f1_score.compute(predictions=predictions,
                                references=labels,
                                average="weighted")

    return (compute_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Your `compute_metrics` function is ready to go now, and you'll return to it when you setup your training.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train
    """)
    return


@app.cell
def _(MODEL_NAME, idx2label, label2idx):
    from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(idx2label),
        id2label=idx2label,
        label2id=label2idx,
        attention_probs_dropout_prob=0.2,
        hidden_dropout_prob=0.3)
    return Trainer, TrainingArguments, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We are almost ready to train our model. The steps that remain include:

    1. Define your training hyperparameters in [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments). The only required parameter is `output_dir` which specifies where to save your model. You'll also add the `report_to="wandb"` argument here. At the end of each epoch, the [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) will evaluate the accuracy and save the training checkpoint. These metrics and checkpoints are automatically pushed to your wandb project.
    2. Pass the training arguments to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
    3. Call [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.
    """)
    return


@app.cell
def _(TrainingArguments):
    training_args = TrainingArguments(
        output_dir="my_emotion_model",
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        warmup_steps=50,
        save_total_limit=2,
        report_to="wandb",  # enable logging metrics and model checkpoints to Weights & Biases
    )
    return (training_args,)


@app.cell
def _(
    Trainer,
    compute_metrics,
    data_collator,
    model,
    tokenized_ds_1,
    tokenizer,
    training_args,
):
    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_ds_1['train'], eval_dataset=tokenized_ds_1['validation'], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
    trainer.train()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can visuzalize the training logs by looking at the wandb.run object or by clicking the link printed out above, or go to wandb.ai to see your results stream in live. The link to see your run in the browser will appear just before the training begins — look for the following output: "wandb: 🚀 View run at [URL to your unique run]"
    """)
    return


@app.cell
def _(wandb):
    wandb.run
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Finally, we can optionally call the `wandb.finish()` method to indicate that the experiment is complete.
    """)
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resuming Training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But wait!! Looks like the model did not converge. Perhaps we should train for a few more epochs. Additionally, since we are training the model on colab it is possible that the preemptible instance was shutdown midway and that the model was not fully trained. Don't worry the wandb integration got us fully covered. We can easily resume training from the last checkpoint by doing the following.

    1. Initialize the last wandb run by passing the `run id` from your Weights & Biases workspace to `wandb.init`
    2. Download the lastest checkpoint using `wandb.artifact`.
    3. Reinitialize the trainer and pass the `artifact_dir` to the `resume_from_checkpoint` argument in the `trainer.train` method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note: Change the `last_run_id` in the below cell to the id from your wandb run`**
    """)
    return


@app.cell
def _(os, wandb):
    last_run_id = "25d6hznl"  # fetch the run_id from your wandb workspace

    # resume the wandb run from the run_id
    run = wandb.init(
        project=os.environ["WANDB_PROJECT"],
        id=last_run_id,
        resume="must",
    )
    return last_run_id, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Note: Change the `latest_checpoint`in the below cell to the checkpoint artifact from your run**
    """)
    return


@app.cell
def _(last_run_id, run):
    # fetch the checkpoint artifact from the run
    # this is usually in the format "<username>/<project>checkpoint-<run_id>:<version>
    latest_checkpoint = f'parambharat/hf_transformers/checkpoint-{last_run_id}:v5'
    _artifact = run.use_artifact(latest_checkpoint, type='model')
    artifact_dir = _artifact.download()
    return (artifact_dir,)


@app.cell
def _(
    Trainer,
    TrainingArguments,
    compute_metrics,
    data_collator,
    model,
    tokenized_ds_1,
    tokenizer,
):
    # recreate the training arguments with more epochs
    training_args_1 = TrainingArguments(output_dir='my_emotion_model', learning_rate=2e-05, per_device_train_batch_size=128, per_device_eval_batch_size=128, num_train_epochs=12, weight_decay=0.01, evaluation_strategy='epoch', save_strategy='epoch', logging_strategy='steps', logging_steps=25, load_best_model_at_end=True, warmup_steps=50, save_total_limit=2, report_to='wandb')
    # reinitialize the trainer object
    trainer_1 = Trainer(model=model, args=training_args_1, train_dataset=tokenized_ds_1['train'], eval_dataset=tokenized_ds_1['validation'], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)  # change the number of epochs to train
    return (trainer_1,)


@app.cell
def _(artifact_dir, trainer_1):
    trainer_1.train(resume_from_checkpoint=artifact_dir)
    return


@app.cell
def _(wandb):
    wandb.run
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inference
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Great, now that you've finetuned a model, you can use it for inference!

    Grab some text you'd like to run inference on:
    """)
    return


@app.cell
def _():
    text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
    return (text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The simplest way to try out your finetuned model for inference is to use it in a [pipeline()](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline). Instantiate a `pipeline` for sentiment analysis with your model, and pass your text to it. Here we will create a new wandb.run to download the model artifact. Then we simply pass the `artifact_dir` as the pretrained model to the `model` argument in the pipeline.
    """)
    return


@app.cell
def _(last_run_id, os, wandb):
    run_1 = wandb.init(project=os.environ['WANDB_PROJECT'], job_type='inference')
    latest_model = f'parambharat/hf_transformers/model-{last_run_id}:latest'
    _artifact = run_1.use_artifact(latest_model, type='model')
    artifact_dir_1 = _artifact.download()
    return (artifact_dir_1,)


@app.cell
def _(artifact_dir_1, text):
    from transformers import pipeline
    classifier = pipeline('sentiment-analysis', model=artifact_dir_1)
    predictions = classifier(text)
    print(predictions)
    return


if __name__ == "__main__":
    app.run()
