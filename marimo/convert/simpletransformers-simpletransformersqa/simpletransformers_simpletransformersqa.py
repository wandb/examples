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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/simpletransformers/SimpleTransformersQA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{simpletransformers-QA} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{simpletransformers-QA} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # W&B 💘 SimpleTransformers
    Use Weights & Biases for machine learning experiment tracking, dataset versioning, and project collaboration.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this notebook covers

    In this notebook we show you how to integrate
    [Weights & Biases](https://wandb.ai/site)
    with your
    [SimpleTransformers](https://github.com/ThilinaRajapakse/simpletransformers)
    code to add experiment tracking to your pipeline. This includes:

    1. dataset and model versioning with W&B,
    [Artifacts](https://docs.wandb.ai/guides/artifacts)
    2. storing configuration, hyperparameters, system metrics, and model metrics in an [interactive dashboard](https://docs.wandb.ai/guides/track/app), and
    3. examining evaluation outputs of your model using
    [W&B Tables](https://docs.wandb.ai/guides/data-vis).

    We'll add these features to a typical NLP pipeline:
    training a question-answering model with a DistilBERT backbone on (a very small subset of) the Stanford QUestion Answering Dataset ([SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)).
    The extra code required is indicated in the notebook with headers that start with "Step".

    We'll end up with an interactive dashboard ([link](https://wandb.ai/wandb/SimpleTransformers-QA?workspace=user-prashanthkurella))
    for our QA experiments that looks like this:

    <img src="https://i.imgur.com/60KMnvE.png" width="650" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 0 : Install Weights & Biases
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's get started by installing W&B.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install -Uq wandb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 1: Import `wandb`, log in, and set the project name
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now that we have installed W&B, let's import it and log in.
    If you don't have a W&B account, you'll be prompted to create one.
    W&B is free for open projects, just like GitHub.
    """)
    return


@app.cell
def _():
    import wandb

    wandb_project = "SimpleTransformers-QA"
    return wandb, wandb_project


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Install SimpleTransformers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SimpleTransformers is a library for quickly spinning up Transformer models
    for natural language processing tasks.
    It has an easy to use interface and requires only a minimal amount of code.

    You can look at [the docs](https://simpletransformers.ai/) for more about what SimpleTransformers can do.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install simpletransformers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download the SQuAD Dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We'll use a subset of the SQuAD Dataset to train a question-answering model.

    The SQuAD Dataset consists of a context and set of questions relevant to the context.
    Our model's task is to understand the context and use it to answer the questions.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # data_dir = "data"
    # raw_data_dir = data_dir + "/" + "raw"
    # 
    # !mkdir -p {raw_data_dir}
    # !curl https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json --output {raw_data_dir}/train.json
    # !curl https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json --output {raw_data_dir}/eval.json
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: Log the raw dataset

    In order to make our work more reproducible and portable,
    we'll store, version, and distribute our dataset with
    [W&B Artifacts](https://docs.wandb.ai/guides/artifacts).

    To log an Artifact, we need to start a `wandb.Run`
    with [`wandb.init`](https://docs.wandb.ai/guides/track/launch).
    We'll give it the `upload-raw-dataset` job type.
    """)
    return


@app.cell
def _(raw_data_dir, wandb, wandb_project):
    # initialize a run to log the datasets
    _run = wandb.init(project=wandb_project, job_type='upload-raw-dataset')
    raw_data_artifact = wandb.Artifact('raw-data', 'dataset')
    raw_data_artifact.add_dir(raw_data_dir)
    _run.log_artifact(raw_data_artifact)
    # log the raw data
    # finish the run
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Perform the train-test split on the raw dataset
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Datasets are typically transformed in an ML pipeline.

    For example, when training models, we usually want to split out some data to hold for validation.

    With W&B Artifacts we can track those splits
    and then reuse them when debugging models, or reproducing and extending results.
    """)
    return


@app.cell
def _(data_dir, raw_data_dir, subprocess):
    import json
    import random

    # make the data dirs for splits
    split_data_dir = data_dir + "/" + "split"
    #! mkdir -p {split_data_dir}
    subprocess.call(['mkdir', '-p', str(split_data_dir)])

    # shuffle and subset train data
    with open(f"{raw_data_dir}/train.json", "r") as f:
        train_data = json.load(f)
    train_data = [item for topic in train_data["data"] for item in topic["paragraphs"] ]
    random.shuffle(train_data)
    train_data = train_data[:int(len(train_data) * 0.01)]

    # shuffle and subset eval data
    with open(f"{raw_data_dir}/eval.json", "r") as f:
        eval_data = json.load(f)
    eval_data = [item for topic in eval_data["data"] for item in topic["paragraphs"] ]
    random.shuffle(eval_data)
    eval_data = eval_data[:int(len(eval_data) * 0.01)]

    # write the subsets to disk
    with open(f"{split_data_dir}/train.json", "w") as f:
        json.dump(train_data, f)
    with open(f"{split_data_dir}/eval.json", "w") as f:
        json.dump(eval_data, f)
    return eval_data, split_data_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 3: Log the train-test split we'll be using
    """)
    return


@app.cell
def _(split_data_dir, wandb, wandb_project):
    # initialize a run to save the datasets
    _run = wandb.init(project=wandb_project, job_type='split-dataset')
    _run.use_artifact('raw-data:latest')
    train_artifact = wandb.Artifact('train-data', 'dataset')
    train_artifact.add_file(f'{split_data_dir}/train.json')
    _run.log_artifact(train_artifact)
    # use the raw data artifact
    eval_artifact = wandb.Artifact('eval-data', 'dataset')
    eval_artifact.add_file(f'{split_data_dir}/eval.json')
    # log the training data as an artifact
    _run.log_artifact(eval_artifact)
    # log the evaluation data as an artifact
    # finish logging the data logging run
    _run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    By using Artifacts, we can track which runs produced and used particular resources,
    like datasets, models, and analyses.
    The relationships are tracked as a
    [graph](https://docs.wandb.ai/ref/app/pages/project-page#graph-view-panel).

    You can view and interact with a complete artifact graph
    for this project in the browser
    [here](https://wandb.ai/wandb/SimpleTransformers-QA/artifacts/run_table/run-3n08kirq-evalresults/ce84b13b2961e7b30e13/graph).

    You'll see square nodes, representing runs,
    and circular nodes, representing generated artifacts.
    Arrows connect runs to the artifacts they generated
    and artifacts to the runs that use them.

    In the screenshot below,
    see if you can find the runs used to upload and split the dataset
    and the dataset artifacts that those runs generated.

    <img src="https://imgur.com/SVZpMWi.png" width="600" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Configure model training
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SimpleTransformers makes it easy to run and configure your transformer model training:
    `train_args` are ["all you need"](https://arxiv.org/abs/1706.03762)
    to train your model.
    """)
    return


@app.cell
def _():
    train_args = {
        "learning_rate":                   3e-5, # learning rate of our model
        "num_train_epochs":                   2, # number of epochs 
        "max_seq_length":                   384, # maximum sequence length in tokens
        "doc_stride":                       128, # stride when processing sentences
        "overwrite_output_dir":            True, # overwrite the output directory
        "reprocess_input_data":           False, # reprocess the input data
        "train_batch_size":                  16, # training batch size
        "gradient_accumulation_steps":        1, # steps before applying gradients
        "evaluate_during_training":        True, # run evaluation during training
        "evaluate_during_training_steps":    40, # steps in training before eval
        "save_eval_checkpoints":          False, # save evaluation checkpoints
        "eval_batch_size":                   16, # evaluation batch size
    }
    return (train_args,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 4: Include `wandb_project` in `train_args` to use W&B for logging our training progress
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SimpleTransformers comes with W&B logging built in -- no extra code required.
    To enable it you just need to pass the `wandb_project` argument.

    You can also customize what's passed to the `wandb.init` function
    used to launch your training run
    with the `wandb_kwargs` argument.
    Refer to the docs
    [here](https://docs.wandb.ai/guides/track/launch)
    for more info.
    """)
    return


@app.cell
def _(train_args, wandb_project):
    train_args.update(
        {
            "logging_steps":                      1, # number of steps before logging
            "wandb_project":          wandb_project, # wandb project name
            "wandb_kwargs": {"job_type": "training"} # additional args for wandb init
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Initialize the model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another killer feature of SimpleTransformers is that it comes with a bunch of
    implementations of widely-used transformer architectures, like BERT, ALBERT, and others.

    It also includes utilities for downloading their pretrained versions and adapting
    them to specific tasks.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-display
    # from simpletransformers.question_answering import QuestionAnsweringModel
    # 
    # # initialize the model with a distilbert backbone
    # model = QuestionAnsweringModel("distilbert", "distilbert-base-cased", args=train_args)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Train the model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every time you call the `train_model` function,
    you launch a new experiment.

    W&B prints out the links to
    [project-level](https://docs.wandb.ai/ref/app/pages/project-page)
    and
    [run-level](https://docs.wandb.ai/ref/app/pages/run-page)
    dashboards.

    Click on those links to view the training progress
    and compare to other experiments.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture --no-display
    # model.train_model(train_data, eval_data=eval_data)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can reorganize your workspace into an interactive dashboard
    to share with team members or put in your portfolio.

    Below is a screenshot of a dashboard made for this project.

    You can view and interact with it in your browser
    [here](https://wandb.ai/wandb/SimpleTransformers-QA?workspace=user-prashanthkurella).

    <img src="https://imgur.com/mpVch9C.png" width="600" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Custom Logging for SimpleTransformers
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    SimpleTransformers automatically logs important metrics to W&B.

    You can also customize what you log using two methods:

    1. [Resuming](https://docs.wandb.ai/guides/track/advanced/resuming) the run,
    "restarting" the experiment so that you can log additional stuff, including more training.
    2. Using the [`wandb.api`](https://docs.wandb.ai/guides/track/public-api-guide) to update existing runs with additional metadata.

    We show both below.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use resuming to add model checkpoints
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Runs that have finished can be resumed
    so that additional information can be added to an experiment.
    For example, you might be using
    [pre-emptible compute](https://www.parkmycloud.com/blog/google-preemptible-vms/)
    where training runs can be stopped prematurely.

    Here we use it to log the model checkpoints to the training run,
    since it was responsible for creating them.
    """)
    return


@app.cell
def _(model, wandb, wandb_project):
    import os
    with wandb.init(id=model.wandb_run_id, resume='allow', project=wandb_project) as _training_run:
        for dir in sorted(os.listdir('outputs')):
            if 'checkpoint' in dir:
                artifact = wandb.Artifact('model-checkpoints', type='checkpoints')
                artifact.add_dir('outputs' + '/' + dir)
                _training_run.log_artifact(artifact)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use resuming to add evaluation results as a `Table`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To evaluate models and their performance,
    it's important to be able to visualize and analyze model predictions.
    W&B supports this workflow with
    [Tables](https://docs.wandb.ai/guides/data-vis).

    Here, we'll grab our model's predictions on the evaluation data,
    convert them into a pandas `DataFrame`,
    and then log them to W&B as a `Table` attached to the resumed run.

    For more on using Tables for NLP, check out our
    [video guide](https://www.youtube.com/watch?v=756JcKiDvqo)
    on applying Tables to the
    [GoEmotions dataset](https://arxiv.org/abs/2005.00547).
    """)
    return


@app.cell
def _(eval_data, model):
    _, outputs = model.eval_model(eval_data)
    return (outputs,)


@app.cell
def _(eval_data, model, outputs, wandb, wandb_project):
    import pandas as pd
    eval_data_df = pd.DataFrame(columns=['id', 'question', 'context'])
    # create an empty dataframe
    for context in eval_data:
        for qas in context['qas']:
            eval_data_df = eval_data_df.append([{'id': qas['id'], 'context': context['context'], 'question': qas['question']}])
    eval_data_df = eval_data_df.reset_index(drop=True)
    results = pd.DataFrame(columns=['id', 'predicted_answer', 'actual_answer', 'category'])
    for entry in outputs['correct_text']:
        results = results.append([{'id': entry, 'predicted_answer': outputs['correct_text'][entry], 'actual_answer': outputs['correct_text'][entry], 'category': 'correct'}])
    for entry in outputs['similar_text']:
    # load the eval data into the dataframe
        results = results.append([{'id': entry, 'predicted_answer': outputs['similar_text'][entry]['predicted'], 'actual_answer': outputs['similar_text'][entry]['truth'], 'category': 'similar'}])
    for entry in outputs['incorrect_text']:
        results = results.append([{'id': entry, 'predicted_answer': outputs['incorrect_text'][entry]['predicted'], 'actual_answer': outputs['incorrect_text'][entry]['truth'], 'category': 'incorrect'}])
    results = results.reset_index(drop=True)
    results = eval_data_df.set_index('id').join(results.set_index('id'))
    results = results.drop_duplicates()
    with wandb.init(resume=model.wandb_run_id, project=wandb_project) as _training_run:
    # reset index for clear indexing
    # create an empty results data frame
    # load all the correctly predicted answers
    # load all the similar answers
    # load all the incorrect answers
    # join the evaluation data with the predictions
    # resume the training run and log the table
        _training_run.log({'eval-results': wandb.Table(dataframe=results)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Inside the W&B web app,
    you can interact with logged table data
    to perform post-hoc analyses,
    including filtering, grouping, and computing derived metrics.

    Below is a screenshot of a table that compares the model's outputs
    to the actual ground truth across multiple runs.

    You can view and interact with it in your browser
    [here](https://wandb.ai/wandb/SimpleTransformers-QA/reports/Shared-panel-21-09-10-12-09-00--VmlldzoxMDExNDQw).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://imgur.com/i5wCEfF.png" width="600" alt="Weights & Biases" />
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use the API to attach the train-test splits to the training run
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Logged information from experiments and workflows
    often needs to be programmatically accessed or updated.
    For those tasks, we provide
    [a public API](https://docs.wandb.ai/guides/track/public-api-guide).

    Here we'll use it to update the training run with
    the dataset artifacts that we uploaded earlier.
    """)
    return


@app.cell
def _(model, wandb, wandb_project):
    # initialize the wandb api object
    api = wandb.Api()
    _training_run = api.run(wandb_project + '/' + model.wandb_run_id)
    # retrieve our training run
    train_data_artifact = api.artifact(wandb_project + '/' + 'train-data:latest')
    eval_data_artifact = api.artifact(wandb_project + '/' + 'eval-data:latest')
    # retrieve the artifacts we'll be using
    _training_run.use_artifact(train_data_artifact)
    # mark the training run as using the training and eval data artifacts
    _training_run.use_artifact(eval_data_artifact)
    return


if __name__ == "__main__":
    app.run()
