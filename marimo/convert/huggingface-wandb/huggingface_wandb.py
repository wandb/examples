# /// script
# dependencies = ["accelerate", "datasets", "evaluate", "transformers @ git+https://github.com/huggingface/transformers", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/Huggingface_wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{huggingface_wandb} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Hugging Face + W&B
    Visualize your [Hugging Face](https://github.com/huggingface/transformers) model's performance quickly with a seamless [W&B](https://wandb.ai/site) integration.

    Compare hyperparameters, output metrics, and system stats like GPU utilization across your models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="https://i.imgur.com/vnejHGh.png" width="800">

    <!--- @wandbcode{huggingface_wandb} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤔 Why should I use W&B?

    <img src="https://wandb.me/mini-diagram" width="650">

    - **Unified dashboard**: Central repository for all your model metrics and predictions
    - **Lightweight**: No code changes required to integrate with Hugging Face
    - **Accessible**: Free for individuals and academic teams
    - **Secure**: All projects are private by default
    - **Trusted**: Used by machine learning teams at OpenAI, Toyota, Lyft and more

    Think of W&B like GitHub for machine learning models— save machine learning experiments to your private, hosted dashboard. Experiment quickly with the confidence that all the versions of your models are saved for you, no matter where you're running your scripts.

    W&B lightweight integrations works with any Python script, and all you need to do is sign up for a free W&B account to start tracking and visualizing your models.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the Hugging Face Transformers repo, we've instrumented the Trainer to automatically log training and evaluation metrics to W&B at each logging step.

    Here's an in depth look at how the integration works: [Hugging Face + W&B Report](https://app.wandb.ai/jxmorris12/huggingface-demo/reports/Train-a-model-with-Hugging-Face-and-Weights-%26-Biases--VmlldzoxMDE2MTU).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🚀 Install, Import, and Log in
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Install the Hugging Face and Weights & Biases libraries, and the GLUE dataset and training script for this tutorial.
    - [Hugging Face Transformers](https://github.com/huggingface/transformers): Natural language models and datasets
    - [Weights & Biases](https://docs.wandb.com/): Experiment tracking and visualization
    - [GLUE dataset](https://gluebenchmark.com/): A language understanding benchmark dataset
    - [GLUE script](https://github.com/huggingface/transformers/blob/master/examples/run_glue.py): Model training script for sequence classification
    """)
    return


@app.cell
def _(subprocess):
    # packages added via marimo's package management: datasets wandb evaluate accelerate !pip install datasets wandb evaluate accelerate -qU
    #! wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py
    subprocess.call(['wget', 'https://raw.githubusercontent.com/huggingface/transformers/master/examples/pytorch/text-classification/run_glue.py'])
    return


@app.cell
def _():
    # the run_glue.py script requires transformers dev
    # packages added via marimo's package management: git+https://github.com/huggingface/transformers !pip install -q git+https://github.com/huggingface/transformers
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🖊️ [Sign up for a free account →](https://app.wandb.ai/login?signup=true)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🔑 Put in your API key
    Once you've signed up, run the next cell. You'll be prompted to create a new API key at [wandb.ai/settings](https://wandb.ai/settings) if you haven't already. Store your API key securely. It can only be viewed once when created.
    """)
    return


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Optionally, we can set environment variables to customize W&B logging. See [documentation](https://docs.wandb.com/library/integrations/huggingface).
    """)
    return


@app.cell
def _():
    # Optional: log both gradients and parameters
    import os
    os.environ['WANDB_WATCH'] = 'all'
    return (os,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 👟 Train the model
    Next, call the downloaded training script [run_glue.py](https://huggingface.co/transformers/examples.html#glue) and see training automatically get tracked to the Weights & Biases dashboard. This script fine-tunes BERT on the Microsoft Research Paraphrase Corpus— pairs of sentences with human annotations indicating whether they are semantically equivalent.
    """)
    return


@app.cell
def _(os, subprocess):
    os.environ['WANDB_PROJECT'] = 'huggingface-demo'
    os.environ['TASK_NAME'] = 'MRPC'
    #! python run_glue.py --model_name_or_path bert-base-uncased --task_name $TASK_NAME --do_train --do_eval --max_seq_length 256 --per_device_train_batch_size 32 --learning_rate 2e-4 --num_train_epochs 3 --output_dir /tmp/$TASK_NAME/ --overwrite_output_dir --logging_steps 50
    subprocess.call(['python', 'run_glue.py', '--model_name_or_path', 'bert-base-uncased', '--task_name', '$TASK_NAME', '--do_train', '--do_eval', '--max_seq_length', '256', '--per_device_train_batch_size', '32', '--learning_rate', '2e-4', '--num_train_epochs', '3', '--output_dir', '/tmp/$TASK_NAME/', '--overwrite_output_dir', '--logging_steps', '50'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 👀 Visualize results in dashboard
    Click the link printed out above, or go to [wandb.ai](https://app.wandb.ai) to see your results stream in live. The link to see your run in the browser will appear after all the dependencies are loaded — look for the following output: "**wandb**: 🚀 View run at [URL to your unique run]"

    **Visualize Model Performance**
    It's easy to look across dozens of experiments, zoom in on interesting findings, and visualize highly dimensional data.

    ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79YL90K1jiq-3jeQK-%2Fhf%20gif%2015.gif?alt=media&token=523d73f4-3f6c-499c-b7e8-ef5be0c10c2a)

    **Compare Architectures**
    Here's an example comparing [BERT vs DistilBERT](https://app.wandb.ai/jack-morris/david-vs-goliath/reports/Does-model-size-matter%3F-Comparing-BERT-and-DistilBERT-using-Sweeps--VmlldzoxMDUxNzU) — it's easy to see how different architectures effect the evaluation accuracy throughout training with automatic line plot visualizations.
    ![](https://gblobscdn.gitbook.com/assets%2F-Lqya5RvLedGEWPhtkjU%2F-M79Y5aLAFsMEcybMZcC%2F-M79Ytpj6q6Jlv9RKZGT%2Fgif%20for%20comparing%20bert.gif?alt=media&token=e3dee5de-d120-4330-b4bd-2e2ddbb8315e)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 📈 Track key information effortlessly by default
    Weights & Biases saves a new run for each experiment. Here's the information that gets saved by default:
    - **Hyperparameters**: Settings for your model are saved in Config
    - **Model Metrics**: Time series data of metrics streaming in are saved in Log
    - **Terminal Logs**: Command line outputs are saved and available in a tab
    - **System Metrics**: GPU and CPU utilization, memory, temperature etc.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🤓 Learn more!
    - [Documentation](https://docs.wandb.ai/tutorials/huggingface/): docs on the Weights & Biases and Hugging Face integration
    - [Videos](http://wandb.me/youtube): tutorials, interviews with practitioners, and more on our YouTube channel
    - Contact: Message us at contact@wandb.com with questions
    """)
    return


if __name__ == "__main__":
    app.run()
