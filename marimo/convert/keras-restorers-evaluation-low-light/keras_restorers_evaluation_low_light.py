# /// script
# dependencies = ["pip", "restorers @ git+https://github.com/soumik12345/restorers.git", "setuptools"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/restorers/Evaluation_low_light.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{restorers-evaluation} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🌈 Restorers + WandB 🪄🐝

    <!--- @wandbcode{restorers-mirnetv2-train} -->

    This notebook shows how to perform inference with a low-light enhancement using [**restorers**](https://github.com/soumik12345/restorers) and [**wandb**](https://wandb.ai/site). For more details regarding usage of restorers, refer to the following report:

    [![](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-gradient.svg)](https://wandb.ai/ml-colabs/low-light-enhancement/reports/Lighting-up-Images-in-the-Deep-Learning-Era--VmlldzozNzE4Njkz)
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: pip setuptools !pip install -q --upgrade pip setuptools
    # packages added via marimo's package management: git+https://github.com/soumik12345/restorers.git !pip install git+https://github.com/soumik12345/restorers.git
    return


@app.cell
def _():
    import wandb
    from restorers.evaluation import LoLEvaluator
    from restorers.metrics import PSNRMetric, SSIMMetric

    return LoLEvaluator, PSNRMetric, SSIMMetric, wandb


@app.cell
def _(wandb):
    # initialize a wandb run for inference
    wandb.init(project="low-light-enhancement", job_type="evaluation")
    return


@app.cell
def _(LoLEvaluator, PSNRMetric, SSIMMetric):
    # Define the Evaluator for LoL dataset
    evaluator = LoLEvaluator(
        # pass the list of Keras metrics to be evaluated for
        metrics=[PSNRMetric(max_val=1.0), SSIMMetric(max_val=1.0)],
        # pass the wandb artifact for the LoL dataset
        dataset_artifact_address="ml-colabs/dataset/LoL:v0",
        input_size=256,
    )
    # initialize model from wandb artifacts
    evaluator.initialize_model_from_wandb_artifact("artifact-address-of-your-model-checkpoint")
    # evaluate
    evaluator.evaluate()
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
