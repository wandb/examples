# /// script
# dependencies = ["-"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/torchtune/torchtune_and_wandb.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{torchtune-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{torchtune-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Getting Started with torchtune and Weigths & Biases

    In this notebook you will learn how to use [torchtune](https://github.com/pytorch/torchtune) with [Weights & Biases](https://wandb.ai) to monitor your training runs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > You need to select a machine a GPU, go to Runtime > Change runtime type > select a GPU (L40, A100 ideally)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup the libraries
    """)
    return


@app.cell
def _(subprocess):
    #! git clone --depth 1 https://github.com/pytorch/torchtune
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/pytorch/torchtune'])
    return


app._unparsable_cell(
    r"""
    cd torchtune/
    """,
    name="_"
)


@app.cell
def _():
    # packages added via marimo's package management: .[dev] !python -m pip install -qqq ".[dev]"
    return


@app.cell
def _():
    import wandb
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Download a Model
    We will download a model from the Hugging Face Hub.
    > you will need to provide an access token or call `huggingface-cli login`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Download a model checkpoint using the provided `tune download` CLI
    """)
    return


@app.cell
def _(subprocess):
    #! tune download mistralai/Mistral-7B-v0.1 --output-dir /tmp/Mistral-7B-v0.1/ --hf-token <HF_TOKEN>
    subprocess.call(['tune', 'download', 'mistralai/Mistral-7B-v0.1', '--output-dir', '/tmp/Mistral-7B-v0.1/', '--hf-token', '<HF_TOKEN>'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's create a torchtune config that enables W&B, to do so, we can grab the original Mistral 7B LoRA recipe and change the following lines to use W&B as our `metric_logger`:
    ```yaml
    # Logging
    metric_logger:
      _component_: torchtune.utils.metric_logging.WandBLogger # <---You only need this to enable W&B
      project: mistral_lora # <--- The W&B project to save our logs to
    log_every_n_steps: 1

    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's save a modified version of the recipe using `%%writefile`:
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%writefile mistral_wandb_lora.yaml
    # tokenizer:
    #   _component_: torchtune.models.mistral.mistral_tokenizer
    #   path: /tmp/Mistral-7B-v0.1/tokenizer.model
    # 
    # # Dataset
    # dataset:
    #   _component_: torchtune.datasets.alpaca_dataset
    #   train_on_input: True
    # seed: null
    # shuffle: True
    # 
    # # Model Arguments
    # model:
    #   _component_: torchtune.models.mistral.lora_mistral_7b
    #   lora_attn_modules: ['q_proj', 'k_proj', 'v_proj']
    #   apply_lora_to_mlp: True
    #   apply_lora_to_output: True
    #   lora_rank: 64
    #   lora_alpha: 16
    # 
    # checkpointer:
    #   _component_: torchtune.utils.FullModelHFCheckpointer
    #   checkpoint_dir: /tmp/Mistral-7B-v0.1
    #   checkpoint_files: [
    #     pytorch_model-00001-of-00002.bin,
    #     pytorch_model-00002-of-00002.bin
    #   ]
    #   recipe_checkpoint: null
    #   output_dir: /tmp/Mistral-7B-v0.1
    #   model_type: MISTRAL
    # resume_from_checkpoint: False
    # 
    # optimizer:
    #   _component_: torch.optim.AdamW
    #   lr: 2e-5
    # 
    # lr_scheduler:
    #   _component_: torchtune.modules.get_cosine_schedule_with_warmup
    #   num_warmup_steps: 100
    # 
    # loss:
    #   _component_: torch.nn.CrossEntropyLoss
    # 
    # # Fine-tuning arguments
    # batch_size: 2
    # epochs: 1
    # max_steps_per_epoch: 100
    # gradient_accumulation_steps: 2
    # compile: False
    # 
    # # Training env
    # device: cuda
    # 
    # # Memory management
    # enable_activation_checkpointing: True
    # 
    # # Reduced precision
    # dtype: bf16
    # ############################### Enable W&B #####################################
    # ################################################################################
    # # Logging
    # metric_logger:
    #   _component_: torchtune.utils.metric_logging.WandBLogger # <---You only need this to enable W&B
    #   project: mistral_lora # <--- The W&B project to save our logs to
    # log_every_n_steps: 1
    # ################################################################################
    # ################################################################################
    # output_dir: /tmp/Mistral-7B-v0.1
    # log_peak_memory_stats: False
    # 
    # # Profiler (disabled)
    # profiler:
    #   _component_: torchtune.utils.profiler
    #   enabled: False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's run the recipe with this modified config with W&B enabled
    """)
    return


@app.cell
def _(subprocess):
    #! tune run lora_finetune_single_device --config mistral_wandb_lora.yaml
    subprocess.call(['tune', 'run', 'lora_finetune_single_device', '--config', 'mistral_wandb_lora.yaml'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    That's it! Now you can click on the URL and continue monitoring your training on the Weights & Biases UI
    """)
    return


if __name__ == "__main__":
    app.run()
