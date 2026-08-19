# /// script
# dependencies = ["accelerate", "bitsandbytes", "ctranslate2", "datasets", "loralib", "peft @ git+https://github.com/huggingface/peft.git", "transformers.git@main @ git+https://github.com/huggingface/transformers.git@main", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/huggingface/LLM_Finetuning_Notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{llm-finetune-hf} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # LLM Finetuning with HuggingFace and Weights and Biases
    <!--- @wandbcode{llm-finetune-hf} -->
    - Fine-tune a lightweight LLM (OPT-125M) with LoRA and 8-bit quantization using Launch
    - Checkpoint the LoRA adapter weights as artifacts
    - Link the best checkpoint in Model Registry
    - Run inference on a quantized model

    The same workflow and principles from this notebook can be applied to fine-tuning some of the stronger OSS LLMs (e.g. Llama2)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fine-tune large models using 🤗 `peft` adapters, `transformers` & `bitsandbytes`

    In this tutorial we will cover how we can fine-tune large language models using the very recent `peft` library and `bitsandbytes` for loading large models in 8-bit.
    The fine-tuning method will rely on a recent method called "Low Rank Adapters" (LoRA), instead of fine-tuning the entire model you just have to fine-tune these adapters and load them properly inside the model.
    After fine-tuning the model you can also share your adapters on the 🤗 Hub and load them very easily. Let's get started!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Install requirements

    First, run the cells below to install the requirements:
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: bitsandbytes datasets accelerate loralib !pip install -q bitsandbytes datasets accelerate loralib
    # packages added via marimo's package management: git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git !pip install -q git+https://github.com/huggingface/transformers.git@main git+https://github.com/huggingface/peft.git
    # packages added via marimo's package management: wandb !pip install -q wandb
    # packages added via marimo's package management: ctranslate2 !pip install -q ctranslate2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Model Loading

    - Here we leverage 8-bit quantization to reduce the memory footprint of the model during training
    """)
    return


@app.cell
def _():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    import torch
    import torch.nn as nn
    import bitsandbytes as bnb
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        load_in_8bit=True,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    return AutoModelForCausalLM, AutoTokenizer, model, nn, os, tokenizer, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Post-processing on the model

    Finally, we need to apply some post-processing on the 8-bit model to enable training, let's freeze all our layers, and cast the layer-norm in `float32` for stability. We also cast the output of the last layer in `float32` for the same reasons.
    """)
    return


@app.cell
def _(model, nn, torch):
    for param in model.parameters():
      param.requires_grad = False  # freeze the model - train adapters later
      if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    class CastOutputToFloat(nn.Sequential):
      def forward(self, x): return super().forward(x).to(torch.float32)
    model.lm_head = CastOutputToFloat(model.lm_head)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Apply LoRA

    Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
    """)
    return


@app.function
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param = all_param + param.numel()
        if param.requires_grad:
            trainable_params = trainable_params + param.numel()
    print(f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}')


@app.cell
def _(model):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')
    model_1 = get_peft_model(model, config)
    print_trainable_parameters(model_1)
    return (model_1,)


@app.cell
def _(model_1):
    model_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Training
    - [W&B HuggingFace integration](https://docs.wandb.ai/guides/integrations/huggingface) automatically tracks important metrics during the course of training
    - Also track the HF checkpoints as artifacts and register them in the model registry!
    - Change the number of steps to 200+ for real results!
    """)
    return


@app.cell
def _(model_1, os, tokenizer):
    import transformers
    from datasets import load_dataset
    import wandb
    project_name = 'llm-finetuning'
    entity = 'wandb'  #@param
    os.environ['WANDB_LOG_MODEL'] = 'checkpoint'  #@param
    wandb.init(project=project_name, entity=entity, job_type='training')
    data = load_dataset('Abirate/english_quotes')
    data = data.map(lambda samples: tokenizer(samples['quote']), batched=True)
    trainer = transformers.Trainer(model=model_1, train_dataset=data['train'], args=transformers.TrainingArguments(per_device_train_batch_size=4, gradient_accumulation_steps=4, report_to='wandb', warmup_steps=5, max_steps=25, learning_rate=0.0002, fp16=True, logging_steps=1, save_steps=5, output_dir='outputs'), data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False))
    model_1.config.use_cache = False
    trainer.train()
    wandb.finish()  # silence the warnings. Please re-enable for inference!
    return entity, project_name, wandb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Adding Model Weights to W&B Model Registry
    - Here we get our best checkpoint from the finetuning run and register it as our best model
    """)
    return


@app.cell
def _(entity, project_name, wandb):
    last_run_id = 'zz0lxkc8'  #@param
    wandb.init(project=project_name, entity=entity, job_type='registering_best_model')
    _best_model = wandb.use_artifact(f'{entity}/{project_name}/checkpoint-{last_run_id}:latest')
    registered_model_name = 'OPT-125M-english'  #@param {type: "string"}
    wandb.run.link_artifact(_best_model, f'{entity}/model-registry/{registered_model_name}', aliases=['staging'])
    wandb.finish()
    return (registered_model_name,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Consuming Model From Registry and Quantizing using ctranslate2
    - LLMs are typically too large to run in full-precision on even decent hardware.
    - You can quantize the model to run it more efficiently with minimal loss in accuracy.
       - CTranslate2 is a great first pass at quantization but doesn't do "smart" quantization. It just converts all weights to half precision.
       - Checkout out GPTQ and AutoGPTQ for SOTA quantization at scale
    """)
    return


@app.cell
def _(entity, project_name, registered_model_name, wandb):
    # Pull model from the registry
    wandb.init(project=project_name, entity=entity, job_type='ctranslate2')
    _best_model = wandb.use_artifact(f'{entity}/model-registry/{registered_model_name}:latest')
    _best_model.download(root=f'model-registry/{registered_model_name}:latest')
    wandb.finish()
    return


@app.cell
def _(AutoModelForCausalLM, AutoTokenizer, os, registered_model_name):
    from peft import PeftModel, PeftConfig

    def convert_qlora2ct2(adapter_path=f'model-registry/{registered_model_name}:latest',
                          full_model_path="opt125m-finetuned",
                          offload_path="opt125m-offload",
                          ct2_path="opt125m-finetuned-ct2",
                          quantization="int8"):


        peft_model_id = adapter_path
        peftconfig = PeftConfig.from_pretrained(peft_model_id)

        model = AutoModelForCausalLM.from_pretrained(
          "facebook/opt-125m",
          offload_folder  = offload_path,
          device_map='auto',
        )

        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

        model = PeftModel.from_pretrained(model, peft_model_id)

        print("Peft model loaded")

        merged_model = model.merge_and_unload()

        merged_model.save_pretrained(full_model_path)
        tokenizer.save_pretrained(full_model_path)

        if quantization == False:
            os.system(f"ct2-transformers-converter --model {full_model_path} --output_dir {ct2_path} --force")
        else:
            os.system(f"ct2-transformers-converter --model {full_model_path} --output_dir {ct2_path} --quantization {quantization} --force")
        print("Convert successfully")

    return (convert_qlora2ct2,)


@app.cell
def _(convert_qlora2ct2, registered_model_name):
    convert_qlora2ct2(adapter_path=f'model-registry/{registered_model_name}:latest')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run Inference Using Quantized CTranslate2 Model
    - Record the results in a W&B Table!
    """)
    return


@app.cell
def _(entity, project_name, tokenizer, wandb):
    import ctranslate2


    run = wandb.init(project=project_name, entity=entity, job_type="inference")
    generator = ctranslate2.Generator("opt125m-finetuned-ct2")

    prompts = ["Hey, are you conscious? Can you talk to me?",
               "What is machine learning?",
               "What is W&B?"]


    wandb_table = wandb.Table(columns=['prompt', 'completion'])
    for prompt in prompts:
      start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(prompt))
      results = generator.generate_batch([start_tokens], max_length=30)
      output = tokenizer.decode(results[0].sequences_ids[0])
      wandb_table.add_data(prompt, output)

    wandb.log({"inference_table": wandb_table})
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
