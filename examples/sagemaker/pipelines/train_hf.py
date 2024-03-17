from types import SimpleNamespace

import wandb

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

from llm_recipes.data import create_alpaca_prompt, create_alpaca_prompt_with_response
from llm_recipes.utils import freeze, parse_args, LLMSampleCB
from llm_recipes.hf import debug_trainer_data


ALPACA_TOTAL_PACKED_SAMPLES = 11_210 # at seq_len=1024
WANDB_PROJECT = "alpaca_ft"
WANDB_ENTITY = "capecape"
WANDB_TAGS = ["7b", "hf_sft"]

config = SimpleNamespace(
    dataset_at='capecape/alpaca_ft/alpaca_gpt4_splitted:v4',
    model_id = 'meta-llama/Llama-2-7b-hf',
    n_freeze = 24, # how many layers to freeze on the model (llama 7b has 32)
    batch_size = 8, # what my GPU can handle, depends on how many layers are we training
    effective_batch_size = 32, # batch size for gradient accumulation
    gradient_checkpointing = True,
    max_seq_length = 1024,
    num_train_epochs = 3, # we do 3 pasess over the dataset.
    freeze_embed = True,
    use_lora = False,
    lr = 2e-5,
    save_model=True, # save model after training also save to wandb
    # for debug purposes
    max_steps=-1, 
    train=True,
    evaluate=True,
    debug_data=False,
    load_in_4bit=False,
    load_in_8bit=False,
)

def get_alpaca_ds(dataset_at):
    artifact = wandb.use_artifact(dataset_at, type='dataset')
    artifact_dir = artifact.download()
    alpaca_ds = load_dataset("json", data_dir=artifact_dir)
    train_dataset = alpaca_ds["train"]
    eval_dataset = alpaca_ds["test"]
    return train_dataset, eval_dataset

def get_train_args(config, output_dir = "./output/"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        report_to="wandb",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=max(config.batch_size//2, 1),
        bf16=True,
        learning_rate=config.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_steps=config.max_steps,
        num_train_epochs=config.num_train_epochs,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        evaluation_strategy="no",
        # logging strategies
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
    )
    return training_args

def main(config):
    # some sane defaults computations
    config.gradient_accumulation_steps = (1024 // config.max_seq_length) * config.effective_batch_size // config.batch_size
    config.tokens_per_step = config.max_seq_length * config.batch_size * config.gradient_accumulation_steps
    print(f"\nWe are training for {config.max_steps} steps with an effective batch size of {config.effective_batch_size} and a gradient accumulation of {config.gradient_accumulation_steps} steps.")
    print(f"Tokens per step max_seq_len * bs * grad_accum_steps: {config.tokens_per_step}\n")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_cache=False,
        load_in_4bit=(not config.load_in_8bit) and config.load_in_4bit,
        load_in_8bit=(not config.load_in_4bit) and config.load_in_8bit,
    )
    if config.use_lora:
        peft_config = LoraConfig(
                r=64,  # the rank of the LoRA matrices
                lora_alpha=16, # the weight
                lora_dropout=0.1, # dropout to add to the LoRA layers
                bias="none", # add bias to the nn.Linear layers?
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        config.peft_config = peft_config
        config.n_freeze = "all"
    else:
        freeze(model, config.n_freeze, config.freeze_embed)

    # wandb stuff
    wandb.init(project=WANDB_PROJECT, 
               entity=WANDB_ENTITY, 
               job_type="train",
               tags=WANDB_TAGS,
               config=config)
    train_dataset, eval_dataset = get_alpaca_ds(config.dataset_at)
    
    # override whatever train args we may need
    training_args = get_train_args(config)
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        packing=True,
        max_seq_length=config.max_seq_length,
        args=training_args,
        formatting_func=create_alpaca_prompt_with_response,
    )
    if config.debug_data:
        debug_trainer_data(trainer)
        return
    if config.train: 
        trainer.train()
        if config.save_model:
            trainer.save_model(training_args.output_dir)
            print("Saving model as artifact to wandb")
            model_at = wandb.Artifact(
                name = f"{wandb.run.id}_alpaca", 
                type="model",
                description="Model trained on Alpaca GPT4 dataset",
                metadata={"finetuned_from":config.model_id})
            model_at.add_dir(training_args.output_dir)
            wandb.log_artifact(model_at)
    if config.evaluate:    
        _map_func = lambda row: {"text": create_alpaca_prompt(row)}
        test_dataset = eval_dataset.map(_map_func) # no answers
        wandb_callback = LLMSampleCB(trainer, test_dataset, num_samples=10, max_new_tokens=256)
        trainer.add_callback(wandb_callback)
        trainer.evaluate()

if __name__ == "__main__":
    parse_args(config)
    main(config)