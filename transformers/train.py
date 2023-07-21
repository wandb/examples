from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import wandb
import os
import ast


os.environ["WANDB_LOG_MODEL"] = "checkpoint"
with wandb.init(entity="gong-demo", project="transformers") as run:
    
    # Load the dataset
    dataset = load_dataset("imdb")
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", fast=True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Binary classification example for IMDB dataset
    
    resume = False
    if os.environ.get("WANDB_RESUME") == "must":
        # check for an existing checkpoint artifact
        my_checkpoint_name = f"checkpoint-{run.id}:latest"
        try:
            my_checkpoint_artifact = run.use_artifact(my_checkpoint_name)
            # Download checkpoint to a folder and return the path
            checkpoint_dir = my_checkpoint_artifact.download()
            resume = True
        except Exception as e:
            print("No previous checkpoint found for run {run.name}")


    train_fraction = 0.02  
    test_fraction = 0.01

    train_dataset = dataset["train"].select(range(int(len(dataset["train"]) * train_fraction)))
    test_dataset = dataset["test"].select(range(int(len(dataset["test"]) * test_fraction)))

    # Tokenize the datasets
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", fast=True)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=50,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        report_to="wandb",
#        use_mps_device=True
    )

    for k, v in wandb.config.items():
        if k in training_args.__dict__.keys():
            config_value = v
            new_type = type(config_value)
            original_type = type(training_args.__dict__[k])
            if original_type != new_type:
                try:
                    if original_type is list or (new_type is str and config_value == "None"):
                        # try eval string into the correct type
                        config_value = ast.literal_eval(v)
                    else:
                        # try to instantiate the type, this will work for int, float, and enums
                        config_value = original_type(v)
                except Exception as e:
                    pass
                if type(config_value) != original_type:
                    print(f"Failed to convert {v} of type {new_type} to original type {original_type}, skipping...")
                    continue
            training_args.__dict__[k] = config_value

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    if resume:
        trainer.train(resume_from_checkpoint=checkpoint_dir)
    else:
        trainer.train()

wandb.finish()
