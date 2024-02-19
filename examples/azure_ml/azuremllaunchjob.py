# !pip install transformers datasets "transformers[torch]" wandb
# !pip install onnx onnxruntime
# !pip install tensorflow
# !pip install -U accelerate

import os
os.environ["WANDB_PROJECT"] = "azure_ml_test"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import wandb
import numpy as np
import pandas as pd
from transformers.integrations import WandbCallback
from pathlib import Path
from transformers import TFAutoModelForSequenceClassification
from transformers.convert_graph_to_onnx import convert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"

# Initialize wandb
wandb.init(project="azure_ml_test")

# Load a dataset and a tokenizer
dataset = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding="max_length", max_length=128)  # You can adjust max_length as needed

encoded_dataset = dataset.map(preprocess_function, batched=True)

# Load a model and specify the number of labels
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# Load the metric
metric = load_metric("glue", "mrpc")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Depending on the task, you might need to modify the prediction or label format
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,  # Load the best model at the end based on eval_accuracy
    metric_for_best_model="eval_accuracy",
    logging_strategy="steps",
    logging_steps=1,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    report_to="wandb",  # Ensure we report to Weights & Biases
)
# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    compute_metrics=compute_metrics,
)

def decode_predictions(predictions, labels, encoded_dataset, label_map):
    pred_labels = np.argmax(predictions, axis=1)
    pred_label_str = [label_map[label] for label in pred_labels]
    true_label_str = [label_map[label] for label in labels]
    sentence1 = encoded_dataset['sentence1']
    sentence2 = encoded_dataset['sentence2']
    return {
        'Sentence 1': sentence1,
        'Sentence 2': sentence2,
        'Prediction': pred_label_str,
        'Ground Truth': true_label_str
    }

class WandbPredictionLoggingCallback(WandbCallback):
    def __init__(self, trainer, tokenizer, val_dataset, label_map, num_samples=100, freq=1):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.val_dataset = val_dataset.select(range(num_samples))
        self.label_map = label_map
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        if state.epoch % self.freq == 0:
            predictions, labels, _ = self.trainer.predict(self.val_dataset)
            decoded_predictions = decode_predictions(predictions, labels, self.val_dataset, self.label_map)
            predictions_df = pd.DataFrame(decoded_predictions)
            predictions_df["epoch"] = state.epoch
            predictions_table = wandb.Table(dataframe=predictions_df)
            wandb.log({"predictions": predictions_table})

# Instantiate the WandbPredictionLoggingCallback
prediction_logging_callback = WandbPredictionLoggingCallback(
    trainer=trainer,
    tokenizer=tokenizer,
    val_dataset=encoded_dataset["validation"],
    label_map={0: 'not equivalent', 1: 'equivalent'},
    num_samples=10,
    freq=1,
)

# Add the callback to the trainer
trainer.add_callback(prediction_logging_callback)

# Train the model
trainer.train()
# Perform the prediction
predictions, labels, _ = trainer.predict(encoded_dataset["validation"])

# Decode the predictions
decoded_predictions = decode_predictions(predictions, labels, encoded_dataset["validation"], {0: 'not equivalent', 1: 'equivalent'})

# Create the pandas DataFrame
df = pd.DataFrame(decoded_predictions)

wandb.log({"final_predictions": wandb.Table(dataframe=df)})

# Define the base path using pathlib for better path management
base_path = Path(__file__).parent

# Save the model and tokenizer in PyTorch format
hf_model_dir = base_path / "hf_model"
hf_model_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(hf_model_dir)
tokenizer.save_pretrained(hf_model_dir)
wandb.log_model(str(hf_model_dir), name="hf_model")

# Alternative PyTorch format
torch_model_dir = base_path / "torch_model"
torch_model_dir.mkdir(parents=True, exist_ok=True)
torch_model_path = torch_model_dir / "model.pt"
torch.save(model.state_dict(), torch_model_path)
tokenizer.save_pretrained(torch_model_dir)
wandb.log_model(str(torch_model_dir), name="pt_model")

# Export to ONNX
onnx_model_dir = base_path / "onnx_model"
onnx_model_dir.mkdir(parents=True, exist_ok=True)
onnx_model_path = onnx_model_dir / "model.onnx"
convert(framework="pt", model=str(hf_model_dir), output=onnx_model_path, opset=11)
wandb.log_model(str(onnx_model_dir), name="onnx_model")

# For TensorFlow format, we need to convert the PyTorch model to TensorFlow
tensorflow_model_dir = base_path / "tf_model"
tensorflow_model_dir.mkdir(parents=True, exist_ok=True)
tf_model = TFAutoModelForSequenceClassification.from_pretrained(str(hf_model_dir), from_pt=True)
tf_model.save_pretrained(tensorflow_model_dir, saved_model=True)
tokenizer.save_pretrained(tensorflow_model_dir)
wandb.log_model(str(tensorflow_model_dir), name="tf_model")

wandb.finish()
