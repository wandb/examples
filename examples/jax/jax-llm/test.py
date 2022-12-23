from transformers import FlaxRobertaModel, RobertaTokenizerFast
from datasets import load_dataset
import jax

dataset = load_dataset(
    "oscar", "unshuffled_deduplicated_en", split="train", streaming=True
)

dummy_input = next(iter(dataset))["text"]

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
input_ids = tokenizer(dummy_input, return_tensors="np").input_ids[:, :10]

model = FlaxRobertaModel.from_pretrained("julien-c/dummy-unknown")

# run a forward pass, should return an object `FlaxBaseModelOutputWithPooling`
model(input_ids)
