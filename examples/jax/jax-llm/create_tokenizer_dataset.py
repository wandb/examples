from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer
import datasets
import pandas as pd

filename = "some_proteins.fasta"


def batch_iterator(batch_size=2, filename="some_proteins.fasta"):
    with open(filename) as f:
        batch = []
        for line in f:
            if line[0] != ">":  # skip non-protein lines
                batch.append(line)
            if len(batch) == batch_size:
                yield (batch)
                batch = []
        if batch != []:
            yield (batch)


tokenizer = SentencePieceBPETokenizer()
tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=100,
    min_frequency=1,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

tokenizer.save("proteins-tmp")
tokenizer = PreTrainedTokenizerFast(tokenizer_file="proteins-tmp")
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", 2),
    ("<s>", 0),
)
tokenizer.mask_token = "<mask>"
tokenizer.cls_token = "</s>"
tokenizer.sep_token = "<s>"
tokenizer.pad_token = "<pad>"
tokenizer.unk_token = "<unk>"

b = batch_iterator()
lines = []
for x in b:
    for y in x:
        lines.append(y)

df = pd.DataFrame(lines, columns=["text"])
ds = datasets.Dataset.from_pandas(df).train_test_split()
ds["validation"] = ds["test"]
ds.save_to_disk("ds")
tokenizer.save_pretrained("proteins-base")
