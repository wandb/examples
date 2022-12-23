from transformers import RobertaConfig

config = RobertaConfig.from_pretrained("roberta-base", vocab_size=100)
config.save_pretrained("./proteins-base")
