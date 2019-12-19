#!/usr/bin/env python
from wandb.sweeps.config import tune
from wandb.sweeps.config.hyperopt import hp
from wandb.sweeps.config.tune.suggest.hyperopt import HyperOptSearch

tune_config = tune.run(
    "train.py",
    search_alg=HyperOptSearch(
        dict(
            dropout=hp.uniform("dropout", 0.15, 0.4),
            hidden_layer_size=hp.choice("hidden_layer_size", [96, 128, 148]),
            layer_1_size=hp.choice("layer_1_size", [10, 12, 14, 16, 18, 20]),
            layer_2_size=hp.choice("layer_2_size", [24, 28, 32, 36, 40, 44]),
            learn_rate=hp.choice("learn_rate", [0.001, 0.01, 0.003]),
            decay=hp.choice("decay", [1e-5, 1e-6, 1e-7]),
            momentum=hp.choice("momentum", [0.8, 0.9, 0.95])),
        metric="val_loss",
        mode="min"),
    num_samples=10,
    )

tune_config.save("sweep-tune-hyperopt.yaml")
