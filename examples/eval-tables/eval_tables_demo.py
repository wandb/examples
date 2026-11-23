"""Log W&B EvalTables across several runs and epochs.

Trains a small MNIST classifier with a few configs. After each epoch, each run
logs an EvalTable of predictions on the same validation images, so the Eval
Tables panel can match rows across steps and runs.
"""

import argparse

import torch
import wandb
from torch.utils.data import DataLoader

from mnist import build_model, evaluate, load_data, train_epoch

INPUT_COLUMNS = ["image", "label"]
OUTPUT_COLUMNS = ["pred", "confidence"]
SCORE_COLUMNS = ["correct", "loss"]

RUN_CONFIGS = [
    {"name": "small", "hidden_size": 16, "lr": 1e-3},
    {"name": "wide", "hidden_size": 128, "lr": 1e-3},
    {"name": "wide-fast", "hidden_size": 128, "lr": 1e-2},
]


def to_uint8(image):
    # Identical pixels give identical media hashes, which rows match on.
    return (image[0] * 255).round().to(torch.uint8).numpy()


def build_eval_table(images, labels, preds, confidence, losses):
    rows = [
        [wandb.Image(to_uint8(image)), label, pred, conf, pred == label, loss]
        for image, label, pred, conf, loss in zip(
            images,
            labels.tolist(),
            preds.tolist(),
            confidence.tolist(),
            losses.tolist(),
        )
    ]
    return wandb.EvalTable(
        columns=[*INPUT_COLUMNS, *OUTPUT_COLUMNS, *SCORE_COLUMNS],
        data=rows,
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
        score_columns=SCORE_COLUMNS,
    )


def run_config(config, args, train_ds, val_images, val_labels):
    torch.manual_seed(args.seed)
    model = build_model(config["hidden_size"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    with wandb.init(
        project=args.project,
        entity=args.entity,
        name=config["name"],
        config={**config, "epochs": args.epochs, "val_rows": args.val_rows},
    ) as run:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, loader, optimizer)
            preds, confidence, losses = evaluate(model, val_images, val_labels)
            run.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": losses.mean().item(),
                    "val_accuracy": (preds == val_labels).float().mean().item(),
                    "val_predictions": build_eval_table(
                        val_images, val_labels, preds, confidence, losses
                    ),
                }
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="eval-tables-demo")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--val-rows", type=int, default=64)
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=".")
    args = parser.parse_args()

    train_ds, val_images, val_labels = load_data(
        args.data_dir, args.train_size, args.val_rows
    )
    for config in RUN_CONFIGS:
        run_config(config, args, train_ds, val_images, val_labels)


if __name__ == "__main__":
    main()
