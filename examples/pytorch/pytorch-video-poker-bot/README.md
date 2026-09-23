# Train a video poker bot with PyTorch and W&B

This example trains a small PyTorch network to play video poker and tracks
it with [Weights & Biases](https://wandb.ai). The poker is just a stand-in
task. The point is to see a complete W&B run:

1. **`wandb.init()`** starts a run and records its config (hyperparameters
   and dataset size).
2. **`run.log()`** sends metrics every epoch, so you can watch training live.
3. **`run.log_artifact()`** uploads the trained checkpoint as a versioned
   model artifact.
4. **Finish**: `train.py` opens the run with `with wandb.init(...) as run:`,
   so the run finishes automatically when the block exits. Without the
   `with` block, you would call `run.finish()` yourself.

`evaluate.py` then starts a second run (`job_type="evaluation"`). It fetches
the model artifact from W&B with **`run.use_artifact()`**, which links the
two runs, and logs its score every 100 hands as it plays.

## How video poker works

You are dealt five cards. You choose which of them to hold: none, all five,
or any mix in between (32 possible combinations). The cards you don't hold
are replaced with new ones from the deck, and that is your final hand. Its
reward is looked up in the paytable.

This example plays **9/6 Jacks or Better**, betting 5 credits per hand:

| Final hand | Reward (credits) |
| --- | --- |
| Royal flush | 4000 |
| Straight flush | 250 |
| Four of a kind | 125 |
| Full house | 45 |
| Flush | 30 |
| Straight | 20 |
| Three of a kind | 15 |
| Two pair | 10 |
| Pair of jacks or better | 5 |
| Anything else | 0 |

The name "9/6" comes from the full house paying 9 and the flush paying 6 for
each credit bet. With optimal holds, the rewards average about 99.5% of the
credits wagered.

The dataset gives each dealt hand the exact expected reward of all 32 hold
choices. The network learns to predict those values, and at play time it
makes the hold with the highest prediction.

## Setup

```bash
cd examples/pytorch/pytorch-video-poker-bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

## Run it

### 1. Generate the dataset

```bash
python generate_dataset.py --output data/hands.npz
```

This deals 1,000,000 random hands and labels each one. It takes about two
minutes and writes a ~44 MB file. For a quick test, add `--hands 20000`.

### 2. Train

```bash
python train.py \
  --dataset data/hands.npz \
  --checkpoint checkpoints/jacks_or_better_network.pt \
  --artifact-name jacks-or-better-network \
  --project video-poker \
  --run-name train-baseline
```

This takes under a minute on a laptop CPU. After each epoch it saves the
checkpoint if validation improved, and at the end it uploads the checkpoint
to W&B as the model artifact `jacks-or-better-network`. Optional flags:
`--epochs` (default 20), `--lr`, `--batch-size`, `--hidden-size`, `--seed`.

### 3. Evaluate

```bash
python evaluate.py \
  --artifact jacks-or-better-network:latest \
  --project video-poker \
  --run-name eval-baseline
```

This downloads the latest version of the model artifact and plays 100,000
hands with it (change with `--hands`), which takes about 10 seconds. It logs
to a separate run in the same project.

## What you will see in W&B

- **Training run**: config on the Overview tab, and these charts per epoch:
  - `train_loss` and `val_loss`
  - `val_optimal_action_pct`: how often the network makes the optimal hold
  - `val_mean_regret`: the expected reward per credit bet that the network's
    holds give up compared with the optimal holds, on average. Lower is
    better, and the saved checkpoint is the epoch with the lowest value.
  - `val_expected_return_pct`: expected rewards as a percentage of credits
    wagered
- **Evaluation run**: running totals logged every 100 hands, so you can
  watch the score settle as more hands are played: `wagered` (credits bet),
  `payout` (credits rewarded), `profit` (the difference) and `return_pct`
  (rewards as a percentage of credits wagered).
- **Model artifact**: every training run adds a new version of
  `jacks-or-better-network`. Its Lineage tab shows which training run
  produced each version and which evaluation runs used it.

## Files

| File | What it does |
| --- | --- |
| `generate_dataset.py` | Builds the labeled training dataset. |
| `train.py` | Trains the network and logs the run to W&B. |
| `evaluate.py` | Fetches the trained model from W&B, plays hands with it, and logs the score. |
| `dataset.py` | Loads a generated dataset for training. |
| `model.py` | The network, hand encoding, training and validation steps, and checkpoint save/load. |
| `game.py` | Cards, deck, dealing and drawing. |
| `jacks_or_better.py` | Hand rankings and the paytable. |
| `ev.py` | Calculates the exact expected reward of each hold choice. |
