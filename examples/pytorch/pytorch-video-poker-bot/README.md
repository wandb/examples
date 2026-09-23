# Jacks or Better video poker bot

Train a small PyTorch network to play **9/6 Jacks or Better**, and log
training + evaluation to [Weights & Biases](https://wandb.ai).

## Video poker, briefly

You are dealt five cards, choose which to hold (32 possible hold patterns),
draw replacements for the rest, and get paid from a fixed paytable.

**Jacks or Better** pays for a pair of Jacks or better, two pair, and the
usual poker hands above that. This example uses the common **9/6** paytable
(full house pays 9×, flush pays 6×, with a max-coin royal bonus).

Optimal play returns about **99.5%** of money wagered. The network here is
trained to imitate an exact expected-value calculator, so it can get close
to that strategy chart.

## Setup

```bash
cd examples/pytorch/pytorch-video-poker-bot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
wandb login
```

Then generate the training dataset (1M random deals labeled with exact hold
EVs, deterministic with seed 42). It takes a couple of minutes:

```bash
python generate_dataset.py data/hands.npz
```

## What to run

| Command | Purpose |
| --- | --- |
| `python generate_dataset.py PATH` | Build the EV-labeled dataset at `PATH` (run first) |
| `python train.py --dataset PATH` | Train on that dataset and log to W&B |
| `python evaluate.py` | Greedy rollout from `checkpoints/hold-network.pt` |

```bash
python generate_dataset.py data/hands.npz
python train.py --dataset data/hands.npz
python evaluate.py --checkpoint checkpoints/hold-network.pt
```

`train.py` is the W&B demo surface: one `wandb.init` context, `run.log` each
epoch, then finish on exit. Useful flag: `--epochs`. For a quicker run, build a
smaller dataset with `python generate_dataset.py data/small.npz --hands 20000`.

## Layout

```
generate_dataset.py   # build an EV-labeled .npz dataset
train.py              # fit HoldNetwork, log metrics + artifact
evaluate.py           # score a checkpoint with greedy play
game.py               # cards, ranks, paytable class, deal/hold/draw
jacks_or_better.py    # JoB classifier + paytable (+ evaluate_hand wrapper)
ev.py                 # exact hold EV calculator (uses JoB classify)
model.py              # encoding, network, train helpers, play/checkpoint
data/                 # dataset loaders (+ generated .npz)
```

## How training works

1. An exact EV calculator labels every hold pattern for each starting hand.
2. A small MLP (`85 → 256 → 256 → 32`) regresses those targets.
3. At play time the network picks the highest-scoring hold and draws.
