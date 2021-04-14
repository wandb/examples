# PyTorch DDP

This example uses [`wandb`](https://docs.wandb.com) with
[PyTorch `DistributedDataParallel`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
to track single-node, multi-GPU training.

We demonstrate two modes of usage:
1. logging from a single process, and
2. logging from all processes.

## Method 1: Log from a single process

In this method we track only the rank0 process
-- the "master" process from which the others spawn.
If you're not interested in intra-batch or inter-batch statistics,
this approach can save you some overhead.

To get started, we call `wandb.init()` once in the rank0 process
and then call `wandb.log()` only from that process.

If you use this method, take care that you never call `wandb` methods
from any process that hasn't called `wandb.init()`.

#### Usage:

```python
python -m torch.distributed.launch \
  --nproc_per_node <NUM_GPUS> \
  --nnodes 1 \
  --node_rank 0 \
  log-ddp.py \
    --epochs 10 \
    --batch 512 \
    --entity <ENTITY> \
    --project <PROJECT>
```

## Method 2: Log from all processes

In this method we track all the processes and group them together.

We call `wandb.init()` in _every_ process.
We use the `group` parameter to group the jobs together into a larger experiment.
Use `job_type` if you want to separate out different types of jobs on different machines,
such as `rollout` and `eval` workers.

Here, you can call `wandb.log()` in any process where you want to log metrics,
since all processes have called `wandb.init()`.

#### Usage:

```python
python -m torch.distributed.launch \
  --nproc_per_node <NUM_GPUS> \
  --nnodes 1 \
  --node_rank 0 \
  log-ddp.py \
    --log_all \
    --epochs 10 \
    --batch 512 \
    --entity <ENTITY> \
    --project <PROJECT>
```
