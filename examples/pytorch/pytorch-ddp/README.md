# PyTorch DDP

This example uses [`wandb`](https://docs.wandb.com) with
[PyTorch `DistributedDataParallel`](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
to track single-node, multi-GPU training.

We demonstrate two modes of usage:
1. logging from a single process, and
2. logging from all processes.

## Method 1: Log from a single process
**File:** `log-rank0.py`

In this script we track only the rank0 process.
1. Call `wandb.init()` once
2. Call `wandb.log()` only from that process. Never call `wandb` methods from a process that hasn't called `wandb.init()`.
    
Usage: 

```python
python -m torch.distributed.launch \
  --nproc_per_node 2 \
  --nnodes 1 \
  --node_rank 0 \
  log-rank0.py \
    --epochs 10 \
    --batch 512 \
    --entity <ENTITY> \
    --project <PROJECT>
```

## Method 2: Log from all processes

**File:** `log-all.py`

In this script we track all the processes and group them together.
1. Call `wandb.init()` in every process. Use the `group` parameter to group the jobs together into a larger experiment. Use `job_type` if you want to separate out different types of jobs on different machines, such as `rollout` and `eval` workers.
2. Call `wandb.log()` in any process where you want to log metrics. This is ok because all processes have called `wandb.init()`, so they can call other `wandb` functions safely.


    Usage: 
    ```python
    python -m torch.distributed.launch \
      --nproc_per_node 2 \
      --nnodes 1 \
      --node_rank 0 \
      log-all.py \
        --epochs 10 \
        --batch 512 \
        --entity <ENTITY> \
        --project <PROJECT>
    ```
