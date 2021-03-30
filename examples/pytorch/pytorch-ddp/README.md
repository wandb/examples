# PyTorch DDP

This example tracks single-node multi-GPU training with PyTorch DDP.
- `log-rank0.py`: In this script we track only the rank0 process.
    
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
    --project <PROJECT>```
- `log-all.py`: In this script we track all the processes and group them together.

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
    --project <PROJECT>```