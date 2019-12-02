# wandb sweeps mpi wrapper scripts

## Description

Demonstrating how to use sweeps with mpi frameworks

## Steps

1) modify outer wrapper `split_command` variable to match how you want to call your training program

By default the script assumes that the script is called `train.py` and that you are not passing any commandline arguments.

2) specify outer wrapper as `program` in the sweep configuration yaml file

```
program:
   wrap_mpi_inner.py
method: grid
parameters:
   ...
```

3) launch your sweep


## Details

These wrapper scripts allow wandb to pass UNIX signals across mpi frameworks.  Signals are used when manually stopping runs from the UI or using the early_terminate sweeps feature.

The outer wrapper catches SIGTERM and uses SIGUSR1 to pass to the inner wrapper.  The inner wrapper catches SIGUSR1 and passes SIGTERM to the user training script.  This is required because mpi frameworks typically do not pass SIGTERM to child processes.

