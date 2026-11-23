### Minimal W&B + Hydra example

Before we begin, don't forget to login to your [wandb](https://wandb.ai) account

```bash
$ wandb login
```

Note: Running this code creates a project called `example-hydra` in your default wandb entity. To change the entity where the project should be hosted, pass `wandb.setup.entity=<entity_name>` to any of the `python3 main.py` commands shown below. For example:
```bash
$ python3 main.py ... wandb.setup.entity=<entity_name>
```

In case of W&B sweeps, `wandb.setup.entity` and `wandb.setup.project` are ignored: every sweep run is logged to the entity and project of the sweep. [`wandb_sweep.yaml`](wandb_sweep.yaml) sets `project: example-hydra`, so sweep runs land in the same project as above (unless you pass `--project` or set `WANDB_PROJECT` when you create the sweep). To host the sweep in a different entity, pass `--entity` (or set `WANDB_ENTITY`) when you create the sweep:
```bash
$ wandb sweep --entity <entity_name> wandb_sweep.yaml
```


### Usage
- Running with [`default`](configs/defaults.yaml) config:
```bash
$ python3 main.py
```
- Running in debugging mode:
```bash
$ python3 main.py logging=debug
```

- Hydra multirun on different experimental configurations as defined in [`experiment/`](configs/experiment):
```bash
$ python3 main.py -m experiment=expt001,expt002
```

- Hydra multirun with the [Ray launcher plugin](https://hydra.cc/docs/plugins/ray_launcher/). The plugin is not included in `requirements.txt`, so install it first:
```bash
$ pip install hydra-ray-launcher
$ python3 main.py -m experiment="glob(*)" hydra/launcher=ray +hydra=with_ray
```
  [`configs/hydra/with_ray.yaml`](configs/hydra/with_ray.yaml) requests 2 GPUs and 16 CPUs for each job, and Ray does not start a job until a node with those resources is free. On a machine with fewer resources, lower the request:
```bash
$ python3 main.py -m experiment="glob(*)" hydra/launcher=ray +hydra=with_ray hydra.launcher.ray.remote.num_gpus=0 hydra.launcher.ray.remote.num_cpus=4
```

- Running [W&B Sweeps](https://wandb.com/sweeps)
  - Initiate WandB sweep as: `$ wandb sweep wandb_sweep.yaml`

  - Run Agent
  Creating a sweep returns a wandb agent command like:
  ![Screenshot showing result returned by wandb sweep command](https://user-images.githubusercontent.com/13994201/153241187-dfa308b6-c52e-4f0a-9f4d-f47b356b1088.png)
  - Next invoke the `wandb agent path/to/sweep` command provided in the output of the previous command.
