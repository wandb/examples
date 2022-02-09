### Minimal W&B + Hydra example

Before we begin, don't forget to login to your [wandb](wandb.ai) account

```bash
$ wandb login
```

Note: Running this code creates a project called `example-hydra` in your default wandb entity. To change the entity where the project should be hosted, pass `wandb.setup.entity=<entity_name>` along with all the commands shown below. For example:
```bash
$ python3 main.py ... wandb.setup.entity=<entity_name>
```

In case of W&B sweeps, add `wandb.setup.entity=<entity_name>` to the last line of `command:` section of [`wandb_sweep.yaml`](wandb_sweep.yaml).

Example:
```
...
command:
  - ${env}
  - ${interpreter}
  - ${args_no_hyphens}
  - wandb.setup.entity=<entity_name>
```


### Usage
- Running with [`default`](configs/defaults.yaml) config:
```bash
$ python3 main.py
```
- Running in debugging mode:
```
$ python3 main.py logging=debug
```

- Hydra multirun on different experimental configurations as defined in [`experiment/`](configs/experiment):
```bash
$ python3 main.py -m experiment=expt001,expt002
```

- Hydra multirun with Ray launcher:
```bash
$ python3 main.py -m experiment="glob(*)" launcher=hydra/ray +hydra=with_ray
```

- Running [W&B Sweeps](https://wandb.com/sweeps)
  - Initiate WandB sweep as: `$ wandb sweep wandb_sweep.yaml`

  - Run Agent
  Creating a sweep returns a wandb agent command like:
  ![Screenshot showing result returned by wandb sweep command](https://user-images.githubusercontent.com/13994201/153241187-dfa308b6-c52e-4f0a-9f4d-f47b356b1088.png)
  - Next invoke the `wandb agent path/to/sweep` command provided in the output of the previous command.
