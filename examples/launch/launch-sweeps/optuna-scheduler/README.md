## Optuna sweep scheduler

Note: This example assumes familiarity with launch setup and creating jobs. Information about launch setup can be found [here](https://docs.wandb.ai/guides/launch/quickstart) and in the "quickstart" example.

### What is Optuna? 

[Optuna](https://optuna.org/) is an open-source hyperparameter tuning [library](https://optuna.readthedocs.io/en/stable/) that exposes significant flexibility to sampling, pruning, and parameter space creation.  

Using sweeps on launch, many of these features can be used to schedule wandb sweeps. To do so, use the `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` job, or create your own using the `optuna_scheduler.py` file found in the `wandb/launch-jobs` repo [here](https://github.com/wandb/launch-jobs/jobs/sweep_schedulers/optuna_scheduler.py).


### Run a basic example:

Run a simple Optuna scheduler using out-of-the-box image-sourced jobs, using the command:

```bash
wandb launch-sweep optuna_config_basic.yaml -q <container queue> -p <project> -e <entity>
```

Confirm there is a launch agent polling on that queue (if not, start one using [this guide](https://docs.wandb.ai/guides/launch/run-agent)).

The `optuna_config_basic.yaml` file configures a basic sweep using an Optuna [PercentilePruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html) and sweeps over one parameter: `param1`, shown below.

```yaml
# optuna_config_basic.yaml
description: A basic configuration for an Optuna scheduler
# a basic training job to run
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
run_cap: 5
metric:
  name: val_acc
  goal: maximize

scheduler:
  job: wandb/sweep-jobs/job-optuna-sweep-scheduler:latest
  resource: local-container  # required for scheduler jobs sourced from images
  num_workers: 2  # number of concurrent runs
  settings:
    pruner:
      type: PercentilePruner
      args:
        percentile: 25.0  # kill 75% of runs
        n_warmup_steps: 10  # pruning disabled for first x steps

parameters:
  param1:
    min: 0
    max: 10
```



### Key Optuna Features: 

1. Samplers

There are a variety of samplers that can be used to pick hyperparameters from a given search space, found [here](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html). They can be configured to work with the `wandb/sweep-jobs/job-optuna-sweep-scheduler:latest` job by defining specific settings in the sweep config. 

For example: 

```yaml
# optuna-config-basic.yaml
...
scheduler:
   settings:
      sampler:
        # https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.CmaEsSampler.html
        type: CmaEsSampler
        args:
           seed: 42
           n_startup_trials: 10
...
```

2. Pruners

Identical to the samplers above, pruners, found [here](https://optuna.readthedocs.io/en/stable/reference/pruners.html) can be instrumented. 

Example: 

```yaml
# optuna_config_basic.yaml
...
scheduler:
   settings:
      pruner:
        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.PercentilePruner.html
        type: PercentilePruner
        args:
           percentile: 25.0  # kill 75% of runs
           n_warmup_steps: 10  # pruning disabled for first x steps
...
```

3. Pythonic search spaces

Rather than defining the parameters to sweep over in yaml or json format, Optuna allows for defining the search space in a python file. For example, here is a very basic yaml hyperparameter configuration, and the subsequent python format. 

```yaml
...
parameters:
   epochs:
      values: [50, 100, 200]
   learning_rate:
      min: 0.001
      max: 0.1

```

To convert this into python, we use the Optuna `trial` object, which use to request values in a distribution from. This function is ultimately evaluated inside the OptunaScheduler job and converted into a config, which is then used to launch runs. Example python conversion to above: 

```python

def objective(trial):
    epochs = trial.suggest_categorical('epochs', [50, 100, 200])
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)

    # because we aren't actually training in this function, just creating 
    #    our hyperparameter space, return -1
    return -1
```

Although more verbose, this method of constructing hyperparameter spaces comes with one huge advantage, conditional logic. For example, if we wanted to condition our batch size on the database, we might construct a pythonic search space like so: 

```python

def objective(trial):
    database = trial.suggest_categorical('database', ['small', 'medium', 'large'])
    
    if database in ['small', 'large']:
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

        # maybe test randomization when the batch_size is small
        randomize = trial.suggest_categorical('randomize', [True, False])
    else:
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])

    return -1
```

To use a pythonic search space in the OptunaScheduler, there are two methods:
1. Create an artifact (of `type="optuna"`) with a python file containing a function exactly: `def objective(trial)` which will be loaded up and used as the sweep hyperparameter space. Use `settings.optuna_source` to set the path to the artifact. The python file inside the artifact can be specified in the sweep config with: `settings.optuna_source_filename`, or otherwise will assumed to be named: `optuna_wandb.py`. 
```yaml
# using an artifact
...
scheduler:
   settings:
      optuna_source: <entity>/<project>/<job name>:<alias>
      optuna_source_filename: optuna_wandb.py
...
```

2. Include that file in the log_code step when creating a code artifact scheduler job, or in the container when building a docker image. If using just a file in the scheduler context (not an artifact), just use `settings.optuna_source` to direct the scheduler toward the file.


```yaml
# using a file
...
scheduler:
   settings:
      optuna_source: wandb_optuna.py
...
```


In this same file, custom samplers and pruners can be defined in specially named functions. As long as they work in an Optuna study, they will be loaded and plugged into the Optuna scheduler instead of the defaults/config settings. For example, we could add the following functions to our `optuna_wandb.py` file to include all 3 special objects:

```python
# optuna_wandb.py

def objective(trial):
    database = trial.suggest_categorical('database', ['small', 'medium', 'large'])
    
    if database in ['small', 'large']:
        batch_size = trial.suggest_int('batch_size', 12, 64)

        # maybe test offset when the batch_size is small
        random_offset = trial.suggest_int('random_offset', 0, 10)
    else:
        batch_size = trial.suggest_int('batch_size', 64, 256)

    return -1


def sampler():
    return optuna.samplers.NSGAIISampler(
        population_size=100,
        crossover_prob=0.2,
        seed=1000000,
    )


def pruner():
    return optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(), patience=1
    )
```