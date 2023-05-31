## Optuna sweep scheduler

Note: This example assumes familiarity with launch setup and creating custom launch scheduler jobs. Information about launch setup can be found [here](https://docs.wandb.ai/guides/launch/quickstart) and in the "quickstart" example. For a simple example of a custom scheduler job, check out the "custom-scheduler" example in this repo. 

### What is Optuna? 

[Optuna](https://optuna.org/) is an open-source hyperparameter tuning [library](https://optuna.readthedocs.io/en/stable/) that exposes significant flexibility to sampling, pruning, and parameter space creation.  

Using sweeps on launch, many of these features can be used to schedule wandb sweeps. To do so, use the `wandb/jobs/job-OptunaScheduler` job, or create your own using the `optuna_scheduler.py` file found in the `wandb/launch-jobs` repo [here](https://github.com/wandb/launch-jobs/jobs/sweep_schedulers/optuna_scheduler.py).

### Key Optuna Features: 

1. Samplers

There are a variety of samplers that can be used to pick hyperparameters from a given search space, found [here](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html). They can be configured to work with the `wandb/jobs/job-OptunaScheduler` job by defining specific settings in the sweep config. 

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
# optuna-config-basic.yaml
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

To use a pythonic search space in the OptunaScheduler, there are two methods. First is to create an artifact (of `type="optuna"`) with a python file containing a function exactly: `def objective(trial)` which will be loaded up and used as the sweep hyperparameter space. The python filename inside the artifact can be specified in the sweep config as: `optuna_source_filename`, or otherwise will assumed to be named: `optuna_wandb.py`. The second is to include that file in the log_code step when creating a code artifact scheduler job. Both methods require setting a filepath parameter in the sweep config, which the Optuna scheduler will use to identify which file contains the `objective` function.

```yaml
...
scheduler:
   settings:
      optuna_source: wandb_optuna.py  # or artifact full path
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