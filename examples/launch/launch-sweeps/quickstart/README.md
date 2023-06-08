
## Sweeps on launch quickstart

Sweeps-on-launch leverages launch to supercharge the wandb hyperparameter tuning with massive parallelism. To do so, please complete the prerequisite launch-setup steps, which can be found [here](https://docs.wandb.ai/guides/launch/quickstart) and are provided in brief below.

### Launch setup

1. [Create a queue](https://docs.wandb.ai/guides/launch/create-queue). In the UI, navigate to the launch application, either in a personal entity or team. Then create a queue in the resource of your choosing. For advanced resources like Kubernetes, Sagemaker and GCP Vertex, additional setup might be required. For local-container queues, no resource configuration is mandatory. 
2. [Launch an agent](https://docs.wandb.ai/guides/launch/run-agent) onto the queue. Depending on the resource, this step might be slightly different. For a local container queue, the agent command should appear in the UI, and look like: `wandb launch-agent --queue <queue> --entity <entity> --max-jobs=8`. One important note is to consider the maximum number of jobs the agent can run using the `--max-jobs` flag. This will allow multiple concurrent runs during the sweep.

### Sweeps setup

A launch-sweep configuration file should look very similar to a normal sweep configuration file, with a two changes. 

1. The `program` execution path must be replaced with a `job`. For this quickstart, lets use a publicly available job, like: `'wandb/jobs/Example Train Job:latest'`. This can be replaced with any training job that is available in the entity/project where the sweep will be run. More information about job creation can be found [here](https://docs.wandb.ai/guides/launch/create-job). This enables our training run to be executed anywhere, no longer tied to a local training script.
2. An Optional `scheduler` key can be provided to the configuration, specifying scheduler-specific parameters for the sweep, like concurrency of runs. 

Example configuration file: 

```yaml
# launch_sweep_config.yaml
description: sweeps on launch quickstart
method: grid
run_cap: 50

# a basic MNIST training job
job: wandb/sweep-jobs/job-fashion-MNIST-train:latest
metric:
  name: epoch/val_loss
  goal: minimize

# some parameters to tune
parameters:
  learning_rate:
    values: [0, 0.0001, 0.001, 0.01, 0.1, 1]
  epochs:
    max: 20
    min: 0
    distribution: int_uniform

# Optional Scheduler Params:
scheduler:
   num_workers: 8  # concurrent sweep runs
```

3. Finally, launch your sweep with the CLI command: 

```bash
wandb launch-sweep <path/to/config> --queue <queue> --project <project> --entity <entity>
```

The sweep should be automatically picked up by your launch agent and begin scheduling runs into the specified project.
