## Use a scheduler job for sweeps

Using sweeps with launch enables superior customizability. The sweep scheduling mechanism can be entirely replaced with a job! To do so, create a sweep scheduler job, or point to a public wandb scheduler job. Examples of what is possible with custom sweep scheduler jobs are available in the [wandb/launch-jobs](https://github.com/wandb/launch-jobs) repo under `jobs/sweep_schedulers`. This example demonstrates how to use the publicly available Wandb Scheduler Job, as well as a process for creating a custom sweep scheduler job. 

### Using the public WandbScheduler job

Use the `launch-sweep` command with the example sweep-config to get started. Remember to push to a queue with an active launch agent.

`wandb launch-sweep scheduler-job-sweep-config.yaml -q <queue> -p <project> -e <entity>`

### Customizing the WandbScheduler job

First, make changes to the `wandb_scheduler.py` file found in this folder (mirrors [wandb/launch-jobs/jobs/sweep_schedulers/wandb_scheduler.py](https://github.com/wandb/launch-jobs/jobs/sweep_schedulers/wandb_scheduler.py))

In this example, we add an additional scheduler setting to manage the verbosity of logging.

```python
...
def _poll(self) -> None:
   _logger.debug(f"_poll. _runs: {self._runs}")
   if self._settings_config.get('verbose_logging'):
      termlog(f"_poll. _runs: {self._runs}")
...
```

Now, to turn this scheduler python file into a job, run `python wandb_scheduler.py --project <project> --entity <entity>`.

This should init a run that only logs the job. The name for this job can be found in the jobs tab in the project, or printed in the python environment.

Insert the scheduler job in the sweep configuration under a scheduler.job key, along with the usual sweep parameters. Example:

```yaml
# custom-job-sweep-config.yaml
...
scheduler:
   job: <entity/project/job-WandbScheduler:latest>
   settings:
      verbose_logging: True
...
```

Then launch the sweep from this config to a queue with an active launch agent.
