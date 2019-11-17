# Run grouping and run resuming example

This example showcases wandb's run grouping and resuming features.

The example has two job types, a `train` job (in `train.py`) and an
`eval` job (in `eval.py`). The eval job is resumable.

Run `./run_experiment.sh to launch the runs that comprise a single experiment.
They will be configured as a group that can be viewed in the W&B UI.

Project page example: https://app.wandb.ai/shawn/group-example

Click on a group name in the sidebar to visit the dedicated page for that group.

Group page example: https://app.wandb.ai/shawn/group-example/groups/experiment-experiment3

## Overview

### Run grouping

Run grouping allows you to associate a set of runs with a single
named "group". The W&B UI provides dedicated pages for run
groups, and automatically configures the project page for grouping,
when grouped runs are present. This is useful for distributed
training, when a set of runs is logically considered a single
run.

Runs within a group can have different "job types". The UI will
separate runs with different job types into separate subgroups.

To associate a run with a group and jobtype, pass the `group`
and `job_type` parameters to wandb.init().

`group` should be a
string that is unique to the group. Any other runs that share
the same group string will be associated with the group.

`job_type` can be any string. Runs that share a job type within
a group will be grouped together as subgroups.


### Run resuming

wandb allows you to stop and resume runs. Resuming a run will
concatenate logged metrics, logs and system data to the run's
previous values.

This is most commonly used when using pre-emptible cloud resources,
or for asynchronous evaluation tasks that happen after every epoch (for
example). Rather than having one evaluation run per epoch, you can
create a single run across all epochs, even though the job itself
runs many times.

To use run resuming, pass a unique run ID to all executions that you
want to save to the same run, with the `WANDB_RUN_ID` environment variable.
You must also pass `WANDB_RESUME=allow`.

## Docs

To learn more, visit our official documentation for [grouping](https://docs.wandb.com/library/advanced/grouping) and [resuming](https://docs.wandb.com/library/advanced/resuming).
