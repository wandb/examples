This is a getting started guide for using W&B (Weights & Biases) Launch, a system designed to help manage and launch machine learning jobs. The guide outlines the process to create a job, set up a queue, and
activate a launch agent. It assumes that the reader is familiar with concepts like ML Practitioner, MLOps, and machine learning in general.

In summary, the guide covers these main steps:

1. Creating a W&B job using wandb.init(), wandb.log(), and run.log_code().
2. Adding the created job to a queue for execution. In this case, creating a "starter queue" for local testing and demonstration purposes.
3. Activating a launch agent to run the queued jobs.

Additionally, the guide provides instructions on how to view the status of your queue and the fine-grained job details.