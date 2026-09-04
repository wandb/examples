# /// script
# dependencies = ["wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Set_Alerts_with_W_&_B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{alerts-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />
    <!--- @wandbcode{alerts-colab} -->

    # Using `wandb.alert()` to Send Alert Messages

    Use W&B Alerts to send yourself a Slack message or email when something happens in your Python script. Follow the steps below to send your first Alert. See the [Alerts docs](https://docs.wandb.com/app/features/alerts) for a more detailed description.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Turn on Alerts in your W&B User Settings
    - Go to your **[User Settings](https://wandb.ai/settings)**
    - Turn on **Scriptable Alerts**
    - Select whether you'd like to get Alerts via email or Slack
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Install the W&B Library and Login
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb !pip install wandb -qqq
    return


@app.cell
def _():
    # Log in to your W&B account
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Launch a script that triggers alerts
    """)
    return


@app.cell
def _(wandb):
    import random
    from wandb import AlertLevel

    # Initialize a new run in Weights & Biases
    wandb.init(project="test_alerts",
                     config={
                         "threshold": 0.3, # The minimum acceptable accuracy
                         "max_steps": 1000, # The max number of steps for this run
                     })
    config = wandb.config

    # Simulating a model training loop
    for training_step in range(config.max_steps):

      # Generate a random number for accuracy
      accuracy = round(random.random() + random.random(), 3)
      wandb.log({"Accuracy": accuracy})

      # If the accuracy is below the threshold, fire an alert and stop the run
      if accuracy <= config.threshold:
        wandb.alert(
          title='Low Accuracy',
          text=f'Accuracy {accuracy} at step {training_step} is below the acceptable theshold',
          level=AlertLevel.WARN,
          wait_duration=5
        )
        print(f"Script stopped as accuracy is below threshold, {accuracy} vs {config.threshold}")
        break

    # Mark the run as finished (useful in Jupyter notebooks)
    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Check your Slack or email

    Check your Slack or emails for the alert message. If you didn't receive any, check your [Settings](https://wandb.ai/settings) to make sure you've got emails or Slack turned on for **Scriptable Alerts**. More details in the
    [Alerts docs](https://docs.wandb.com/app/features/alerts).
    """)
    return


if __name__ == "__main__":
    app.run()
