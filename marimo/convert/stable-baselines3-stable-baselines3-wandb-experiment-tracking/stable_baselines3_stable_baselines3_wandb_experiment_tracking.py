# /// script
# dependencies = ["pyvirtualdisplay", "stable-baselines3", "wandb"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/stable_baselines3/Stable_Baselines3_wandb_experiment_tracking.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{sb3-integration} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stable Baselines 3 - Track Experiments with Weights and Biases

    <!--- @wandbcode{sb3-integration, version=webinar} -->

    Github repo: https://github.com/araffin/rl-tutorial-jnrr19

    Stable-Baselines3: https://github.com/DLR-RM/stable-baselines3

    Documentation: https://stable-baselines.readthedocs.io/en/master/

    RL Baselines3 zoo: https://github.com/DLR-RM/rl-baselines3-zoo

    Weights & Biases: https://wandb.ai/site

    Weights & Biases Docs: https://docs.wandb.ai/

    ## Introduction

    [Weights & Biases (W&B)](https://wandb.ai/site) is a tool for machine learning experiment tracking, dataset versioning, and project collaboration.

    <div><img /></div>

    <img src="https://wandb.me/mini-diagram" width="650" alt="Weights & Biases" />

    <div><img /></div>

    In this notebook, you will learn how to track reinforcement learning experiments using W&B. In particular, W&B helps track your experiment configs, metrics, and videos of the agents playing the game. At the end, you should see a run page like https://wandb.ai/wandb/cartpole_test/runs/37ppqzxc

    ## Install Dependencies and Set up Virtual Displays for Video Recordings
    """)
    return


@app.cell
def _(subprocess):
    #! apt install python-opengl xvfb
    subprocess.call(['apt', 'install', 'python-opengl', 'xvfb'])
    # packages added via marimo's package management: pyvirtualdisplay stable_baselines3[extra] wandb !pip install pyvirtualdisplay stable_baselines3[extra] wandb
    from pyvirtualdisplay import Display
    virtual_display = Display(visible=0, size=(1400, 900))
    virtual_display.start()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Track experiments with W&B

    Here is a clean end-to-end example to run. It will prompt you to login in to W&B if you haven't.
    """)
    return


@app.cell
def _():
    import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
    import wandb
    from wandb.integration.sb3 import WandbCallback


    config = {
        "policy_type": "MlpPolicy",
        "total_timesteps": 25000,
        "env_name": "CartPole-v1",
    }
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    def make_env():
        env = gym.make(config["env_name"])
        env = Monitor(env)  # record stats such as returns
        return env


    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )
    run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    After finishing the cell above you should see a dashbaord similar to the gif below:

    ![](https://user-images.githubusercontent.com/5555347/122989248-97b5bd00-d370-11eb-95d6-52d56cfbce19.gif)
    """)
    return


if __name__ == "__main__":
    app.run()
