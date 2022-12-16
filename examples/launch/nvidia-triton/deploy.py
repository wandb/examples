import wandb
import os
import shutil
import subprocess


settings = wandb.Settings()
settings.update({"enable_job_creation": True})

config = {
    "wandb_model_artifact": "jzhao/examples-examples_keras_keras-cnn-fashion/model-fresh-dream-1:v8",

}

run = wandb.init(entity="jzhao", config=config, project="examples-examples_keras_keras-cnn-fashion", settings=settings)
config = run.config

model_artifact = config["wandb_model_artifact"]
artifact = run.use_artifact(config["wandb_model_artifact"])
artifact_name, artifact_version_str = artifact.name.split(":")
artifact_version = artifact_version_str.strip("v")
path = artifact.download()

# assume the artifact is a tensorflow artifact, create a "model.savedmodel" folder
model_path = os.path.join("/model_repository", artifact_name, artifact_version, "model.savedmodel")
# create the local model repository for Triton
os.makedirs(model_path, exist_ok=True)
# copy over the saved model files to triton
shutil.copytree(path, model_path, dirs_exist_ok=True)


run.finish()

# start the triton server
proc = subprocess.run(["tritonserver", "--model-repository", "/model_repository"])
