import boto3
import os
import sys
import json
import tritonclient.http as httpclient

import wandb


settings = wandb.Settings()
settings.update({"enable_job_creation": True})

config = {
    "triton_url": "localhost:8000",
    "triton_bucket": "wandb-jason-test",
    "triton_model_repository_path": "model_repository",
    "wandb_model_artifact": "jzhao/examples-examples_keras_keras-cnn-fashion/model-fresh-dream-1:v8",

}

run = wandb.init(entity="jzhao", config=config, project="examples-examples_keras_keras-cnn-fashion", settings=settings)
config = run.config

s3_client = boto3.client("s3")
triton_client = httpclient.InferenceServerClient(url=config["triton_url"])


model_artifact = config["wandb_model_artifact"]
artifact = run.use_artifact(config["wandb_model_artifact"])
artifact_name, artifact_version_str = artifact.name.split(":")
# Triton model version numbers must be integer
artifact_version = artifact_version_str.strip("v")

# Triton model API doesn't handle versions very well
for model in triton_client.get_model_repository_index():
    if model.get("name") != artifact_name:
        continue
    if model.get("version", None) is None or model.get("state", None) is None:
        continue
    if model.get("version") == artifact_version and model.get("state") == "READY":
        print(f"Model {artifact_name} is already loaded, skipping deployment")
        sys.exit(0)
            
path = artifact.download()

# copy the content of the model to the remote model repository, assume model repository is in s3 for now
print("Uploading model to Triton model repository...")
remote_path = os.path.join(config["triton_model_repository_path"], artifact_name, artifact_version, "model.savedmodel")

for root, _, files in os.walk(path):
    for f in files:
        full_path = os.path.join(root, f)
        rel_path = os.path.relpath(full_path, path)
        remote_obj_path = os.path.join(remote_path, rel_path)
        print(f"Uploading {rel_path} to {remote_obj_path}")
        s3_client.upload_file(full_path, config["triton_bucket"], remote_obj_path)


print("Finished uploading model to Triton model repository")


# verify model exists and load the model, note we cannot verify version numbers prior to loading
model_exists = False
for model in triton_client.get_model_repository_index():
    if model["name"] == artifact_name:
        model_exists = True

if not model_exists:
    raise Exception(f"Triton: Failed to add model {artifact_name} to repository")

# generate model config policy to load specific version
# see: https://github.com/triton-inference-server/server/issues/4416
version_config = { "version_policy": { "specific": { "versions": [artifact_version]}}}
triton_client.load_model(artifact_name, config=json.dumps(version_config))
if not triton_client.is_model_ready(artifact_name):
    raise Exception(f"Triton: Failed to load model {artifact_name}")

print("Successfully loaded Triton model")

run.log_code()
run.finish()
