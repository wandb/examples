"""
Deploy a model artifact logged to W&B to Nvidia Triton
"""

import json
import os

import boto3
import tritonclient.http as httpclient
import wandb
from google.protobuf import json_format, text_format
from tritonclient.grpc import model_config_pb2


# def config_pbtxt_to_dict(fname):
#     with open(fname) as f:
#         model_config = model_config_pb2.ModelConfig()
#         text_format.Parse(f.read(), model_config)
#         return json_format.MessageToDict(model_config)


def s3_config_pbtxt_to_dict(bucket, pbtxt_path):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(bucket)
    for obj in bucket.objects.all():
        print(obj.key)
        if obj.key == pbtxt_path:
            model_config = model_config_pb2.ModelConfig()
            body = obj.get()["Body"]
            text_format.Parse(body.read(), model_config)
            return json_format.MessageToDict(model_config)
    return {}  # no pbtxt found; do you have to load and unload model for the autogen?


def dict_to_config_pbtxt(d, out_fname):
    with open(out_fname, "w") as f:
        model_config = model_config_pb2.ModelConfig()
        json_format.ParseDict(d, model_config)
        text_format.PrintMessage(model_config, f)


config = {
    "entity": "megatruong",
    "project": "fashion-mnist-keras-triton",
    "artifact_name": "model-sage-feather-1",
    "artifact_version": 1,
    "triton_url": "localhost:8000",
    "triton_bucket": "andrew-triton-bucket",
    "triton_model_repo_path": "models",
    "triton_model_config_overrides": {"max_batch_size": 32},
    "number_of_model_copies": 1,
}


if not isinstance(config["artifact_version"], int):
    raise ValueError("Triton requires model version to be an integer")

if "triton_url" not in config:
    raise ValueError("`triton_url` must be specified in config")

if "triton_bucket" not in config:
    raise ValueError(
        "`triton_bucket` must be specified in config in the form of your-bucket-name (omit the s3://)"
    )


with wandb.init(config=config, job_type="deploy_to_triton") as run:

    # 1. Convert wandb artifact into triton-compatible format and upload to S3 bucket
    print(">> Downloading wandb artifact")
    wandb_artifact_str = (
        "{entity}/{project}/{artifact_name}:v{artifact_version}".format(**run.config)
    )
    art = run.use_artifact(wandb_artifact_str)
    path = art.download()

    def _deploy(i):
        print("starting job", i)
        model_name = run.config["artifact_name"] + f"_copy_{i}"

        print(">> Uploading model to Triton model repository (this may take a while)")
        remote_path = os.path.join(
            run.config["triton_model_repo_path"],
            model_name,
            str(run.config["artifact_version"]),
            "model.savedmodel",
        )

        s3_client = boto3.client("s3")
        for root, _, files in os.walk(path):
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, path)
                remote_obj_path = os.path.join(remote_path, rel_path)
                print(f"Uploading {rel_path} to {remote_obj_path}")
                s3_client.upload_file(
                    full_path, config["triton_bucket"], remote_obj_path
                )

        # 4. Load the model to triton
        print(">> Loading model into triton")
        client = httpclient.InferenceServerClient(
            url=run.config["triton_url"], verbose=True
        )
        version_config = {
            "version_policy": {
                "specific": {"versions": [run.config["artifact_version"]]}
            }
        }

        base_pbtxt_config = s3_config_pbtxt_to_dict(
            bucket=run.config["triton_bucket"],
            pbtxt_path=f"{run.config['triton_model_repo_path']}/{model_name}/config.pbtxt",
        )
        triton_configs = {
            **base_pbtxt_config,
            **version_config,
            **run.config["triton_model_config_overrides"],
        }
        dict_to_config_pbtxt(triton_configs, "overloaded.pbtxt")
        client.load_model(model_name, config=json.dumps(triton_configs))

        if not client.is_model_ready(model_name):
            print(f"Failed to load model {model_name}")

    # process pool does not help; just loop over models to deploy n times
    for c in range(run.config["number_of_model_copies"]):
        _deploy(c)

    # Log code so it can be Launch'd
    run.log_code()
