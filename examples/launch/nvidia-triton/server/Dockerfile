FROM nvcr.io/nvidia/tritonserver:22.11-py3
ENTRYPOINT ["tritonserver", "--model-repository", "s3://andrew-triton-bucket/models", "--model-control-mode", "explicit"]