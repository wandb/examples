# Deploy a model from Artifacts to Triton

## 1. Build and run the Triton server container
```
docker build -t tritonserver-wandb server && \
docker run --env-file ./env.list --rm -p 8000:8000 tritonserver-wandb
```

## 2. Build and run the deployer container, which loads a model from wandb Artifacts into Triton
```
docker build -t triton-wandb-deploy deployer && \
docker run --env-file ./env.list --rm --net=host triton-wandb-deploy
```

## Note on env file
You may need to include:
1. WANDB_API_KEY
2. AWS_DEFAULT_REGION
3. AWS_ACCESS_KEY_ID
4. AWS_SECRET_ACCESS_KEY

## Note on testing
- Tested using keras savedmodel and torchscript on CPU
- Note: For pytorch, your model needs to have already been converted to torchscript and saved to Artifacts before uploading -- currently investigating if we can do the conversion automatically.
