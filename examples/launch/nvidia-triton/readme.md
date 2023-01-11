## 1. Build and run the Triton server container
```
docker build -t tritonserver-wandb server && \
docker run --env-file ./env.list --rm --net=host tritonserver-wandb
```

## 2. Build and run the deployer container, which loads a model from wandb Artifacts into Triton
```
docker build -t triton-wandb-deploy && \
docker run --env-file ./env.list --rm --net=host triton-wandb-deploy
```

## Note on env file
You may need to include:
1. WANDB_API_KEY
2. AWS_DEFAULT_REGION
3. AWS_ACCESS_KEY_ID
4. AWS_SECRET_ACCESS_KEY