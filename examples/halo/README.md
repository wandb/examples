# Track Halo training with W&B

[Halo](https://github.com/whitecircle/halo) lets teams scale the Hugging Face models and training code they already use without converting models to a framework-specific format. Faster kernels reduce training time and peak memory. With W&B, one run shows training quality, throughput, and memory together.

This example fine-tunes Qwen3-4B with LoRA for 20 steps and records:

- training loss and learning rate
- tokens per second and step time
- allocated, reserved, and peak GPU memory

## Run

Install Halo by following its [installation guide](https://github.com/whitecircle/halo#installation), authenticate W&B, and launch from the Halo repository root:

```bash
wandb login
halo launch sft /path/to/examples/halo/train.yaml
```

The dashboard is created in the `halo-examples` project with the run name `qwen3-4b-lora`. Edit `project_name` and `run_name` in `train.yaml` to choose another destination.

`enable_efficiency_metrics: true` adds Halo's measured throughput, step-time, and memory series to the normal Transformers logs. The first two steps are warmup; compare steady-state steps when evaluating a run.

Remove `max_steps` and set `num_train_epochs` for complete training. Keep the model, sequence length, batch size, precision, and hardware fixed when comparing runs.

Read more at [whitecircle.com/halo](https://whitecircle.com/halo).
