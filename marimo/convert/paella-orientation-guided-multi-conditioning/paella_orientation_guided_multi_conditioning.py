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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/paella/Orientation-Guided-Multi-Conditioning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{paella-orientation-guided} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Orientation Guided Multi-Conditional Image Generation with Paella + WandB Playground 🪄🐝

    <!--- @wandbcode{paella-orientation-guided} -->

    A demo of Orientation Guided Multi-Conditional Image Generation using [Paella](https://github.com/dome272/Paella) and [Weights & Biases](https://wandb.ai/site).
    """)
    return


@app.cell
def _():
    import os
    import time
    import wandb
    import requests
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from tqdm.notebook import tqdm
    import matplotlib.pyplot as plt

    import torch
    from torch import nn
    import torchvision

    import open_clip
    from rudalle import get_vae
    from einops import rearrange
    from open_clip import tokenizer

    from Paella.modules import DenoiseUNet

    return (
        DenoiseUNet,
        get_vae,
        np,
        open_clip,
        os,
        rearrange,
        time,
        tokenizer,
        torch,
        wandb,
    )


@app.cell
def _(os, torch, wandb):
    wandb_project = "paella" #@param {"type": "string"}
    wandb_entity = "geekyrakshit" #@param {"type": "string"}

    wandb.init(project=wandb_project, entity=wandb_entity, job_type="orientation-guided-multi-conditioning")


    config = wandb.config
    config.model_artifact = "geekyrakshit/paella/text-model:v0"
    config.seed = 42
    config.batch_size = 5
    config.latent_shape = (32, 32)
    config.prompt_1 = "a cute portrait of a dog"
    config.prompt_2 = "a cute portrait of a cat"
    config.orientation_mode = "horizontal" # ["vertical", "horizontal"]
    config.interpolation_mode = "spherical-lerp" # ["lerp", "spherical-lerp"]

    # Seed Everything
    torch.manual_seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Download Model from Weights & Biases Artifacts
    text_model_path = os.path.join(wandb.use_artifact(config.model_artifact, type='model').download(), "model_600000.pt")
    return config, text_model_path


@app.cell
def _(torch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return (device,)


@app.cell
def _(config, np, wandb):
    def log_orientation_guided_multi_conditioning_results(images):
        images = [wandb.Image(image) for image in (images.cpu().numpy() * 255.0).astype(np.uint8)]
        table = wandb.Table(
            columns=[
                "Seed",
                "Prompt-1", "Prompt-2",
                "Orientation-Mode", "Interpolation-Mode", "Latent-Shape",
                "Generated-Images"
            ]
        )
        table.add_data(
            config.seed,
            config.prompt_1, config.prompt_2,
            config.orientation_mode, config.interpolation_mode, config.latent_shape,
            images
        )
        wandb.log({"Orientation-Guided-Multi-Conditioning-Results": table})

    return (log_orientation_guided_multi_conditioning_results,)


@app.cell
def _(torch):
    def log(t, eps=1e-20):
        return torch.log(t + eps)

    def gumbel_noise(t):
        noise = torch.zeros_like(t).uniform_(0, 1)
        return -log(-log(noise))

    def gumbel_sample(t, temperature=1., dim=-1):
        return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

    def sample(
        model, c, x=None, mask=None, T=12, size=(32, 32),
        starting_t=0, temp_range=[1.0, 1.0], typical_filtering=True,
        typical_mass=0.2, typical_min_tokens=1, classifier_free_scale=-1,
        renoise_steps=11, renoise_mode='start'
    ):
        with torch.inference_mode():
            r_range = torch.linspace(0, 1, T+1)[:-1][:, None].expand(-1, c.size(0)).to(c.device)
            temperatures = torch.linspace(temp_range[0], temp_range[1], T)
            preds = []
            if x is None:
                x = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
            elif mask is not None:
                noise = torch.randint(0, model.num_labels, size=(c.size(0), *size), device=c.device)
                x = noise * mask + (1-mask) * x
            init_x = x.clone()
            for i in range(starting_t, T):
                if renoise_mode == 'prev':
                    prev_x = x.clone()
                r, temp = r_range[i], temperatures[i]
                logits = model(x, c, r)
                if classifier_free_scale >= 0:
                    logits_uncond = model(x, torch.zeros_like(c), r)
                    logits = torch.lerp(logits_uncond, logits, classifier_free_scale)
                x = logits
                x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
                if typical_filtering:
                    x_flat_norm = torch.nn.functional.log_softmax(x_flat, dim=-1)
                    x_flat_norm_p = torch.exp(x_flat_norm)
                    entropy = -(x_flat_norm * x_flat_norm_p).nansum(-1, keepdim=True)

                    c_flat_shifted = torch.abs((-x_flat_norm) - entropy)
                    c_flat_sorted, x_flat_indices = torch.sort(c_flat_shifted, descending=False)
                    x_flat_cumsum = x_flat.gather(-1, x_flat_indices).softmax(dim=-1).cumsum(dim=-1)

                    last_ind = (x_flat_cumsum < typical_mass).sum(dim=-1)
                    sorted_indices_to_remove = c_flat_sorted > c_flat_sorted.gather(1, last_ind.view(-1, 1))
                    if typical_min_tokens > 1:
                        sorted_indices_to_remove[..., :typical_min_tokens] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, x_flat_indices, sorted_indices_to_remove)
                    x_flat = x_flat.masked_fill(indices_to_remove, -float("Inf"))
                x_flat = gumbel_sample(x_flat, temperature=temp)
                x = x_flat.view(x.size(0), *x.shape[2:])
                if mask is not None:
                    x = x * mask + (1-mask) * init_x
                if i < renoise_steps:
                    if renoise_mode == 'start':
                        x, _ = model.add_noise(x, r_range[i+1], random_x=init_x)
                    elif renoise_mode == 'prev':
                        x, _ = model.add_noise(x, r_range[i+1], random_x=prev_x)
                    else: # 'rand'
                        x, _ = model.add_noise(x, r_range[i+1])
                preds.append(x.detach())
        return preds

    return (sample,)


@app.cell
def _(
    DenoiseUNet,
    device,
    get_vae,
    open_clip,
    rearrange,
    text_model_path,
    torch,
):
    vqmodel = get_vae().to(device)
    vqmodel.eval().requires_grad_(False)

    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-g-14', pretrained='laion2b_s12b_b42k')
    clip_model = clip_model.to(device).eval().requires_grad_(False)


    def encode(x):
        return vqmodel.model.encode((2 * x - 1))[-1][-1]
    
    def decode(img_seq, shape=(32,32)):
            img_seq = img_seq.view(img_seq.shape[0], -1)
            b, n = img_seq.shape
            one_hot_indices = torch.nn.functional.one_hot(img_seq, num_classes=vqmodel.num_tokens).float()
            z = (one_hot_indices @ vqmodel.model.quantize.embed.weight)
            z = rearrange(z, 'b (h w) c -> b c h w', h=shape[0], w=shape[1])
            img = vqmodel.model.decode(z)
            img = (img.clamp(-1., 1.) + 1) * 0.5
            return img
    
    state_dict = torch.load(text_model_path, map_location=device)
    model = DenoiseUNet(num_labels=8192).to(device)
    model.load_state_dict(state_dict)
    model.eval().requires_grad_()
    print()
    return clip_model, decode, model


@app.cell
def _(
    clip_model,
    config,
    decode,
    device,
    model,
    sample,
    time,
    tokenizer,
    torch,
    wandb,
):
    text = tokenizer.tokenize([config.prompt_1, config.prompt_2] * config.batch_size).to(device)

    with torch.inference_mode():
        with torch.autocast(device_type="cuda"):
            clip_embeddings = clip_model.encode_text(text).float()
            clip_embeddings = clip_embeddings[:, :, None, None]
            clip_embeddings = clip_embeddings.expand(
                -1, -1, config.latent_shape[0], config.latent_shape[1]
            )
        
            if config.orientation_mode == 'vertical':
                interp_mask = torch.linspace(0, 1, config.latent_shape[0], device=device)
                interp_mask = interp_mask[None, None, :, None]
                interp_mask = interp_mask.expand(config.batch_size, 1, -1, config.latent_shape[1])
            else: 
                interp_mask = torch.linspace(0, 1, config.latent_shape[1], device=device)
                interp_mask = interp_mask[None, None, None, :]
                interp_mask = interp_mask.expand(config.batch_size, 1, config.latent_shape[0], -1)
        
            if config.interpolation_mode == "lerp":
                clip_embeddings = clip_embeddings[0::2] * (1 - interp_mask) + clip_embeddings[1::2] * interp_mask
            elif config.interpolation_mode == "spherical-lerp":
                low, high = clip_embeddings[0::2], clip_embeddings[1::2]
                low_norm = low / torch.norm(low, dim=1, keepdim=True)
                high_norm = high / torch.norm(high, dim=1, keepdim=True)
                omega = torch.acos((low_norm * high_norm).sum(1)).unsqueeze(1)
                so = torch.sin(omega)
                clip_embeddings = (torch.sin((1.0 - interp_mask) * omega) / so) * low
                clip_embeddings = clip_embeddings + (torch.sin(interp_mask * omega) / so) * high
    
            s = time.time()
            sampled = sample(
                model, clip_embeddings, T=12, size=config.latent_shape, starting_t=0,
                temp_range=[1.0, 1.0], typical_filtering=True, typical_mass=0.2,
                typical_min_tokens=1, classifier_free_scale=5, renoise_steps=11, renoise_mode="start"
            )
            wandb.log({"Sampling-Time": time.time() - s})
        sampled = decode(sampled[-1], config.latent_shape).permute(0, 2, 3, 1)
    return (sampled,)


@app.cell
def _(log_orientation_guided_multi_conditioning_results, sampled, wandb):
    log_orientation_guided_multi_conditioning_results(sampled)
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
