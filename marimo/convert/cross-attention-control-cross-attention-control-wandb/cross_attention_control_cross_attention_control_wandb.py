# /// script
# dependencies = ["diffusers", "ftfy", "transformers", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/cross-attention-control/Cross_Attention_Control_WandB.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{cross-attention-control} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 Cross-Attention Control with Stable Diffusion + WandB Playground 🪄🐝

    <!--- @wandbcode{cross-attention-control} -->

    An implementation of Prompt-to-Prompt Image Editing
    with Cross Attention Control using [Stable Diffusion](https://github.com/CompVis/stable-diffusion), [HuggingFace Diffusers](https://github.com/huggingface/diffusers) and [Weights & Biases](https://wandb.ai/site).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 1: Setup required libraries
    """)
    return


@app.cell
def _():
    #@title

    # packages added via marimo's package management: diffusers transformers ftfy wandb !pip install -q diffusers transformers ftfy wandb
    return


@app.cell
def _():
    #@title

    import io
    import wandb
    import random
    import numpy as np
    from PIL import Image
    from tqdm.auto import tqdm
    from difflib import SequenceMatcher
    from google.colab import files as colab_files

    import torch
    from torch import autocast

    from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
    from diffusers import (
        AutoencoderKL, UNet2DConditionModel, LMSDiscreteScheduler
    )

    return (
        AutoencoderKL,
        CLIPModel,
        CLIPTokenizer,
        Image,
        LMSDiscreteScheduler,
        SequenceMatcher,
        UNet2DConditionModel,
        autocast,
        np,
        random,
        torch,
        tqdm,
        wandb,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2: Set up Models and Weights & Biases Run

    - `wandb_project`: Weights & Biases project.
    - `wandb_project`: Weights & Biases entity.
    - `huggingface_access_token`: HuggingFace Access Token. Check out this page from the official HuggingFace docs as to how to generate your own access token.
    - `config.device`: Accelerator device. Choose `mps` if you're running the code on an M1 Mac.
    - `config.model_path_clip`: Alias for pre-trained CLIP Model.
    - `config.model_path_diffusion`: Alias for pre-trained Stable Diffusion Model.
    """)
    return


@app.cell
def _(
    AutoencoderKL,
    CLIPModel,
    CLIPTokenizer,
    UNet2DConditionModel,
    torch,
    wandb,
):
    wandb_project = "cross-attention-control" #@param {"type": "string"}
    wandb_entity = "wandb" #@param {"type": "string"}

    wandb.init(project=wandb_project, entity=wandb_entity, job_type="generate")
    config = wandb.config

    huggingface_access_token = "" #@param {"type": "string"}
    torch_dtype = torch.float16

    config.model_precision_type = "fp16"
    config.device = "cuda" #@param['cuda', 'cpu', 'mps']
    config.model_path_clip = "openai/clip-vit-large-patch14" #@param['openai/clip-vit-large-patch14']
    config.model_path_diffusion = "CompVis/stable-diffusion-v1-4" #@param['CompVis/stable-diffusion-v1-4']


    clip_tokenizer = CLIPTokenizer.from_pretrained(config.model_path_clip)
    clip_model = CLIPModel.from_pretrained(
        config.model_path_clip,
        torch_dtype=torch_dtype
    )
    clip = clip_model.text_model


    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(
        model_path_diffusion,
        subfolder="unet",
        use_auth_token=huggingface_access_token,
        revision=config.model_precision_type,
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        model_path_diffusion,
        subfolder="vae",
        use_auth_token=huggingface_access_token,
        revision=config.model_precision_type,
        torch_dtype=torch.float16
    )


    unet.to(config.device)
    vae.to(config.device)
    clip.to(config.device)
    return clip, clip_tokenizer, config, unet, vae


@app.cell
def _(Image, SequenceMatcher, clip_tokenizer, config, torch, unet):
    #@title

    def init_attention_weights(weight_tuples):
        tokens_length = clip_tokenizer.model_max_length
        weights = torch.ones(tokens_length)
    
        for i, w in weight_tuples:
            if i < tokens_length and i >= 0:
                weights[i] = w
    
    
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.last_attn_slice_weights = weights.to(config.device)
            if module_name == "CrossAttention" and "attn1" in name:
                module.last_attn_slice_weights = None
    

    def init_attention_edit(tokens, tokens_edit):
        tokens_length = clip_tokenizer.model_max_length
        mask = torch.zeros(tokens_length)
        indices_target = torch.arange(tokens_length, dtype=torch.long)
        indices = torch.zeros(tokens_length, dtype=torch.long)

        tokens = tokens.input_ids.numpy()[0]
        tokens_edit = tokens_edit.input_ids.numpy()[0]
    
        for name, a0, a1, b0, b1 in SequenceMatcher(
            None, tokens, tokens_edit
        ).get_opcodes():
            if b0 < tokens_length:
                if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                    mask[b0:b1] = 1
                    indices[b0:b1] = indices_target[a0:a1]

        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.last_attn_slice_mask = mask.to(config.device)
                module.last_attn_slice_indices = indices.to(config.device)
            if module_name == "CrossAttention" and "attn1" in name:
                module.last_attn_slice_mask = None
                module.last_attn_slice_indices = None


    def init_attention_func():
        def new_attention(self, query, key, value, sequence_length, dim):
            batch_size_attention = query.shape[0]
            hidden_states = torch.zeros(
                (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
            )
            slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
            for i in range(hidden_states.shape[0] // slice_size):
                start_idx = i * slice_size
                end_idx = (i + 1) * slice_size
                attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
                )
                attn_slice = attn_slice.softmax(dim=-1)
            
                if self.use_last_attn_slice:
                    if self.last_attn_slice_mask is not None:
                        new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                        attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                    else:
                        attn_slice = self.last_attn_slice
                
                    self.use_last_attn_slice = False
                    
                if self.save_last_attn_slice:
                    self.last_attn_slice = attn_slice
                    self.save_last_attn_slice = False
                
                if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                    attn_slice = attn_slice * self.last_attn_slice_weights
                    self.use_last_attn_weights = False

                attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

                hidden_states[start_idx:end_idx] = attn_slice

            # reshape hidden_states
            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            return hidden_states

        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention":
                module.last_attn_slice = None
                module.use_last_attn_slice = False
                module.use_last_attn_weights = False
                module.save_last_attn_slice = False
                module._attention = new_attention.__get__(module, type(module))
            
    def use_last_tokens_attention(use=True):
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.use_last_attn_slice = use
            
    def use_last_tokens_attention_weights(use=True):
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.use_last_attn_weights = use
            
    def use_last_self_attention(use=True):
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn1" in name:
                module.use_last_attn_slice = use
            
    def save_last_tokens_attention(save=True):
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn2" in name:
                module.save_last_attn_slice = save
            
    def save_last_self_attention(save=True):
        for name, module in unet.named_modules():
            module_name = type(module).__name__
            if module_name == "CrossAttention" and "attn1" in name:
                module.save_last_attn_slice = save


    def postprocess(image):
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        return Image.fromarray(image)

    return (
        init_attention_edit,
        init_attention_func,
        init_attention_weights,
        postprocess,
        save_last_self_attention,
        save_last_tokens_attention,
        use_last_self_attention,
        use_last_tokens_attention,
        use_last_tokens_attention_weights,
    )


@app.cell
def _(
    Image,
    LMSDiscreteScheduler,
    autocast,
    clip,
    clip_tokenizer,
    config,
    init_attention_edit,
    init_attention_func,
    init_attention_weights,
    np,
    postprocess,
    random,
    save_last_self_attention,
    save_last_tokens_attention,
    torch,
    tqdm,
    unet,
    use_last_self_attention,
    use_last_tokens_attention,
    use_last_tokens_attention_weights,
    vae,
    wandb,
):
    #@title

    @torch.no_grad()
    def stablediffusion(
        prompt="",
        prompt_edit="",
        prompt_edit_token_weights=[],
        prompt_edit_tokens_start=0.0,
        prompt_edit_tokens_end=1.0,
        prompt_edit_spatial_start=0.0,
        prompt_edit_spatial_end=1.0,
        guidance_scale=7.5,
        steps=50,
        seed=None,
        width=512,
        height=512,
        init_image=None,
        init_image_strength=0.5,
    ):
        log_key = (
            "Generated Image without Promp Edit"
            if prompt_edit == ""
            else "Generated Image with Promp Edit"
        )
        print(log_key)

        # Change size to multiple of 64 to prevent size mismatches inside model
        width = width - width % 64
        height = height - height % 64
    
        #If seed is None, randomly select seed from 0 to 2^32-1
        if seed is None: seed = random.randrange(2**32 - 1)
        generator = torch.cuda.manual_seed(seed)
    
        # Set inference timesteps to scheduler
        scheduler = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000
        )
        scheduler.set_timesteps(steps)
    
        # Preprocess image if it exists (img2img)
        if init_image is not None:
            #Resize and transpose for numpy b h w c -> torch b c h w
            init_image = init_image.resize(
                (width, height), resample=Image.LANCZOS
            )
            init_image = np.array(
                init_image
            ).astype(np.float32) / 255.0 * 2.0 - 1.0
            init_image = torch.from_numpy(
                init_image[np.newaxis, ...].transpose(0, 3, 1, 2)
            )
        
            # If there is alpha channel, composite alpha for white,
            # as the diffusion model does not support alpha channel
            if init_image.shape[1] > 3:
                init_image = init_image[:, :3] * init_image[:, 3:] + (
                    1 - init_image[:, 3:]
                )
            
            #Move image to GPU
            init_image = init_image.to(config.device)
        
            #Encode image
            with autocast(config.device):
                init_latent = vae.encode(
                    init_image
                ).latent_dist.sample(generator=generator) * 0.18215
            
            t_start = steps - int(steps * init_image_strength)
            
        else:
            init_latent = torch.zeros(
                (1, unet.in_channels, height // 8, width // 8),
                device=config.device
            )
            t_start = 0
    
        # Generate random normal noise
        noise = torch.randn(
            init_latent.shape, generator=generator, device=config.device
        )
        latent = scheduler.add_noise(
            init_latent, noise, t_start
        ).to(config.device)
    
        # Process clip
        with autocast(config.device):
            tokens_unconditional = clip_tokenizer(
                "",
                padding="max_length",
                max_length=clip_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
                return_overflowing_tokens=True
            )
            embedding_unconditional = clip(
                tokens_unconditional.input_ids.to(config.device)
            ).last_hidden_state

            tokens_conditional = clip_tokenizer(
                prompt,
                padding="max_length",
                max_length=clip_tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
                return_overflowing_tokens=True
            )
            embedding_conditional = clip(
                tokens_conditional.input_ids.to(config.device)
            ).last_hidden_state

            # Process prompt editing
            if prompt_edit != "":
                tokens_conditional_edit = clip_tokenizer(
                    prompt_edit,
                    padding="max_length",
                    max_length=clip_tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                    return_overflowing_tokens=True
                )
                embedding_conditional_edit = clip(
                    tokens_conditional_edit.input_ids.to(config.device)
                ).last_hidden_state
            
                init_attention_edit(
                    tokens_conditional, tokens_conditional_edit
                )
            
            init_attention_func()
            init_attention_weights(prompt_edit_token_weights)
            
            timesteps = scheduler.timesteps[t_start:]
        
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
                t_index = t_start + i

                sigma = scheduler.sigmas[t_index]
                latent_model_input = latent
                latent_model_input = (
                    latent_model_input / ((sigma**2 + 1) ** 0.5)
                ).to(unet.dtype)

                # Predict the unconditional noise residual
                noise_pred_uncond = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embedding_unconditional
                ).sample
            
                # Prepare the Cross-Attention layers
                if prompt_edit is not None:
                    save_last_tokens_attention()
                    save_last_self_attention()
                else:
                    #Use weights on non-edited prompt when edit is None
                    use_last_tokens_attention_weights()
                
                # Predict the conditional noise residual and save
                # the cross-attention layer activations
                noise_pred_cond = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=embedding_conditional
                ).sample
            
                # Edit the Cross-Attention layer activations
                if prompt_edit != "":
                    t_scale = t / scheduler.num_train_timesteps
                    if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                        use_last_tokens_attention()
                    if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                        use_last_self_attention()
                    
                    # Use weights on edited prompt
                    use_last_tokens_attention_weights()

                    # Predict the edited conditional noise residual
                    # using the cross-attention masks
                    noise_pred_cond = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=embedding_conditional_edit
                    ).sample
                
                #Perform guidance
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                latent = scheduler.step(noise_pred, t_index, latent).prev_sample

                wandb.log({
                    log_key: wandb.Image(
                        postprocess(
                            vae.decode((latent / 0.18215).to(vae.dtype)).sample
                        )
                    )
                }, step=i)

            # scale and decode the image latents with vae
            latent = latent / 0.18215
            image = vae.decode(latent.to(vae.dtype)).sample

        return postprocess(image)

    return (stablediffusion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 3: Enter Prompts and Additional Configs

    - `config.prompt`: The prompt as a string.
    - `config.prompt_edit`: The second prompt as a string, used to edit the first prompt using cross attention, set `"\"` to disable.
    - `config.prompt_edit_token_weights`: Values to scale the importance of the tokens in cross attention layers, as a list of tuples representing `(token id, strength)`, this is used to increase or decrease the importance of a word in the prompt, it is applied to prompt_edit when possible (if `prompt_edit` is `"\"`, weights are applied to prompt).
    - `config.prompt_edit_tokens_start`: How strict is the generation with respect to the initial prompt, increasing this will let the network be more creative for smaller details/textures, should be smaller than `prompt_edit_tokens_end`.
    - `config.prompt_edit_tokens_end`: How strict is the generation with respect to the initial prompt, decreasing this will let the network be more creative for larger features/general scene composition, should be bigger than `prompt_edit_tokens_start`.
    - `config.prompt_edit_spatial_start`: How strict is the generation with respect to the initial image (generated from the first prompt, not from img2img), increasing this will let the network be more creative for smaller details/textures, should be smaller than `prompt_edit_spatial_end`.
    - `config.prompt_edit_spatial_end`: How strict is the generation with respect to the initial image (generated from the first prompt, not from img2img), decreasing this will let the network be more creative for larger features/general scene composition, should be bigger than `prompt_edit_spatial_start`.
    - `config.guidance_scale`: Standard classifier-free guidance strength for stable diffusion.
    - `config.steps`: Number of diffusion steps as an integer, higher usually produces better images but is slower.
    - `config.seed`: Random Seed.
    - `config.image_width`: Width of generated image.
    - `config.image_height`: Height of generated image.
    """)
    return


@app.cell
def _(clip_tokenizer, config):
    def display_prompt_tokens(prompt):
        tokens = clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=clip_tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
            return_overflowing_tokens=True
        ).input_ids[0]
        for idx, token in enumerate(tokens):
            decoded_token = clip_tokenizer.decode(token)
            if decoded_token == "<|startoftext|>":
                continue
            elif decoded_token == "<|endoftext|>":
                break
            else:
                print(idx, "->", decoded_token)


    # the prompt as a string
    config.prompt = "A photo of a Person with flower headpiece and elegant jewels" #@param {"type": "string"}

    # the second prompt as a string, used to edit the first prompt
    # using cross attention, set "" to disable
    config.prompt_edit = "A photo of a Person with butterfly headpiece and elegant jewels" #@param {"type": "string"}

    display_prompt_tokens(config.prompt_edit)
    return


@app.cell
def _(config):
    # values to scale the importance of the tokens in
    # cross attention layers, as a list of tuples representing
    # (token id, strength), this is used to increase or decrease
    # the importance of a word in the prompt, it is applied to prompt_edit when possible (if prompt_edit is None, weights are applied to prompt)
    config.prompt_edit_token_weights = [(7, 4)] #@param {type:"raw"}

    # how strict is the generation with respect to the initial prompt,
    # increasing this will let the network be more creative for smaller
    # details/textures, should be smaller than prompt_edit_tokens_end
    config.prompt_edit_tokens_start = 0.0 #@param {type:"slider", min:0, max:1, step:0.1}

    # how strict is the generation with respect to the initial prompt,
    # decreasing this will let the network be more creative for larger
    # features/general scene composition, should be bigger than
    # prompt_edit_tokens_start
    config.prompt_edit_tokens_end = 1.0 #@param {type:"slider", min:0, max:1, step:0.1}

    # how strict is the generation with respect to the initial image
    # (generated from the first prompt, not from img2img), increasing
    # this will let the network be more creative for smaller
    # details/textures, should be smaller than prompt_edit_spatial_end
    config.prompt_edit_spatial_start = 0.0 #@param {type:"slider", min:0, max:1, step:0.1}

    # how strict is the generation with respect to the initial image
    # (generated from the first prompt, not from img2img), decreasing
    # this will let the network be more creative for larger
    # features/general scene composition, should be bigger than
    # prompt_edit_spatial_start
    config.prompt_edit_spatial_end = 0.8 #@param {type:"slider", min:0, max:1, step:0.1}

    # standard classifier-free guidance strength for stable diffusion
    config.guidance_scale = 7.5 #@param {type:"slider", min:0, max:100, step:0.1}

    # number of diffusion steps as an integer, higher usually produces
    # better images but is slower
    config.steps = 50 #@param {type:"slider", min:0, max:1000, step:1}

    # random seed as an integer
    config.seed = 98374234 #@param {type:"number"}

    # image width and heigh
    config.image_width = 768 #@param {type:"slider", min:512, max:1024, step:1}
    config.image_height = 512 #@param {type:"slider", min:512, max:1024, step:1}
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 4: Generate Images with Prompt and Prompt Edits.

    Image genetaed will be automatically logged to the respective **Weights & Biases** workspace as an interactive [**Table**](https://docs.wandb.ai/guides/data-vis) with all configs.

    ![](https://i.imgur.com/CqIJgPg.png)
    """)
    return


@app.cell
def _(config, stablediffusion, wandb):
    #@title

    generated_image_with_prompt_edit = stablediffusion(
        prompt=config.prompt,
        prompt_edit=config.prompt_edit,
        prompt_edit_token_weights=config.prompt_edit_token_weights,
        prompt_edit_tokens_start=config.prompt_edit_tokens_start,
        prompt_edit_tokens_end=config.prompt_edit_tokens_end,
        prompt_edit_spatial_start=config.prompt_edit_spatial_start,
        prompt_edit_spatial_end=config.prompt_edit_spatial_end,
        guidance_scale=config.guidance_scale,
        steps=config.steps,
        seed=config.seed,
        width=config.image_width,
        height=config.image_height,
        init_image=None,
        init_image_strength=0.5
    )

    if config.prompt_edit != "":
        generated_image_without_prompt_edit = stablediffusion(
            prompt=config.prompt,
            prompt_edit="",
            prompt_edit_token_weights=config.prompt_edit_token_weights,
            prompt_edit_tokens_start=config.prompt_edit_tokens_start,
            prompt_edit_tokens_end=config.prompt_edit_tokens_end,
            prompt_edit_spatial_start=config.prompt_edit_spatial_start,
            prompt_edit_spatial_end=config.prompt_edit_spatial_end,
            guidance_scale=config.guidance_scale ,
            steps=config.steps,
            seed=config.seed,
            width=config.image_width,
            height=config.image_height,
            init_image=None,
            init_image_strength=0.5
        )
        table = wandb.Table(
            columns=[
                "Seed",
                "Guidance Scale",
                "Image Height",
                "Image Width",
                "Number of Steps",
                "Prompt",
                "Image Generated With Prompt",
                "Prompt Edit",
                "Edit Token Weights",
                "Image Generated With Prompt Edit"
            ]
        )
        table.add_data(
            config.seed,
            config.guidance_scale,
            config.image_height,
            config.image_width,
            config.steps,
            config.prompt,
            wandb.Image(generated_image_without_prompt_edit),
            config.prompt_edit,
            config.prompt_edit_token_weights,
            wandb.Image(generated_image_with_prompt_edit)
        )
        wandb.log({
            "Image Editing with Cross Attention Control": table
        })


    wandb.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **References:**
    - https://arxiv.org/abs/2208.01626
    - https://github.com/bloc97/CrossAttentionControl
    """)
    return


if __name__ == "__main__":
    app.run()
