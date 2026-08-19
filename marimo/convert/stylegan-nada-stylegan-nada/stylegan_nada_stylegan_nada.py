# /// script
# dependencies = ["CLIP @ git+https://github.com/openai/CLIP.git", "ftfy", "regex", "tqdm", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/stylegan_nada/StyleGAN-NADA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{stylegan-nada-colab} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🔥🔥 StyelGAN-NADA + WandB Playground 🪄🐝

    <!--- @wandbcode{stylegan-nada-colab} -->

    **Original Implementation:** https://github.com/rinongal/StyleGAN-nada
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 1: Setup required libraries and models.
    This may take a few minutes.

    You may optionally enable downloads with pydrive in order to authenticate and avoid drive download limits when fetching pre-trained ReStyle and StyleGAN2 models.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %tensorflow_version 1.x

    import os

    restyle_dir = os.path.join("/content", "restyle")
    stylegan_ada_dir = os.path.join("/content", "stylegan_ada")
    stylegan_nada_dir = os.path.join("/content", "stylegan_nada")

    output_dir = os.path.join("/content", "output")

    output_model_dir = os.path.join(output_dir, "models")
    output_image_dir = os.path.join(output_dir, "images")
    return os, output_dir, restyle_dir, stylegan_nada_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Installing Requirements
    """)
    return


@app.cell
def _(subprocess):
    #! git clone --depth 1 https://github.com/yuval-alaluf/restyle-encoder.git $restyle_dir
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/yuval-alaluf/restyle-encoder.git', '$restyle_dir'])

    #! wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    subprocess.call(['wget', 'https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip'])
    #! sudo unzip ninja-linux.zip -d /usr/local/bin/
    subprocess.call(['sudo', 'unzip', 'ninja-linux.zip', '-d', '/usr/local/bin/'])
    #! sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
    subprocess.call(['sudo', 'update-alternatives', '--install', '/usr/bin/ninja', 'ninja', '/usr/local/bin/ninja', '1', '--force'])

    # packages added via marimo's package management: ftfy regex tqdm wandb !pip install ftfy regex tqdm wandb
    # packages added via marimo's package management: git+https://github.com/openai/CLIP.git !pip install git+https://github.com/openai/CLIP.git

    #! git clone --depth 1 https://github.com/NVlabs/stylegan2-ada/ $stylegan_ada_dir
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/NVlabs/stylegan2-ada/', '$stylegan_ada_dir'])
    #! git clone --depth 1 https://github.com/rinongal/stylegan-nada.git $stylegan_nada_dir
    subprocess.call(['git', 'clone', '--depth', '1', 'https://github.com/rinongal/stylegan-nada.git', '$stylegan_nada_dir'])
    return


@app.cell
def _():
    import wandb

    return (wandb,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _(os, restyle_dir, stylegan_nada_dir):
    from argparse import Namespace

    import sys

    import numpy as np
    from PIL import Image
    from glob import glob

    import torch
    import torchvision.transforms as transforms

    sys.path.append(restyle_dir)
    sys.path.append(stylegan_nada_dir)
    sys.path.append(os.path.join(stylegan_nada_dir, "ZSSGAN"))

    device = 'cuda'

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return Image, Namespace, device, glob, np, torch, transforms


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 2: Choose a model type.
    Model will be downloaded and converted to a pytorch compatible version.

    Re-runs of the cell with the same model will re-use the previously downloaded version.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Defining the Configs for Selecting the Model
    """)
    return


@app.cell
def _():
    project = "stylegan-nada" #@param {"type": "string"}
    source_model_type = 'ffhq' #@param['ffhq', 'cat', 'dog', 'church', 'horse', 'car']

    artifact_adressed = {
        "car": "geekyrakshit/stylegan-nada/car:v0",
        "horse": "geekyrakshit/stylegan-nada/horse:v0",
        "church": "geekyrakshit/stylegan-nada/church:v0",
        "dog": "geekyrakshit/stylegan-nada/dog:v0",
        "cat": "geekyrakshit/stylegan-nada/cat:v0",
        "ffhq": "geekyrakshit/stylegan-nada/ffhq:v0"
    }

    model_names = {
        "ffhq": "ffhq.pt",
        "cat": "afhqcat.pkl",
        "dog": "afhqdog.pkl",
        "church": "stylegan2-church-config-f.pkl",
        "car": "stylegan2-car-config-f.pkl",
        "horse": "stylegan2-horse-config-f.pkl"
    }

    dataset_sizes = {
        "ffhq": 1024,
        "cat": 512,
        "dog": 512,
        "church": 256,
        "horse": 256,
        "car": 512,
    }
    return (
        artifact_adressed,
        dataset_sizes,
        model_names,
        project,
        source_model_type,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initializing WandB
    """)
    return


@app.cell
def _(project, source_model_type, wandb):
    wandb.init(project=project, job_type="train")
    config = wandb.config
    config.source_model_type = source_model_type
    return (config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Fetching Models from [WandB Artifacts](https://docs.wandb.ai/guides/artifacts/artifacts-core-concepts)
    """)
    return


@app.cell
def _(artifact_adressed, model_names, source_model_type, wandb):
    _artifact = wandb.use_artifact(artifact_adressed[source_model_type])
    pretrained_model_dir = _artifact.download()
    pt_file_name = model_names[source_model_type].split('.')[0] + '.pt'
    return pretrained_model_dir, pt_file_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 3: Train the model.
    Describe your source and target class. These describe the direction of change you're trying to apply (e.g. "photo" to "sketch", "dog" to "the joker" or "dog" to "avocado dog").

    Alternatively, upload a directory with a small (~3) set of target style images (there is no need to preprocess them in any way) and set `style_image_dir` to point at them. This will use the images as a target rather than the source/class texts.

    We reccomend leaving the 'improve shape' button unticked at first, as it will lead to an increase in running times and is often not needed.
    For more drastic changes, turn it on and increase the number of iterations.

    As a rule of thumb:
    - Style and minor domain changes ('photo' -> 'sketch') require ~200-400 iterations.
    - Identity changes ('person' -> 'taylor swift') require ~150-200 iterations.
    - Simple in-domain changes ('face' -> 'smiling face') may require as few as 50.
    - The `style_image_dir` option often requires ~400-600 iterations.
    """)
    return


@app.cell
def _(config, dataset_sizes, np, source_model_type):
    from tqdm import notebook
    from ZSSGAN.model.ZSSGAN import ZSSGAN
    from ZSSGAN.utils.file_utils import save_images, get_dir_img_list
    from ZSSGAN.utils.training_utils import mixing_noise
    from IPython.display import display
    source_class = 'Human'
    config.source_class = source_class
    target_class = 'The Joker'
    config.target_class = target_class
    style_image_dir = ''
    config.style_image_dir = style_image_dir
    seed = 3  #@param {"type": "string"}
    config.seed = seed
    target_img_list = get_dir_img_list(style_image_dir) if style_image_dir else None
    improve_shape = False  #@param {"type": "string"}
    config.improve_shape = improve_shape
    model_choice = ['ViT-B/32', 'ViT-B/16']
    model_weights = [1.0, 0.0]  #@param {'type': 'string'}
    if improve_shape or style_image_dir:
        model_weights[1] = 1.0
    mixing = 0.9 if improve_shape else 0.0  #@param {"type": "integer"}
    auto_layers_k = int(2 * (2 * np.log2(dataset_sizes[source_model_type]) - 2) / 3) if improve_shape else 0
    auto_layer_iters = 1 if improve_shape else 0
    training_iterations = 251
    config.training_iterations = training_iterations
    output_interval = 10  #@param{type:"boolean"}
    config.output_interval = output_interval
    save_interval = 10
    config.save_interval = save_interval  #@param {type: "integer"}
    return (
        ZSSGAN,
        auto_layer_iters,
        auto_layers_k,
        mixing,
        mixing_noise,
        model_choice,
        model_weights,
        notebook,
        output_interval,
        save_interval,
        seed,
        source_class,
        target_class,
        target_img_list,
        training_iterations,
    )


@app.cell
def _(
    auto_layer_iters,
    auto_layers_k,
    config,
    dataset_sizes,
    mixing,
    model_choice,
    model_weights,
    os,
    output_dir,
    pretrained_model_dir,
    pt_file_name,
    save_interval,
    source_class,
    source_model_type,
    target_class,
    target_img_list,
    training_iterations,
):
    training_args = {
        "size": dataset_sizes[source_model_type],
        "batch": 2,
        "n_sample": 4,
        "output_dir": output_dir,
        "lr": 0.002,
        "frozen_gen_ckpt": os.path.join(pretrained_model_dir, pt_file_name),
        "train_gen_ckpt": os.path.join(pretrained_model_dir, pt_file_name),
        "iter": training_iterations,
        "source_class": source_class,
        "target_class": target_class,
        "lambda_direction": 1.0,
        "lambda_patch": 0.0,
        "lambda_global": 0.0,
        "lambda_texture": 0.0,
        "lambda_manifold": 0.0,
        "auto_layer_k": auto_layers_k,
        "auto_layer_iters": auto_layer_iters,
        "auto_layer_batch": 8,
        "output_interval": 50,
        "clip_models": model_choice,
        "clip_model_weights": model_weights,
        "mixing": mixing,
        "phase": None,
        "sample_truncation": 0.7,
        "save_interval": save_interval,
        "target_img_list": target_img_list,
        "img2img_batch": 16,
        "channel_multiplier": 2,
        "sg3": False,
        "sgxl": False,
    }
    config.training_args = training_args
    return (training_args,)


@app.cell
def _(
    Namespace,
    ZSSGAN,
    config,
    glob,
    np,
    os,
    seed,
    torch,
    training_args,
    wandb,
):
    args = Namespace(**training_args)
    resume_training_from_artifact = False
    config.resume_training_from_artifact = resume_training_from_artifact  #@param{type:"boolean"}
    checkpoint_artifact_address = 'geekyrakshit/stylegan-nada/model-winter-frost-8:v14'
    config.checkpoint_artifact_address = checkpoint_artifact_address
    print('Loading base models...')  #@param {'type': 'string'}
    net = ZSSGAN(args)
    print('Done')
    g_reg_ratio = 4 / 5
    g_optim = torch.optim.Adam(net.generator_trainable.parameters(), lr=args.lr * g_reg_ratio, betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
    if resume_training_from_artifact and checkpoint_artifact_address is not None:
        _artifact = wandb.use_artifact(checkpoint_artifact_address)
        _artifact_dir = _artifact.download()
        _checkpoint = torch.load(glob(os.path.join(_artifact_dir, '*.pt'))[0])
        net.generator_trainable.generator.load_state_dict(_checkpoint['g_ema'])
        g_optim.load_state_dict(_checkpoint['g_optim'])
    sample_dir = os.path.join(args.output_dir, 'sample')
    config.sample_dir = sample_dir
    ckpt_dir = os.path.join(args.output_dir, 'checkpoint')
    config.ckpt_dir = ckpt_dir
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.manual_seed(seed)
    # Set up output directories.
    np.random.seed(seed)
    return args, ckpt_dir, g_optim, net


@app.cell
def _(
    args,
    ckpt_dir,
    device,
    g_optim,
    mixing_noise,
    net,
    notebook,
    output_interval,
    source_model_type,
    torch,
    wandb,
):
    fixed_z = torch.randn(args.n_sample, 512, device=device)
    for i in notebook.tqdm(range(args.iter)):
        net.train()
        _sample_z = mixing_noise(args.batch, 512, args.mixing, device)
        [_sampled_src, _sampled_dst], clip_loss = net(_sample_z)
        wandb.log({'CLIP-Loss': clip_loss.item()}, step=i)
        net.zero_grad()
        clip_loss.backward()
        g_optim.step()
        if i % output_interval == 0:
            net.eval()
            with torch.no_grad():
                [_sampled_src, _sampled_dst], _loss = net([fixed_z], truncation=args.sample_truncation)
                if source_model_type == 'car':
                    _sampled_dst = _sampled_dst[:, :, 64:448, :]
                _sampled_dst = torch.permute(_sampled_dst, (0, 2, 3, 1)).cpu()
                _sampled_dst = [wandb.Image(dst.numpy()) for dst in _sampled_dst]
                wandb.log({'Samples': _sampled_dst}, step=i)
        if args.save_interval > 0 and i > 0 and (i % args.save_interval == 0):
            model_file = f'{ckpt_dir}/{str(i).zfill(6)}.pt'
            torch.save({'g_ema': net.generator_trainable.generator.state_dict(), 'g_optim': g_optim.state_dict()}, model_file)
            _artifact = wandb.Artifact(f'model-{wandb.run.name}', type='model')
            _artifact.add_file(model_file)
            wandb.log_artifact(_artifact, aliases=['latest', f'step_{i}'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Step 4: Generate samples with the new model
    """)
    return


@app.cell
def _(ZSSGAN, args, config, device, glob, os, source_model_type, torch, wandb):
    truncation = 0.7
    config.truncation = truncation
    samples = 9
    config.samples = samples
    _artifact = wandb.use_artifact(f'model-{wandb.run.name}:latest')
    _artifact_dir = _artifact.download()
    _checkpoint = torch.load(glob(os.path.join(_artifact_dir, '*.pt'))[0])
    print('Loading models from checkpoint artifact...')
    net_1 = ZSSGAN(args)
    net_1.generator_trainable.generator.load_state_dict(_checkpoint['g_ema'])
    print('Done')
    with torch.no_grad():
        net_1.eval()
        _sample_z = torch.randn(samples, 512, device=device)
        [_sampled_src, _sampled_dst], _loss = net_1([_sample_z], truncation=truncation)
        if source_model_type == 'car':
            _sampled_dst = _sampled_dst[:, :, 64:448, :]
        _sampled_dst = torch.permute(_sampled_dst, (0, 2, 3, 1)).cpu()
        _sampled_dst = [wandb.Image(dst.numpy()) for dst in _sampled_dst]
        wandb.log({'Generated Samples': _sampled_dst})
    return (net_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Editing a real image with Re-Style inversion (currently only FFHQ inversion is supported):
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 1: Fetch ReStyle Models from [WandB Artifacts](https://docs.wandb.ai/guides/artifacts/artifacts-core-concepts)

    This may take a few minutes
    """)
    return


@app.cell
def _(wandb):
    from restyle.utils.common import tensor2im
    from restyle.models.psp import pSp
    from restyle.models.e4e import e4e
    _artifact = wandb.use_artifact('geekyrakshit/stylegan-nada/restyle:v0')
    pretrained_model_dir_1 = _artifact.download()
    return e4e, pSp, pretrained_model_dir_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 2: Choose a re-style model

    We reccomend choosing the e4e model as it performs better under domain translations. Choose pSp for better reconstructions on minor domain changes (typically those that require less than 150 training steps).
    """)
    return


@app.cell
def _(Namespace, e4e, os, pSp, pretrained_model_dir_1, torch, transforms):
    encoder_type = 'e4e'  #@param['psp', 'e4e']
    restyle_experiment_args = {'model_path': os.path.join(pretrained_model_dir_1, f'restyle_{encoder_type}_ffhq_encode.pt'), 'transform': transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
    model_path = restyle_experiment_args['model_path']
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    restyle_net = (pSp if encoder_type == 'psp' else e4e)(opts)
    restyle_net.eval()
    restyle_net.cuda()
    print('Model successfully loaded!')
    return opts, restyle_experiment_args, restyle_net


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step 3: Align and invert an image
    """)
    return


@app.cell
def _(os):
    def run_alignment(image_path):
        import dlib
        from scripts.align_faces_parallel import align_face
        if not os.path.exists("shape_predictor_68_face_landmarks.dat"):
            print('Downloading files for aligning face image...')
            os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
            os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
            print('Done.')
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath=image_path, predictor=predictor) 
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image

    return (run_alignment,)


@app.cell
def _(
    Image,
    opts,
    os,
    restyle_experiment_args,
    restyle_net,
    run_alignment,
    subprocess,
    torch,
):
    image_url = "https://engineering.nyu.edu/sites/default/files/styles/square_large_default_2x/public/2018-06/yann-lecun.jpg" #@param {"type": "string"}
    file_name = "yann-lecun.jpg" #@param {"type": "string"}

    if not os.path.isfile(file_name):
        #! wget {image_url}
        subprocess.call(['wget', str(image_url)])

    image_path = os.path.join("/content", file_name)
    original_image = Image.open(image_path).convert("RGB")

    input_image = run_alignment(image_path)

    img_transforms = restyle_experiment_args['transform']
    transformed_image = img_transforms(input_image)

    def get_avg_image(net):
        avg_image = net(
            net.latent_avg.unsqueeze(0),
            input_code=True,
            randomize_noise=False,
            return_latents=False,
            average_code=True
        )[0]
        avg_image = avg_image.to('cuda').float().detach()
        return avg_image

    opts.n_iters_per_batch = 5
    opts.resize_outputs = False  # generate outputs at full resolution

    from restyle.utils.inference_utils import run_on_batch

    with torch.no_grad():
        avg_image = get_avg_image(restyle_net)
        result_batch, result_latents = run_on_batch(
            transformed_image.unsqueeze(0).cuda(), restyle_net, opts, avg_image
        )
    return (result_latents,)


@app.cell
def _(net_1, result_latents, source_class, target_class, torch, wandb):
    inverted_latent = torch.Tensor(result_latents[0][4]).cuda().unsqueeze(0).unsqueeze(1)
    with torch.no_grad():
        net_1.eval()
        [_sampled_src, _sampled_dst] = net_1(inverted_latent, input_is_latent=True)[0]
        _sampled_src = torch.permute(_sampled_src, (0, 2, 3, 1)).cpu().numpy()[0]
        _sampled_dst = torch.permute(_sampled_dst, (0, 2, 3, 1)).cpu().numpy()[0]
        table = wandb.Table(columns=['Source-Class-Text', 'Source-Image', 'Target-Class-Text', 'Translated-Image'], data=[[source_class, wandb.Image(_sampled_src), target_class, wandb.Image(_sampled_dst)]])
        wandb.log({'Restyle': table})
    return


@app.cell
def _(wandb):
    wandb.finish()
    return


if __name__ == "__main__":
    app.run()
