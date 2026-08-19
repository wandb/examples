# /// script
# dependencies = ["ddsp", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/Logging_Timbre_Transfer_with_W&B.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{tables_whalesong} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <img src="http://wandb.me/logo-im-png" width="400" alt="Weights & Biases" />

    <!--- @wandbcode{tables_whalesong} -->

    # Log timbre transfer audio experiments to W&B

    Given some input audio (a microphone recording or a file upload), resynthesize the melody of the audio as if it were played on a violin, flute, trumpet, or tenor sax. Log all your experiments to an interactive W&B Table for easy exploration and tuning.

    ### Source Colab

    This notebook is a Weights & Biases integration and wrapper around the amazing [Timbre Transfer Demo with DDSP (Differentiable Digital Signal Processing) from Tensorflow Magenta](https://colab.research.google.com/github/magenta/ddsp/blob/master/ddsp/colab/demos/timbre_transfer.ipynb)

    # Timbre Transfer with Interactive Visualization

    The notebook processes audio input with timbre transfer, resynthesizing the melody using a model pretrained for various instruments (violin, flute, trumpet, etc).

    ### [Explore an example with whale songs on W&B](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)

    <img src="https://i.imgur.com/T3vVzWZ.png" height=400 alt="interactive audio wandb Table"/></a>

    This notebook extracts features from input audio:
    * uploaded files
    * microphone recordings (to use this option, make sure to allow microphone access in your browser)
    * URLs to sound files (hardcoded for this demo, feel free to edit the variable SONG_URL)

    The available models are trained to generate audio conditioned on a time series of fundamental frequency and loudness. The input audio, synthesized song, and visualizations of the signal will be uploaded to an interactive W&B Table. You can experiment with different recordings, instruments, and various audio settings (using sliders) in this notebook. All of this configuraion will be organized alongside the song versions in one W&B project.

    <img src="https://i.imgur.com/Jo3vrGm.png" height=400 alt="interactive audio wandb Table"/></a>

    ## Additional Resources
    * Full W&B Example: [Visualizing Audio Data with W&B Tables](https://wandb.ai/stacey/cshanty/reports/Whale2Song-W-B-Tables-for-Audio--Vmlldzo4NDI3NzM)
    * [DDSP ICLR paper](https://openreview.net/forum?id=B1x1ma4tDr)
    * [Audio Examples](http://goo.gl/magenta/ddsp-examples)
    * marine mammal recordings from [Watkins Marine Mammal Sound Database](https://cis.whoi.edu/science/B/whalesounds/index.cfm), Woods Hole Oceanographic Institution
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 0.Dependencies and helper functions

    Install dependencies and wandb, and download the model. The DDSP part transfers a lot of data and _should take a minute or two according to the source colab_. Also define helper functions to process audio data.
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %tensorflow_version 2.x
    print('Installing from pip package...')
    # packages added via marimo's package management: ddsp==1.9.0 !pip install -qU ddsp==1.9.0
    # packages added via marimo's package management: wandb !pip install -qqq wandb
    import warnings
    # Ignore a bunch of deprecation warnings
    warnings.filterwarnings('ignore')
    import copy
    import os
    import time
    import crepe
    import ddsp
    import ddsp.training
    from ddsp.colab import colab_utils
    from ddsp.colab.colab_utils import audio_bytes_to_np, auto_tune, get_tuning_factor, download, play, record, specplot, upload, DEFAULT_SAMPLE_RATE
    from ddsp.training.postprocessing import detect_notes, fit_quantile_transform
    from ddsp import core
    from ddsp import spectral_ops
    import gin
    from google.colab import files
    import librosa
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import numpy as np
    import pickle
    import tensorflow.compat.v2 as tf
    import tensorflow_datasets as tfds
    import wandb
    from urllib.request import urlretrieve
    TRIM = -15
    DEFAULT_SAMPLE_RATE = spectral_ops.CREPE_SAMPLE_RATE
    print('Done!')  # 16000
    return (
        DEFAULT_SAMPLE_RATE,
        TRIM,
        audio_bytes_to_np,
        auto_tune,
        core,
        ddsp,
        detect_notes,
        files,
        fit_quantile_transform,
        get_tuning_factor,
        gin,
        librosa,
        np,
        os,
        pickle,
        play,
        plt,
        record,
        spectral_ops,
        tf,
        time,
        upload,
        urlretrieve,
        wandb,
    )


@app.cell
def _(core, ddsp, librosa, np, plt, spectral_ops, time):
    def process_song(audio, song_id, save_fig='_wave.png'):
        ddsp.spectral_ops.reset_crepe()  # Setup the session.
        _start_time = time.time()
        audio_features = ddsp.training.metrics.compute_audio_features(audio)
        audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)  # Compute features.
        audio_features_mod = None
        print('Audio features took %.1f seconds' % (time.time() - _start_time))
        TRIM = -15
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 8))
        ax[0].plot(audio_features['loudness_db'][:TRIM])
        ax[0].set_ylabel('loudness_db')
        ax[1].plot(librosa.hz_to_midi(audio_features['f0_hz'][:TRIM]))
        ax[1].set_ylabel('f0 [midi]')  # Plot Features.
        ax[2].plot(audio_features['f0_confidence'][:TRIM])
        ax[2].set_ylabel('f0 confidence')
        _ = ax[2].set_xlabel('Time step [frame]')
        save_fig_path = song_id + save_fig
        fig.savefig(save_fig_path)
        return (audio_features, save_fig_path)

    def specplot_local(audio, song_id, save_fig='_spec.png', vmin=-5, vmax=1, rotate=True, size=512 + 256, **matshow_kwargs):
        """Plot the log magnitude spectrogram of audio."""
        if len(audio.shape) == 2:
            audio = audio[0]
        logmag = spectral_ops.compute_logmag(core.tf_float32(audio), size=size)
        if rotate:
            logmag = np.rot90(logmag)
        save_fig_path = song_id + save_fig
        plt.imsave(save_fig_path, logmag, vmin=vmin, vmax=vmax, cmap=plt.cm.magma)
        return save_fig_path  # If batched, take first element.  #plt.xticks([])  #plt.yticks([])  #plt.xlabel('Time')  #plt.ylabel('Frequency')

    return process_song, specplot_local


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.Initialize and login to W&B
    """)
    return


@app.cell
def _():
    WANDB_PROJECT = "timbre_demo"
    return (WANDB_PROJECT,)


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2.Song setup (run for every new song)
    """)
    return


@app.cell
def _(WANDB_PROJECT, np, wandb):
    wandb.init(project=WANDB_PROJECT)

    # generate one random song id, feel free to replace
    SONG_ID = str(np.random.choice(1000, 1)[0])

    # hardcoded to a favorite marine mammal melody, feel free to replace
    SONG_URL = "https://whoicf2.whoi.edu/science/B/whalesounds/WhaleSounds/6301900Y.wav"
    return SONG_ID, SONG_URL


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3.Audio input: Record, upload, or URL

    You have several options for audio input:
    1. **Record** audio from your microphone (NOTE: allow microphone access in your browser to do this)
    2. **Upload** audio from a file (.mp3 or .wav)
    3. **Download a URL** (this is hardcoded for the demo, and you can change SONG_URL in Step 2 to edit this)

    Additional notes:
    * Audio should be monophonic (single instrument / voice)
    * Extracts fundmanetal frequency (f0) and loudness features.
    """)
    return


@app.cell
def _(SONG_URL, audio_bytes_to_np, np, record, upload, urlretrieve):
    record_or_upload = "URL"  #@param ["Record", "Upload (.mp3 or .wav)", "URL"]

    record_seconds =     5#@param {type:"number", min:1, max:10, step:1}

    if record_or_upload == "Record":
      audio = record(seconds=record_seconds)
    elif record_or_upload == "URL":
      filename = SONG_URL.strip().split('/')[-1]
      urlretrieve(SONG_URL, filename)
      wav_bytes = open(filename, "rb").read()
      audio = audio_bytes_to_np(wav_bytes)
    else:
      # Load audio sample here (.mp3 or .wav3 file)
      # Just use the first file.
      filenames, audios = upload()
      audio = audios[0]

    audio = audio[np.newaxis, :]
    return (audio,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Upload sample song to W&B

    You will see a URL to your W&B run, which will show a playable version of the song and some audio visualizations in a new Table.
    """)
    return


@app.cell
def _(
    DEFAULT_SAMPLE_RATE,
    SONG_ID,
    audio,
    np,
    process_song,
    specplot_local,
    wandb,
):
    columns = ['id', 'orig_song', 'orig_plot', 'orig_spec']
    audio_features, orig_waveplot = process_song(audio, SONG_ID, '_orig_plot.png')
    orig_specplot = specplot_local(audio, SONG_ID, '_orig_spec.png')
    orig_song = wandb.Audio(np.squeeze(audio), sample_rate=DEFAULT_SAMPLE_RATE)
    _data = [[SONG_ID, orig_song, wandb.Image(orig_waveplot), wandb.Image(orig_specplot)]]
    _table = wandb.Table(data=_data, columns=columns)
    wandb.run.log({'sample_song': _table})
    wandb.run.finish()
    return (audio_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4.Synthetic output (run for every new song)
    """)
    return


@app.cell
def _(
    audio,
    audio_features,
    ddsp,
    files,
    gin,
    os,
    pickle,
    subprocess,
    tf,
    time,
):
    #@title Choose an instrument (load a model)
    #@markdown Run for every new audio input
    model = 'Tenor_Saxophone'  #@param ['Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone', 'Upload your own (checkpoint folder as .zip)']
    MODEL = model

    def find_model_dir(dir_name):
        for root, dirs, filenames in os.walk(dir_name):  # Iterate through directories until model directory is found
            for filename in filenames:
                if filename.endswith('.gin') and (not filename.startswith('.')):
                    model_dir = root
                    break
        return model_dir
    if model in ('Violin', 'Flute', 'Flute2', 'Trumpet', 'Tenor_Saxophone'):
        PRETRAINED_DIR = '/content/pretrained'
        subprocess.call(['rm', '-r', '$PRETRAINED_DIR', '&>', '/dev/null'])
        subprocess.call(['mkdir', '$PRETRAINED_DIR', '&>', '/dev/null'])  # Pretrained models.
        GCS_CKPT_DIR = 'gs://ddsp/models/timbre_transfer_colab/2021-07-08'
        model_dir = os.path.join(GCS_CKPT_DIR, 'solo_%s_ckpt' % model.lower())  # Copy over from gs:// for faster loading.
        subprocess.call(['gsutil', 'cp', '$model_dir/*', '$PRETRAINED_DIR', '&>', '/dev/null'])  #! rm -r $PRETRAINED_DIR &> /dev/null
        model_dir = PRETRAINED_DIR
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')  #! mkdir $PRETRAINED_DIR &> /dev/null
    else:
        UPLOAD_DIR = '/content/uploaded'
        subprocess.call(['mkdir', '$UPLOAD_DIR'])
        uploaded_files = files.upload()
        for fnames in uploaded_files.keys():  #! gsutil cp $model_dir/* $PRETRAINED_DIR &> /dev/null
            print('Unzipping... {}'.format(fnames))
            subprocess.call(['unzip', '-o', '/content/$fnames', '-d', '$UPLOAD_DIR', '&>', '/dev/null'])
        model_dir = find_model_dir(UPLOAD_DIR)
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')
    DATASET_STATS = None
    dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')  # User models.
    print(f'Loading dataset statistics from {dataset_stats_file}')
    try:  #! mkdir $UPLOAD_DIR
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                DATASET_STATS = pickle.load(f)
    except Exception as err:
        print('Loading dataset statistics from pickle failed: {}.'.format(err))
    with gin.unlock_config():  #! unzip -o "/content/$fnames" -d $UPLOAD_DIR &> /dev/null
        gin.parse_config_file(gin_file, skip_unknown=True)
    ckpt_files = [f for f in tf.io.gfile.listdir(model_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(model_dir, ckpt_name)
    time_steps_train = gin.query_parameter('F0LoudnessPreprocessor.time_steps')
    # Load the dataset statistics.
    n_samples_train = gin.query_parameter('Harmonic.n_samples')
    hop_size = int(n_samples_train / time_steps_train)
    time_steps = int(audio.shape[1] / hop_size)
    n_samples = time_steps * hop_size
    gin_params = ['Harmonic.n_samples = {}'.format(n_samples), 'FilteredNoise.n_samples = {}'.format(n_samples), 'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps), 'oscillator_bank.use_angular_cumsum = True']
    with gin.unlock_config():
        gin.parse_config(gin_params)
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:, :n_samples]
    model = ddsp.training.models.Autoencoder()
    # Parse gin config,
    model.restore(ckpt)
    _start_time = time.time()
    _ = model(audio_features, training=False)
    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`.
    # Ensure dimensions and sampling rates are equal
    # print("===Trained model===")
    # print("Time Steps", time_steps_train)
    # print("Samples", n_samples_train)
    # print("Hop Size", hop_size)
    # print("\n===Resynthesis===")
    # print("Time Steps", time_steps)
    # print("Samples", n_samples)
    # print('')
    # Trim all input vectors to correct lengths 
    # Set up the model just to predict audio given new conditioning
    # Build model by running a batch through it.
    print('Restoring model took %.1f seconds' % (time.time() - _start_time))  # Avoids cumsum accumulation errors.
    return DATASET_STATS, MODEL, model


@app.cell
def _(
    DATASET_STATS,
    MODEL,
    SONG_ID,
    TRIM,
    audio_features,
    auto_tune,
    ddsp,
    detect_notes,
    fit_quantile_transform,
    get_tuning_factor,
    librosa,
    np,
    plt,
    wandb,
):
    #@title Modify conditioning
    threshold = 1
    #@markdown These models were not explicitly trained to perform timbre transfer, so they may sound unnatural if the incoming loudness and frequencies are very different then the training data (which will always be somewhat true). 
    ADJUST = True
    quiet = 20
    #@markdown ## Note Detection
    autotune = 0
    #@markdown You can leave this at 1.0 for most cases
    pitch_shift = 0  #@param {type:"slider", min: 0.0, max:2.0, step:0.01}
    loudness_shift = 0
    SONG_CFG = {'threshold': threshold, 'adjust': ADJUST, 'quiet': quiet, 'autotune': autotune, 'pitch_shift': pitch_shift, 'loudness_shift': loudness_shift}
    #@markdown ## Automatic
    audio_features_mod = {k: v.copy() for k, v in audio_features.items()}
      #@param{type:"boolean"}
    def shift_ld(audio_features, ld_shift=0.0):
    #@markdown Quiet parts without notes detected (dB)
        """Shift loudness by a number of ocatves."""  #@param {type:"slider", min: 0, max:60, step:1}
        audio_features['loudness_db'] = audio_features['loudness_db'] + ld_shift
    #@markdown Force pitch to nearest note (amount)
        return audio_features  #@param {type:"slider", min: 0.0, max:1.0, step:0.1}

    #@markdown ## Manual
    def shift_f0(audio_features, pitch_shift=0.0):
        """Shift f0 by a number of ocatves."""
    #@markdown Shift the pitch (octaves)
        audio_features['f0_hz'] = audio_features['f0_hz'] * 2.0 ** pitch_shift  #@param {type:"slider", min:-2, max:2, step:1}
        audio_features['f0_hz'] = np.clip(audio_features['f0_hz'], 0.0, librosa.midi_to_hz(110.0))
    #@markdown Adjsut the overall loudness (dB)
        return audio_features  #@param {type:"slider", min:-20, max:20, step:1}
    mask_on = None
    # save settings
    if ADJUST and DATASET_STATS is not None:
        mask_on, note_on_value = detect_notes(audio_features['loudness_db'], audio_features['f0_confidence'], threshold)
        if np.any(mask_on):
            target_mean_pitch = DATASET_STATS['mean_pitch']
            pitch = ddsp.core.hz_to_midi(audio_features['f0_hz'])
            mean_pitch = np.mean(pitch[mask_on])
            p_diff = target_mean_pitch - mean_pitch
            p_diff_octave = p_diff / 12.0
            round_fn = np.floor if p_diff_octave > 1.5 else np.ceil
            p_diff_octave = round_fn(p_diff_octave)
            audio_features_mod = shift_f0(audio_features_mod, p_diff_octave)
            _, loudness_norm = fit_quantile_transform(audio_features['loudness_db'], mask_on, inv_quantile=DATASET_STATS['quantile_transform'])
    ## Helper functions.
            mask_off = np.logical_not(mask_on)
            loudness_norm[mask_off] = loudness_norm[mask_off] - quiet * (1.0 - note_on_value[mask_off][:, np.newaxis])
            loudness_norm = np.reshape(loudness_norm, audio_features['loudness_db'].shape)
            audio_features_mod['loudness_db'] = loudness_norm
            if autotune:
                f0_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod['f0_hz']))
                tuning_factor = get_tuning_factor(f0_midi, audio_features_mod['f0_confidence'], mask_on)
                f0_midi_at = auto_tune(f0_midi, tuning_factor, mask_on, amount=autotune)
                audio_features_mod['f0_hz'] = ddsp.core.midi_to_hz(f0_midi_at)
        else:
            print('\nSkipping auto-adjust (no notes detected or ADJUST box empty).')
    else:
        print('\nSkipping auto-adujst (box not checked or no dataset statistics found).')
    audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
    audio_features_mod = shift_f0(audio_features_mod, pitch_shift)
    has_mask = int(mask_on is not None)
    n_plots = 3 if has_mask else 2
    fig, axes = plt.subplots(nrows=n_plots, ncols=1, sharex=True, figsize=(2 * n_plots, 8))
    if has_mask:  # Detect sections that are "on".
        ax = axes[0]
        ax.plot(np.ones_like(mask_on[:TRIM]) * threshold, 'k:')
        ax.plot(note_on_value[:TRIM])
        ax.plot(mask_on[:TRIM])
        ax.set_ylabel('Note-on Mask')
        ax.set_xlabel('Time step [frame]')  # Shift the pitch register.
        ax.legend(['Threshold', 'Likelihood', 'Mask'])
    ax = axes[0 + has_mask]
    ax.plot(audio_features['loudness_db'][:TRIM])
    ax.plot(audio_features_mod['loudness_db'][:TRIM])
    ax.set_ylabel('loudness_db')
    ax.legend(['Original', 'Adjusted'])
    ax = axes[1 + has_mask]
    ax.plot(librosa.hz_to_midi(audio_features['f0_hz'][:TRIM]))
    ax.plot(librosa.hz_to_midi(audio_features_mod['f0_hz'][:TRIM]))
    ax.set_ylabel('f0 [midi]')
    _ = ax.legend(['Original', 'Adjusted'])  # Quantile shift the note_on parts.
    final_wave_plot = SONG_ID + '_final.png'
    fig.savefig(final_wave_plot)
    SONG_CFG['final_wave_plot'] = wandb.Image(final_wave_plot)
    # Manual Shifts.
    # Plot Features.
    SONG_CFG['instrument'] = MODEL  # Turn down the note_off parts.  # Auto-tune.
    return SONG_CFG, audio_features_mod


@app.cell
def _(
    DEFAULT_SAMPLE_RATE,
    SONG_CFG,
    SONG_ID,
    audio,
    audio_features,
    audio_features_mod,
    model,
    np,
    play,
    specplot_local,
    time,
    wandb,
):
    af = audio_features if audio_features_mod is None else audio_features_mod
    _start_time = time.time()
    outputs = model(af, training=False)
    audio_gen = model.get_audio_from_outputs(outputs)
    print('Prediction took %.1f seconds' % (time.time() - _start_time))
    print('Original')
    play(audio)
    orig_song_1 = wandb.Audio(np.squeeze(audio), sample_rate=DEFAULT_SAMPLE_RATE)
    SONG_CFG['orig_song'] = orig_song_1
    print('Resynthesis')
    play(audio_gen)
    synth_song = wandb.Audio(np.squeeze(audio_gen), sample_rate=DEFAULT_SAMPLE_RATE)
    SONG_CFG['synth_song'] = synth_song
    final_spec_plot = specplot_local(audio_gen, SONG_ID, fig_name='_final_spec.png')
    SONG_CFG['final_spec'] = wandb.Image(final_spec_plot)
    return orig_song_1, synth_song


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5.Upload synthesized song to W&B
    """)
    return


@app.cell
def _(SONG_CFG, SONG_ID, WANDB_PROJECT, orig_song_1, synth_song, wandb):
    wandb.init(project=WANDB_PROJECT)
    output_columns = ['id', 'orig_song', 'synth_song', 'synth_waves', 'synth_spec', 'instrument', 'threshold', 'adjust', 'quiet', 'autotune', 'pitch_shift', 'loudness_shift']
    s = SONG_CFG
    _data = [[SONG_ID, orig_song_1, synth_song, s['final_wave_plot'], s['final_spec'], s['instrument'], s['threshold'], s['adjust'], s['quiet'], s['autotune'], s['pitch_shift'], s['loudness_shift']]]
    _table = wandb.Table(data=_data, columns=output_columns)
    wandb.run.log({'synth_song': _table})
    wandb.run.finish()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Source colab: Timbre transfer demo from Magenta

    This colab relies substantially on the following Timbre Transfer demo:

    ##### Copyright 2021 Google LLC.

    Licensed under the Apache License, Version 2.0 (the "License");
    """)
    return


@app.cell
def _():
    # Copyright 2021 Google LLC. All Rights Reserved.

    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at

    #     http://www.apache.org/licenses/LICENSE-2.0

    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    # ==============================================================================
    return


if __name__ == "__main__":
    app.run()
