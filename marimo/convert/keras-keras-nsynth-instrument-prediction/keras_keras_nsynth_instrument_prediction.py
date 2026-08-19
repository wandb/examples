# /// script
# dependencies = ["soundfile", "wandb"]
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
    <a href="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/keras/keras_nsynth_instrument_prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    <!--- @wandbcode{keras-nsynth} -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # NSynth Instrument Prediction using Keras

    <!--- @wandbcode{keras-nsynth} -->

    Based on the [Medium post](https://bit.ly/2UaNKQp) made by David Schwertfeger
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: wandb soundfile !pip install -Uq wandb soundfile
    return


@app.cell
def _():
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import wandb

    return tf, tfds, wandb


@app.cell
def _(wandb):
    wandb.login()
    return


@app.cell
def _(tf):
    tf.__version__
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Set Configuration Values
    """)
    return


@app.cell
def _():
    _N_CLASSES = 11
    _SAMPLE_RATE = 16000
    _DURATION = 4 #seconds?
    return


@app.cell
def _():
    _FFT_SIZE = 1024
    _HOP_SIZE = 512
    _N_MEL_BINS = 64
    _N_SPECTROGRAM_BINS = (_FFT_SIZE // 2) + 1
    _F_MIN = 0.0
    _F_MAX = _SAMPLE_RATE / 2
    return


@app.cell
def _():
    _TRAIN_DS_SIZE = 289205  # Adjust this to reduce the amount of data during training
    _TRAIN_EPOCHS = 2
    _TRAIN_BATCH_SIZE = 256
    _TRAIN_STEPS = 40000 // _TRAIN_BATCH_SIZE
    return


@app.cell
def _():
    _VAL_DS_SIZE = 12678  # Adjust this to reduce the amount of data during validation
    _VAL_BATCH_SIZE = 256
    _VAL_STEPS = _VAL_DS_SIZE / _VAL_BATCH_SIZE
    return


@app.cell
def _():
    _TEST_DS_SIZE = 4096
    _TEST_BATCH_SIZE = 256
    _TEST_STEPS = _TEST_DS_SIZE / _TEST_BATCH_SIZE
    return


@app.cell
def _():
    model_config_defaults = {
        #Dataset Specific
        "n_classes": _N_CLASSES,
        "sample_rate" : _SAMPLE_RATE,
        "duration": _DURATION,

        #Model specific
        "fft_size" : _FFT_SIZE,
        "hop_size" : _HOP_SIZE,
        "n_mels" : _N_MEL_BINS,
        "f_min" : _F_MIN,
        "f_max" : _F_MAX,

        #Training data
        "train_ds_size": _TRAIN_DS_SIZE,
        "train_epochs": _TRAIN_EPOCHS,
        "train_batch_size": _TRAIN_BATCH_SIZE,
        "train_steps": _TRAIN_STEPS,

        #Validation data
        "val_ds_size": _VAL_DS_SIZE,
        "val_batch_size": _VAL_BATCH_SIZE,
        "val_steps": _VAL_STEPS,

        #Testing data
        "test_ds_size": _TEST_DS_SIZE,
        "test_batch_size": _TEST_BATCH_SIZE,
        "test_steps": _TEST_STEPS,
    }
    return (model_config_defaults,)


@app.cell
def _(model_config_defaults, wandb):
    run = wandb.init(config = model_config_defaults, project="keras_nsynth_instrument_prediction-test")
    model_config = run.config
    return model_config, run


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load ~~and Save Data~~
    """)
    return


@app.cell
def _(tfds):
    # Load NSynth's test split as a tf.data.Dataset
    # https://www.tensorflow.org/datasets/catalog/nsynth
    (raw_train_ds, raw_validation_ds, raw_test_ds), ds_info = tfds.load(name='nsynth/full', 
                   split=['train', 'valid', "test"],
                   try_gcs=True,
                   with_info=True)
    return ds_info, raw_test_ds, raw_train_ds, raw_validation_ds


@app.cell
def _(ds_info):
    ds_info
    return


@app.cell
def _(tf):
    def prep_data(raw_ds, batch_size, data_type):
      # Let's train a model to predict the instrument family from audio
      # https://magenta.tensorflow.org/datasets/nsynth#instrument-families
      ds = raw_ds.map(lambda x: (x['audio'], x['instrument']['family']))

      # Build your input pipeline
      if data_type in ["train", "validation"]:
        prepped_ds = (ds
                      .shuffle(1000, reshuffle_each_iteration=True) #is having 2 shuffles redundant?
                      .batch(batch_size)
                      .prefetch(tf.data.AUTOTUNE)
                      .repeat()
                      )
      else:
        prepped_ds = (ds
                      .batch(batch_size)
                      .prefetch(tf.data.AUTOTUNE)
                      )
      return prepped_ds

    return (prep_data,)


@app.cell
def _(model_config, prep_data, raw_test_ds, raw_train_ds, raw_validation_ds):
    train_ds = prep_data(raw_train_ds, model_config["train_batch_size"], "train")
    validation_ds = prep_data(raw_validation_ds, model_config["val_batch_size"], "validation")
    test_ds = prep_data(raw_test_ds, model_config["test_batch_size"], "test")
    return test_ds, train_ds, validation_ds


@app.cell
def _():
    #Causes disk error
    # tf.data.experimental.save(train_ds, "./train")
    # tf.data.experimental.save(validation_ds, "./val")
    # tf.data.experimental.save(test_ds, "./test")
    return


@app.cell
def _():
    # train_data_artifact = wandb.Artifact(name="nsynth_train", type="dataset")
    # train_data_artifact.add_dir("./train")
    # run.log_artifact(train_data_artifact)
    return


@app.cell
def _():
    # val_data_artifact = wandb.Artifact(name="nsynth_val", type="dataset")
    # val_data_artifact.add_dir("./val")
    # run.log_artifact(val_data_artifact)
    return


@app.cell
def _():
    # test_data_artifact = wandb.Artifact(name="nsynth_test", type="dataset")
    # test_data_artifact.add_dir("./test")
    # run.log_artifact(test_data_artifact)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Keras Model
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Custom LogMel Layer
    """)
    return


@app.cell
def _(tf):
    class LogMelSpectrogram(tf.keras.layers.Layer):
        """Compute log-magnitude mel-scaled spectrograms."""

        def __init__(self, sample_rate, fft_size, hop_size, n_mels,
                     f_min=0.0, f_max=None, **kwargs):
            super(LogMelSpectrogram, self).__init__(**kwargs)
            self.sample_rate = sample_rate
            self.fft_size = fft_size
            self.hop_size = hop_size
            self.n_mels = n_mels
            self.f_min = f_min
            self.f_max = f_max if f_max else sample_rate / 2
            self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=self.n_mels,
                num_spectrogram_bins=fft_size // 2 + 1,
                sample_rate=self.sample_rate,
                lower_edge_hertz=self.f_min,
                upper_edge_hertz=self.f_max)

        def build(self, input_shape):
            self.non_trainable_weights.append(self.mel_filterbank)
            super(LogMelSpectrogram, self).build(input_shape)

        def call(self, waveforms):
            """Forward pass.

            Parameters
            ----------
            waveforms : tf.Tensor, shape = (None, n_samples)
                A Batch of mono waveforms.

            Returns
            -------
            log_mel_spectrograms : (tf.Tensor), shape = (None, time, freq, ch)
                The corresponding batch of log-mel-spectrograms
            """
            def _tf_log10(x):
                numerator = tf.math.log(x)
                denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
                return numerator / denominator

            def power_to_db(magnitude, amin=1e-16, top_db=80.0):
                """
                https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
                """
                ref_value = tf.reduce_max(magnitude)
                log_spec = 10.0 * _tf_log10(tf.maximum(amin, magnitude))
                log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref_value))
                log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

                return log_spec

            spectrograms = tf.signal.stft(waveforms,
                                          frame_length=self.fft_size,
                                          frame_step=self.hop_size,
                                          pad_end=False)

            magnitude_spectrograms = tf.abs(spectrograms)

            mel_spectrograms = tf.matmul(tf.square(magnitude_spectrograms),
                                         self.mel_filterbank)

            log_mel_spectrograms = power_to_db(mel_spectrograms)

            # add channel dimension
            log_mel_spectrograms = tf.expand_dims(log_mel_spectrograms, 3)

            return log_mel_spectrograms

        def get_config(self):
            config = {
                'fft_size': self.fft_size,
                'hop_size': self.hop_size,
                'n_mels': self.n_mels,
                'sample_rate': self.sample_rate,
                'f_min': self.f_min,
                'f_max': self.f_max,
            }
            config.update(super(LogMelSpectrogram, self).get_config())

            return config

    return (LogMelSpectrogram,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Use LogMel Layer in Keras Model
    """)
    return


@app.cell
def _():
    from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                         Dropout, Flatten, Input, MaxPool2D)
    from tensorflow.keras.models import Model

    return (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        Flatten,
        Input,
        MaxPool2D,
        Model,
    )


@app.cell
def _(
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LogMelSpectrogram,
    MaxPool2D,
    Model,
):
    def ConvModel(n_classes, sample_rate, duration, fft_size, hop_size, n_mels, f_min=0.0, f_max=None, **kwargs):
        n_samples = sample_rate * duration
        input_shape = (n_samples,)

        x = Input(shape=input_shape, name='input', dtype='float32')    
        y = LogMelSpectrogram(sample_rate, fft_size, hop_size, n_mels, f_min, f_max)(x)
    
        # data normalization (on frequency axis)
        y = BatchNormalization(axis=2)(y)
    
        # effectively 1D convolution, since kernel spans entire frequency-axis
        y = Conv2D(32, (3, n_mels), activation='relu')(y)
        y = BatchNormalization()(y)
        y = MaxPool2D((1, y.shape[2]))(y)

        y = Conv2D(32, (3, 1), activation='relu')(y)
        y = BatchNormalization()(y)
        y = MaxPool2D(pool_size=(2, 1))(y)

        y = Flatten()(y)
        y = Dense(64, activation='relu')(y)
        y = Dropout(0.25)(y)
        y = Dense(n_classes, activation='softmax')(y)

        return Model(inputs=x, outputs=y)

    return (ConvModel,)


@app.cell
def _(ConvModel, model_config):
    model = ConvModel(**model_config)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aside: Visualize Keras Model
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %%capture
    # !pip install visualkeras
    return


@app.cell
def _():
    import visualkeras

    return (visualkeras,)


@app.cell
def _(model, visualkeras):
    visualkeras.layered_view(model, to_file='model.png', legend=True)
    return


@app.cell
def _(run, wandb):
    run.log({"model_image": wandb.Image("model.png", caption="Visualized Keras Model")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Define Callbacks
    """)
    return


@app.cell
def _():
    from wandb.keras import WandbMetricsLogger

    return (WandbMetricsLogger,)


@app.cell
def _():
    labels = ["bass", "brass", "flute", "guitar", "keyboard", "mallet", "organ", "reed", "string", "synth_lead", "vocal"]
    return (labels,)


@app.cell
def _(WandbMetricsLogger):
    wandb_callback = WandbMetricsLogger(log_freq=2)
    return (wandb_callback,)


@app.cell
def _():
    import numpy as np
    from tensorflow.keras.callbacks import Callback
    from sklearn.metrics import accuracy_score

    return Callback, accuracy_score, np


@app.cell
def _(Callback, accuracy_score, np, run, wandb):
    class AudioPredictionCallback(Callback):
        def __init__(self, labels, prediction_data, sr):

              super(AudioPredictionCallback, self).__init__()
              self.labels = labels
              self.prediction_data = prediction_data
              self.sr = sr

        def on_epoch_end(self, epoch, logs=None):
          id_list = []
          input_audio = []
          true_index = []
          true_labels = []
          prediction_probs_list = []
          predicted_index = []
          predicted_labels = []

          for batch_x, batch_y in self.prediction_data:
            prediction_probs = self.model.predict(batch_x)
            predictions = prediction_probs.argmax(axis=1)

            for x in batch_x:
              wandb_audio = wandb.Audio(x, sample_rate=self.sr)
              input_audio.append(wandb_audio)
        
            for y in batch_y:
              true_index.append(y.numpy())
              true_labels.append(self.labels[y])

            for prediction_prob in prediction_probs:
              prediction_probs_list.append(prediction_prob)

            for pred in predictions:
              predicted_index.append(pred)
              predicted_labels.append(self.labels[pred])
      
          #All ids should match on repeate calls as the data is assumed to be never shuffled
          id_list = list(range(len(input_audio)))

          table_data = np.array([id_list, input_audio, true_labels, predicted_labels]).T
          columns = ["id", "audio", "true", "prediction"]
          prediction_table = wandb.Table(data=table_data, columns=columns)

          prediction_table_artifact = wandb.Artifact(name="audio_table", type="prediction")
          prediction_table_artifact.add(prediction_table, "audio_table")

          acc = accuracy_score(true_labels, predicted_labels)

          cm = wandb.plot.confusion_matrix(
            y_true=true_index,
            preds=predicted_index,
            class_names = self.labels,
            title="Confusion Matrix")
          pr_curve = wandb.plot.pr_curve(true_index, prediction_probs_list, labels=self.labels, title="Precision vs. Recall")

          run.log_artifact(prediction_table_artifact)
          run.log({
              "test_cm": cm,
              "test_pr_curve": pr_curve,
              "test_acc": acc
          })

    return (AudioPredictionCallback,)


@app.cell
def _(AudioPredictionCallback, labels, model_config, test_ds):
    audio_prediction_callback = AudioPredictionCallback(labels, test_ds, model_config["sample_rate"])
    return (audio_prediction_callback,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Train Model
    """)
    return


@app.cell
def _(
    audio_prediction_callback,
    model,
    model_config,
    train_ds,
    validation_ds,
    wandb_callback,
):
    model.fit(train_ds, 
              epochs=model_config["train_epochs"], 
              steps_per_epoch=model_config["train_steps"], 
              validation_data = validation_ds,
              validation_steps = model_config["val_steps"],
              callbacks=[wandb_callback, audio_prediction_callback],
              verbose="auto",
              shuffle=True #Do i need this?
              )
    return


@app.cell
def _(run):
    run.finish()
    return


if __name__ == "__main__":
    app.run()
