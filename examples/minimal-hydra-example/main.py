import hydra
import omegaconf
import tensorflow as tf
import wandb.keras


def get_model(model_cfg):

    cfg = model_cfg
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=[32, 32, 3]),
            tf.keras.layers.Conv2D(3, cfg.l1_size, activation=cfg.activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(cfg.l1_size, cfg.l2_size, activation=cfg.activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(cfg.l2_size, cfg.l3_size, activation=cfg.activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(cfg.last_size),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    return model


@hydra.main(config_path="configs/", config_name="defaults")
def run_experiment(cfg: omegaconf.DictConfig) -> None:

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)

    with wandb.init(**cfg.wandb.setup) as run:
      model = get_model(cfg.model)

      optimizer = tf.optimizers.get(cfg.optimizer)
      model.compile(loss="categorical_crossentropy", optimizer=cfg.model.optimizer)
      model.fit(
          x_train,
          y_train,
          validation_data=(x_test, y_test),
          callbacks=[wandb.keras.WandbCallback()],
      )


if __name__ == "__main__":
    run_experiment()
