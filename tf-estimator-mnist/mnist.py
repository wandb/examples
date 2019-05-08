import tensorflow as tf
import numpy as np
import argparse
import wandb

mnist = tf.contrib.learn.datasets.load_dataset("mnist")


def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)


def main():
    # Specify feature
    feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]

    # Build 2 layer DNN classifier
    # NOTE: We change the summary logging frequency to be every epoch with save_summary_steps
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[256, 32],
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=10,
        dropout=0.1,
        config=tf.estimator.RunConfig(
            save_summary_steps=mnist.train.images.shape[0] / wandb.config.batch_size)
    )

    # Turn on logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": input(mnist.train)[0]},
        y=input(mnist.train)[1],
        num_epochs=None,
        batch_size=wandb.config.batch_size,
        shuffle=True,
    )

    # Train the classifier
    classifier.train(input_fn=train_input_fn, steps=wandb.config.max_steps)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": input(mnist.test)[0]},
        y=input(mnist.test)[1],
        num_epochs=1,
        shuffle=False
    )

    # Evaluate accuracy
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
    print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Number of steps to run trainer.')
    args = parser.parse_args()
    wandb.init(tensorboard=True)
    wandb.config.batch_size = 256
    wandb.config.max_steps = args.max_steps
    main()
