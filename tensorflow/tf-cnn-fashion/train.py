#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.

Designed to show wandb integration with tensorflow.

This code was adapted from https://github.com/aymericdamien/TensorFlow-Examples/
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import wandb


def main():
    wandb.init()

    # Import Fashion MNIST data
    data = input_data.read_data_sets('data/fashion')

    categories = {
        0: 'T-shirt/Top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot'}

    flags = tf.app.flags
    flags.DEFINE_string('data_dir', '/tmp/data',
                        'Directory with the mnist data.')
    flags.DEFINE_integer('batch_size', 128, 'Batch size.')
    flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')

    flags.DEFINE_integer('num_steps', 50000,
                         'Num of batches to train.')
    flags.DEFINE_integer('display_step', 100,
                         'Steps between displaying output.')
    flags.DEFINE_integer('n_hidden_1', 256, '1st layer number of neurons.')
    flags.DEFINE_integer('n_hidden_2', 256, '2nd layer number of neurons.')
    flags.DEFINE_integer(
        'num_input', 784, 'MNIST data input (img shape: 28*28)')
    flags.DEFINE_integer('num_classes', 10, 'MNIST total classes (0-9 digits)')

    FLAGS = flags.FLAGS

    # Import all of the tensorflow flags into wandb
    wandb.config.update(FLAGS)

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # tf Graph input
    X = tf.placeholder("float", [None, FLAGS.num_input])
    Y = tf.placeholder("float", [None, FLAGS.num_classes])

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([FLAGS.num_input, FLAGS.n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([FLAGS.n_hidden_1, FLAGS.n_hidden_2])),
        'out': tf.Variable(tf.random_normal([FLAGS.n_hidden_2, FLAGS.num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([FLAGS.n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([FLAGS.n_hidden_2])),
        'out': tf.Variable(tf.random_normal([FLAGS.num_classes]))
    }

    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(1, FLAGS.num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % FLAGS.display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={
                    X: mnist.test.images,
                    Y: mnist.test.labels})

                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

                wandb.log({'acc': acc, 'loss': loss,
                           'val_acc': acc, 'val_loss': val_loss})


if __name__ == '__main__':
    main()
