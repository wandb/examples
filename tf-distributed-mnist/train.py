# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple MNIST classifier which displays summaries in TensorBoard.
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import ast
import json
import wandb

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def train():
    tf_config_json = os.environ.get("TF_CONFIG", "{}")
    tf_config = json.loads(tf_config_json)

    task = tf_config.get("task", {})
    cluster_spec = tf_config.get("cluster", {})
    cluster_spec_object = tf.train.ClusterSpec(cluster_spec)
    job_name = task["type"]
    task_id = task["index"]
    server_def = tf.train.ServerDef(
        cluster=cluster_spec_object.as_cluster_def(),
        protocol="grpc",
        job_name=job_name,
        task_index=task_id)
    server = tf.train.Server(server_def)

    is_chief = (job_name == 'worker') and (task_id == 0)
    if job_name == 'ps':
        server.join()

    if is_chief:
        print("Worker %d: Initializing session..." % task_id)
        tf.reset_default_graph()
    else:
        print("Worker %d: Waiting for session to be initialized..." % task_id)

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir,
                                      one_hot=True,
                                      fake_data=FLAGS.fake_data)

    # Create a multilayer model.

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:{0}/task:{1}".format(job_name, task_id),
            cluster=cluster_spec)):
        # with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
        # worker_device="/job:{0}/task:{1}".format(job_name,task_id),
        # cluster=cluster_spec)):

        # count the number of updates
        global_step = tf.get_variable(
            'global_step',
            [],
            initializer=tf.constant_initializer(0),
            trainable=False)

        # Input placeholders
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
            tf.summary.image('input', image_shaped_input, 10)

        # We can't initialize these variables to 0 - the network will get stuck.
        def weight_variable(shape):
            """Create a weight variable with appropriate initialization."""
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            """Create a bias variable with appropriate initialization."""
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
                tf.summary.scalar('stddev', stddev)
                tf.summary.scalar('max', tf.reduce_max(var))
                tf.summary.scalar('min', tf.reduce_min(var))
                tf.summary.histogram('histogram', var)

        def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
            """Reusable code for making a simple neural net layer.
            It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
            It also sets up name scoping so that the resultant graph is easy to read,
            and adds a number of summary ops.
            """
            # Adding a name scope ensures logical grouping of the layers in the graph.
            with tf.name_scope(layer_name):
                # This Variable will hold the state of the weights for the layer
                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights)
                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases)
                with tf.name_scope('Wx_plus_b'):
                    preactivate = tf.matmul(input_tensor, weights) + biases
                    tf.summary.histogram('pre_activations', preactivate)
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations

        hidden1 = nn_layer(x, 784, 500, 'layer1')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder_with_default(1.0, shape=())
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(hidden1, keep_prob)

        # Do not apply softmax activation yet, see below.
        y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

        with tf.name_scope('cross_entropy'):
            # The raw formulation of cross-entropy,
            #
            # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
            #                               reduction_indices=[1]))
            #
            # can be numerically unstable.
            #
            # So here we use tf.nn.softmax_cross_entropy_with_logits on the
            # raw outputs of the nn_layer above, and then average across
            # the batch.
            #diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
            # with tf.name_scope('total'):
                #cross_entropy = tf.reduce_mean(diff)
            logits = tf.nn.softmax(y, name='logits')
            cross_entropy = - \
                tf.reduce_sum(y_ * tf.log(logits), name='cross_entropy')
            tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(
                    tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to
        # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()

        init_op = tf.global_variables_initializer()

    def feed_dict(train):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train or FLAGS.fake_data:
            xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
            k = FLAGS.dropout
        else:
            xs, ys = mnist.test.images, mnist.test.labels
            k = 1.0
        return {x: xs, y_: ys, keep_prob: k}

    sv = tf.train.Supervisor(is_chief=is_chief,
                             global_step=global_step,
                             init_op=init_op,
                             logdir=FLAGS.logdir)
    # sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True,
    #                               device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.worker_index])

    with sv.prepare_or_wait_for_session(server.target) as sess:
        train_writer = tf.summary.FileWriter(
            FLAGS.logdir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.logdir + '/test')
        # Train the model, and also write summaries.
        # Every 10th step, measure test-set accuracy, and write test summaries
        # All other steps, run train_step on training data, & add training summaries

        for i in range(FLAGS.max_steps):
            if i % 10 == 0:  # Record summaries and test-set accuracy
                summary, acc = sess.run(
                    [merged, accuracy], feed_dict=feed_dict(False))
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))
                if i % 100 == 0:
                    wandb.tensorflow.log(summary)
            else:  # Record train set summaries, and train
                if i % 100 == 99:  # Record execution stats
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step],
                                          feed_dict=feed_dict(True),
                                          options=run_options,
                                          run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    train_writer.add_summary(summary, i)
                    print('Adding run metadata for', i)
                    wandb.tensorflow.log(summary)
                else:  # Record a summary
                    summary, _ = sess.run(
                        [merged, train_step], feed_dict=feed_dict(True))
                    train_writer.add_summary(summary, i)
        train_writer.close()
        test_writer.close()


def main(_):
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                        default=False,
                        help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/input_data'),
        help='Directory for storing input data')
    parser.add_argument(
        '--logdir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                             'tensorflow/logs'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    wandb.init(project="distributed-mnist", config=FLAGS)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
