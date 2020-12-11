# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import argparse
import errno
import os

import numpy as np
import smdistributed.dataparallel.tensorflow as dist
import tensorflow as tf
from tensorflow import keras

dist.init()

layers = tf.layers

tf.logging.set_verbosity(tf.logging.INFO)


def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope("conv_layer1"):
        h_conv1 = layers.conv2d(
            feature, 32, kernel_size=[5, 5], activation=tf.nn.relu, padding="SAME"
        )
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope("conv_layer2"):
        h_conv2 = layers.conv2d(
            h_pool1, 64, kernel_size=[5, 5], activation=tf.nn.relu, padding="SAME"
        )
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dropout(
        layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu),
        rate=0.5,
        training=mode == tf.estimator.ModeKeys.TRAIN,
    )

    # Compute logits (1 per class) and compute loss.
    logits = layers.dense(h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    return tf.argmax(logits, 1), loss


def train_input_generator(x_train, y_train, batch_size=64):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index : index + batch_size], y_train[index : index + batch_size],
            index += batch_size


def main(_):
    """Download dataset and build the model.

    Note:
    Keras automatically creates a cache directory in ~/.keras/datasets for
    storing the downloaded MNIST data. This creates a race
    condition among the workers that share the same filesystem. If the
    directory already exists by the time this worker gets around to creating
    it, ignore the resulting exception and continue.
    """
    # Training settings
    parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
    parser.add_argument('model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()

    cache_dir = os.path.join(os.path.expanduser("~"), ".keras", "datasets")
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    # Download and load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(f"MNIST-data-{dist.rank()}")

    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    x_test = np.reshape(x_test, (-1, 784)) / 255.0

    # Build model...
    with tf.name_scope("input"):
        image = tf.placeholder(tf.float32, [None, 784], name="image")
        label = tf.placeholder(tf.float32, [None], name="label")
    predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)

    lr_scaler = dist.size()

    # SMDataParallel: adjust learning rate based on lr_scaler.
    opt = tf.train.AdamOptimizer(0.001 * lr_scaler)

    # SMDataParallel: add SMDataParallel Distributed Optimizer.
    opt = dist.DistributedOptimizer(opt)

    global_step = tf.train.get_or_create_global_step()
    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [
        # SMDataParallel: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        dist.BroadcastGlobalVariablesHook(0),
        # SMDataParallel: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=20000 // dist.size()),
    ]

    if dist.rank() == 0:
        hooks.append(
            tf.train.LoggingTensorHook(tensors={"step": global_step, "loss": loss}, every_n_iter=10)
        )

    # SMDataParallel: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(dist.local_rank())

    # SMDataParallel: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = args.model_dir if dist.rank() == 0 else None
    training_batch_generator = train_input_generator(x_train, y_train, batch_size=100)
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=checkpoint_dir, hooks=hooks, config=config
    ) as mon_sess:
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = next(training_batch_generator)
            mon_sess.run(train_op, feed_dict={image: image_, label: label_})


if __name__ == "__main__":
    tf.app.run()
