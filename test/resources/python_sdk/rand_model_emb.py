#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

import numpy as np
import tensorflow as tf
import logging

from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.estimator.model_fn import ModeKeys as Modes


# https://github.com/tensorflow/tensorflow/issues/15868
# Module: test_s3_checkpoint_save_timeout
# Purpose: Train with random data and produce a large enough graph file, which should cause a
# request time out when saving to S3 on the default C++ SDK S3 request timeout configuration.
# This test script is meant to test if the patch, in the github issue above, and s3 request
# timeout environment variable were applied properly.
def model_fn(features, labels, mode, params):
    hidden_dim = params.get('hidden_dim', 512)
    classes = params.get('classes', 2)
    learning_rate = params.get('learning_rate', 0.001)
    embedding_dropout = params.get('embedding_dropout', 0.5)

    drop = (mode == Modes.TRAIN)

    word_seq = features['inputs']

    with tf.variable_scope("embedding"):
        emb_parts = _partitioned_embeddings(params)
        word_vectors = tf.nn.embedding_lookup(emb_parts, word_seq, name='word_vectors', partition_strategy='mod')

    z = tf.layers.dropout(word_vectors, rate=embedding_dropout, training=drop)
    l = LSTM(hidden_dim)(z)
    logits = Dense(classes, activation="sigmoid")(l)

    if mode in (Modes.PREDICT, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.train.get_or_create_global_step()
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.PREDICT:
        predictions = {
            'classes': predicted_indices,
            'probabilities': probabilities
        }
        export_outputs = {
            'serving_default': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        logging.info(params)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        logging.info('returned estimator spec')
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == Modes.EVAL:
        actual_index = tf.argmax(input=labels, axis=1)
        ones = tf.ones(tf.shape(actual_index), tf.int64)
        actual_endings = tf.equal(ones, actual_index)
        predicted_endings = tf.equal(ones, predicted_indices)

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(actual_index, predicted_indices),
            'precision': tf.metrics.precision(actual_endings, predicted_endings),
            'recall': tf.metrics.recall(actual_endings, predicted_endings)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _partitioned_embeddings(params):
    logging.info('initialize embedding layer')
    partitions = params.get('partitions', 10)
    embedding_dim = params.get('embedding_dim', 500)
    max_vocab_size = params.get('max_vocab_size', 134367)

    emb = np.random.rand(max_vocab_size, embedding_dim)
    end_pad = partitions - ((emb.shape[0] + 1) % partitions)
    padded = np.lib.pad(emb, ((1,end_pad), (0,0)), 'constant', constant_values=(0.0, 0.0)).astype(np.float32)
    logging.info('read in embeddings')

    constants = []
    for i in range(partitions):
        constants.append(tf.constant(padded[i::partitions]))

    logging.info('create partitioned constants')

    return constants


def serving_input_fn(params):
    inputs = tf.placeholder(tf.int32, shape=[None, 7])
    tensors = {'inputs': inputs}

    return build_raw_serving_input_receiver_fn(tensors)()


def train_input_fn(training_dir, params):
    return _input_fn(params)()


def eval_input_fn(training_dir, params):
    return _input_fn(params)()


def _input_fn(params,shuffle=False):
    window_size = params.get('windows_size', 7)
    batch_size = params.get('batch_size', 128)
    logging.info('window size = {}'.format(window_size))
    max_int = params.get('max_vocab_size', 134367) - 1

    word_ids = np.random.random_integers(0, high=max_int, size=(batch_size * 10, window_size)).astype(np.int32)
    x = {'inputs': word_ids}

    classes = np.random.random_integers(0, high=1, size=batch_size * 10).tolist()
    labels = []
    for i in range(len(classes)):
        labels.append([classes[i], abs(classes[i] - 1)])

    y_list = np.array(labels, dtype=np.float32)
    logging.info(y_list.shape)

    return tf.estimator.inputs.numpy_input_fn(
        x=x,
        y=y_list,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=shuffle)
