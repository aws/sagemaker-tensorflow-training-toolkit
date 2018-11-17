from __future__ import print_function

import tensorflow as tf

INPUT_TENSOR_NAME = 'inputs'
CHANNEL_BASEDIR = '/opt/ml/input/data/'


def estimator_fn(run_config, hyperparameters):
    feature_columns = [tf.feature_column.embedding_column(
        tf.feature_column.categorical_column_with_vocabulary_file(
            INPUT_TENSOR_NAME, CHANNEL_BASEDIR + "vocabulary.txt"),
        combiner="mean",
        dimension=32)]

    return tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                      hidden_units=[10],
                                      n_classes=2,
                                      config=run_config)


def train_input_fn(training_dir, hyperparameters):
    return _generate_input_fn("train")
    
    
def eval_input_fn(training_dir, hyperparameters):
    return _generate_input_fn("test")


def serving_input_fn(hyperparameters):
    feature_spec = {INPUT_TENSOR_NAME: tf.VarLenFeature(dtype=tf.string)}
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


def _generate_input_fn(data_dir):
    dataset = tf.data.Dataset.from_generator(lambda: _input_data_reader(data_dir), (tf.string, tf.int32)).repeat(3)
    doc, label = dataset.make_one_shot_iterator().get_next()
    return {INPUT_TENSOR_NAME: doc}, label


def _input_data_reader(subdir):
    data_file = CHANNEL_BASEDIR + subdir + ".txt"
    with open(data_file, "rt") as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            label, doc = line.strip().split("\t")
            yield [doc.lower().split(" ")], [[int(label)]]
