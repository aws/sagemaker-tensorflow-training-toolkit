import logging
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)


def get_logger():
    _logger = logging.getLogger("tf_container")
    return _logger


logger = get_logger()
