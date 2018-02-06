import logging
import traceback
import os
import tensorflow as tf


def get_logger():
    _logger = logging.getLogger('tensorflow')
    _logger.propagate = False
    tf.logging.set_verbosity(tf.logging.INFO)
    return _logger


logger = get_logger()


def train_and_log_exceptions(test_wrapper, ouput_path):
    try:
        logger.info("going to training")
        test_wrapper.train()

        # Write out the success file
        logger.info("writing success training")
        with open(os.path.join(ouput_path, 'success'), 'w') as s:
            s.write('Done')
    except Exception as e:
        trc = traceback.format_exc()

        message = """
Exception during training:
{}
{}
""".format(e, trc)

        # Write out an error file
        logger.error("writing error")
        logger.error("error file is".format(os.path.join(ouput_path, 'failure')))

        logger.error(message)
        with open(os.path.join(ouput_path, 'failure'), 'w') as s:
            s.write(message)
        raise e
