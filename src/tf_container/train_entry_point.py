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

import argparse
import json
import os
import subprocess
import time
from threading import Thread

import tensorflow as tf

import container_support as cs
import tf_container.run
import tf_container.serve as serve

_logger = tf_container.run.get_logger()


def _wait_until_master_is_down(master):
    while True:
        try:
            # this subprocess call is python 2/3 compatible and will throw an exception when the status code is != 0
            subprocess.check_call(['curl', '{}:2222'.format(master)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(10)
        except subprocess.CalledProcessError:
            _logger.info("master {} is down, stopping parameter server".format(master))
            return


def save_tf_config_env_var(tf_config):
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    _logger.info('----------------------TF_CONFIG--------------------------')
    _logger.info(os.environ['TF_CONFIG'])
    _logger.info('---------------------------------------------------------')


def _run_ps_server(current_host, hosts, tf_config):
    """After the training finishes, parameter servers won't stop running because server.join() has an infinite loop.
    That is a known issue: https://github.com/tensorflow/ecosystem/issues/19
    The solution below, runs the parameter server in a secondary thread while the main thread pings the master waiting
    for it to stop responding. After that, it will exit the application gracefully given that python threads cannot be
    stopped

    Args:
        current_host: (str) name of the current host
        hosts: list (str) list of all the hostnames
        tf_config: dict (str) tensorflow config map

    Returns:
    """

    def start_ps_server(current_host, hosts, tf_config):
        cluster_spec = tf.train.ClusterSpec(tf_config['cluster'])
        task_index = hosts.index(current_host)
        server = tf.train.Server(cluster_spec, job_name='ps', task_index=task_index)
        server.join()

    t = Thread(target=start_ps_server, args=(current_host, hosts, tf_config))
    t.start()


def _get_default_training_params(env):
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--training_steps', type=int, default=1000)
    my_parser.add_argument('--evaluation_steps', type=int, default=100)
    hp = env.argparse_hyperparameters(my_parser)

    return hp.training_steps, hp.evaluation_steps


def _get_master(tf_config):
    return tf_config['cluster']['master'][0][:-5]


def _get_trainer_class():
    # We used the Experiment API in tf.contrib.learn initially. It's not
    # officially supported, and it's not working properly with TF 1.6, so
    # we've switched to using tf.estimator.train_and_evaluate instead for
    # versions 1.6 and up. However, we still want to use the old API for
    # 1.4 and 1.5, since the new API isn't fully backwards compatible.

    major, minor, patch = tf.__version__.split('.')
    if major != '1':
        raise ValueError('We only support TensorFlow 1.x.y currently.')

    if minor in ['4', '5']:
        import tf_container.experiment_trainer
        return tf_container.experiment_trainer.Trainer

    import tf_container.trainer
    return tf_container.trainer.Trainer


def _get_checkpoint_dir(env):
    if 'checkpoint_path' not in env.hyperparameters:
        return env.model_dir

    checkpoint_path = env.hyperparameters['checkpoint_path']

    # If this is not part of a tuning job, then we can just use the specified checkpoint path
    if '_tuning_objective_metric' not in env.hyperparameters:
        return checkpoint_path

    job_name = env.job_name

    # If the checkpoint path already matches the format 'job_name/checkpoints', then we don't
    # need to worry about checkpoints from multiple training jobs being saved in the same location
    if job_name is None or checkpoint_path.endswith(os.path.join(job_name, 'checkpoints')):
        return checkpoint_path
    else:
        return os.path.join(checkpoint_path, job_name, 'checkpoints')


def configure_mkl():
    """Sets MKL env variables with default settings.
    More information about how to setup MLK: ENV KMP_AFFINITY= KMP_BLOCKTIME=1 KMP_SETTINGS=0"""
    os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
    os.environ['KMP_BLOCKTIME'] = '1'
    os.environ['KMP_SETTINGS'] = '0'


def train():
    env = cs.TrainingEnvironment()

    checkpoint_dir = _get_checkpoint_dir(env)
    train_steps = env.hyperparameters.get('training_steps', 1000)
    eval_steps = env.hyperparameters.get('evaluation_steps', 100)

    # https://github.com/tensorflow/tensorflow/issues/15868
    # The default request timeout for S3, within the C++ SDK, is 3 seconds, which times out when
    # saving checkpoints of larger sizes.
    os.environ['S3_REQUEST_TIMEOUT_MSEC'] = str(env.hyperparameters.get('s3_checkpoint_save_timeout', 60000))

    if env.user_script_archive.lower().startswith('s3://'):
        env.download_user_module()
    env.pip_install_requirements()

    customer_script = env.import_user_module()

    trainer_class = _get_trainer_class()
    train_wrapper = trainer_class(customer_script=customer_script,
                                  current_host=env.current_host,
                                  hosts=env.hosts,
                                  train_steps=train_steps,
                                  eval_steps=eval_steps,
                                  input_channels=env.channel_dirs,
                                  model_path=checkpoint_dir,
                                  output_path=env.output_dir,
                                  customer_params=env.hyperparameters)

    tf_config = train_wrapper.build_tf_config()

    # only creating a parameter servers for distributed runs
    if len(env.hosts) > 1:
        _run_ps_server(env.current_host, env.hosts, tf_config)

    save_tf_config_env_var(tf_config)

    configure_mkl()

    train_wrapper.train()

    # only the master should export the model at the end of the execution
    if checkpoint_dir != env.model_dir and train_wrapper.task_type == 'master' and train_wrapper.saves_training():
        serve.export_saved_model(checkpoint_dir, env.model_dir)

    if train_wrapper.task_type != 'master':
        _wait_until_master_is_down(_get_master(tf_config))
