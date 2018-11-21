# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from __future__ import absolute_import

import json
import logging
import os
import subprocess
import time

import sagemaker_containers.beta.framework as framework

import sagemaker_tensorflow_container.s3_utils as s3_utils


logger = logging.getLogger(__name__)


SAGEMAKER_PARAMETER_SERVER_ENABLED = 'sagemaker_parameter_server_enabled'


def _is_host_master(hosts, current_host):
    return current_host == hosts[0]


def _build_tf_config(hosts, current_host, ps_task=False):
    """Builds a dictionary containing cluster information based on number of hosts and number of
    parameter servers.

    Args:
        hosts (list[str]): List of host names in the cluster
        current_host (str): Current host name
        ps_task (bool): Set to True if this config is built for a parameter server process
            (default: False)

    Returns:
        dict[str: dict]: A dictionary describing the cluster setup for distributed training.
            For more information regarding TF_CONFIG:
            https://cloud.google.com/ml-engine/docs/tensorflow/distributed-training-details
    """
    # Assign the first host as the master. Rest of the hosts if any will be worker hosts.
    # The first ps_num hosts will also have a parameter task assign to them.
    masters = hosts[:1]
    workers = hosts[1:]
    ps = hosts if len(hosts) > 1 else None

    def host_addresses(hosts, port=2222):
        return ['{}:{}'.format(host, port) for host in hosts]

    tf_config = {
        'cluster': {
            'master': host_addresses(masters)
        },
        'environment': 'cloud'
    }

    if ps:
        tf_config['cluster']['ps'] = host_addresses(ps, port='2223')

    if workers:
        tf_config['cluster']['worker'] = host_addresses(workers)

    if ps_task:
        if ps is None:
            raise ValueError(
                'Cannot have a ps task if there are no parameter servers in the cluster')
        task_type = 'ps'
        task_index = ps.index(current_host)
    elif _is_host_master(hosts, current_host):
        task_type = 'master'
        task_index = 0
    else:
        task_type = 'worker'
        task_index = workers.index(current_host)

    tf_config['task'] = {'index': task_index, 'type': task_type}
    return tf_config


def _env_vars_with_tf_config(env, ps_task):
    env_vars = env.to_env_vars()
    env_vars['TF_CONFIG'] = json.dumps(_build_tf_config(
        hosts=env.hosts,
        current_host=env.current_host,
        ps_task=ps_task))
    return env_vars


def _run_ps(env):
    env_vars = _env_vars_with_tf_config(env, ps_task=True)
    # Parameter server processes should always run on CPU. Sets CUDA_VISIBLE_DEVICES to '-1' forces
    # TensorFlow to use CPU.
    env_vars['CUDA_VISIBLE_DEVICES'] = json.dumps(-1)
    framework.entry_point.run(env.module_dir, env.user_entry_point,
                              env.to_cmd_args(), env_vars, wait=False)


def _run_worker(env):
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        del os.environ['CUDA_VISIBLE_DEVICES']
    env_vars = _env_vars_with_tf_config(env, ps_task=False)
    framework.entry_point.run(env.module_dir, env.user_entry_point, env.to_cmd_args(), env_vars)


def _wait_until_master_is_down(master):
    while True:
        try:
            subprocess.check_call(
                ['curl', '{}:2222'.format(master)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info('master {} is still up, waiting for it to exit'.format(master))
            time.sleep(10)
        except subprocess.CalledProcessError:
            logger.info('master {} is down, stopping parameter server'.format(master))
            return


def train(env):
    """Get training job environment from env and run the training job.

    Args:
        env (sagemaker_containers.beta.framework.env.TrainingEnv): Instance of TrainingEnv class
    """
    parameter_server_enabled = env.additional_framework_parameters.get(
        SAGEMAKER_PARAMETER_SERVER_ENABLED, False)
    if len(env.hosts) > 1 and parameter_server_enabled:

        logger.info('Running distributed training job with parameter servers')
        logger.info('Launching parameter server process')
        _run_ps(env)
        logger.info('Launching worker process')
        _run_worker(env)

        if not _is_host_master(env.hosts, env.current_host):
            _wait_until_master_is_down(env.hosts[0])

    else:
        framework.entry_point.run(env.module_dir, env.user_entry_point,
                                  env.to_cmd_args(), env.to_env_vars())


def main():
    """Training entry point
    """
    hyperparameters = framework.env.read_hyperparameters()
    env = framework.training_env(hyperparameters=hyperparameters)
    s3_utils.configure(env.hyperparameters.get('model_dir'), os.environ.get('SAGEMAKER_REGION'))
    logger.setLevel(env.log_level)
    train(env)
