# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
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

import os
import sys

from mock import MagicMock, patch
import pytest
from sagemaker_containers.beta.framework import runner
import tensorflow as tf

from sagemaker_tensorflow_container import training

MODULE_DIR = 's3://my/bucket'
MODULE_NAME = 'script_name'
LOG_LEVEL = 'Debug'
HOST1 = 'host1'
HOST2 = 'host2'
HOST_LIST = [HOST1, HOST2]
CURRENT_HOST = HOST1
CMD_ARGS = {'some_key': 'some_value'}
CLUSTER_WITH_PS = {
    'master': ['{}:2222'.format(HOST1)],
    'worker': ['{}:2222'.format(HOST2)],
    'ps': ['{}:2223'.format(HOST1), '{}:2223'.format(HOST2)]
}
MASTER_TASK = {'index': 0, 'type': 'master'}
WORKER_TASK = {'index': 0, 'type': 'worker'}
PS_TASK_1 = {'index': 0, 'type': 'ps'}
PS_TASK_2 = {'index': 1, 'type': 'ps'}
MODEL_DIR = 's3://bucket/prefix'
REGION = 'us-west-2'


@pytest.fixture
def distributed_training_env():
    env = simple_training_env()

    env.hosts = HOST_LIST
    env.additional_framework_parameters = {
        training.SAGEMAKER_PARAMETER_SERVER_ENABLED: True
    }
    return env


@pytest.fixture
def single_machine_training_env():
    return simple_training_env()


def simple_training_env():
    env = MagicMock()
    env.module_dir = MODULE_DIR
    env.user_entry_point = MODULE_NAME
    env.hyperparameters = {'model_dir': MODEL_DIR}
    env.log_level = LOG_LEVEL
    env.additional_framework_parameters = {}
    env.hosts = CURRENT_HOST
    env.current_host = CURRENT_HOST
    env.to_env_vars = lambda: {}
    return env


def test_is_host_master():
    assert training._is_host_master(HOST_LIST, CURRENT_HOST) is True
    assert training._is_host_master(HOST_LIST, 'host2') is False
    assert training._is_host_master(HOST_LIST, 'somehost') is False


@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_single_machine(run_module, single_machine_training_env):
    training.train(single_machine_training_env)
    run_module.assert_called_with(MODULE_DIR, MODULE_NAME,
                                  single_machine_training_env.to_cmd_args(),
                                  single_machine_training_env.to_env_vars(),
                                  runner=runner.ProcessRunnerType)


@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_train_horovod(run_module, single_machine_training_env):
    single_machine_training_env.additional_framework_parameters['sagemaker_mpi_enabled'] = True

    training.train(single_machine_training_env)
    run_module.assert_called_with(MODULE_DIR, MODULE_NAME,
                                  single_machine_training_env.to_cmd_args(),
                                  single_machine_training_env.to_env_vars(),
                                  runner=runner.MPIRunnerType)


@pytest.mark.skipif(sys.version_info.major != 3,
                    reason="Skip this for python 2 because of dict key order mismatch")
@patch('tensorflow.train.ClusterSpec')
@patch('tensorflow.train.Server')
@patch('sagemaker_containers.beta.framework.entry_point.run')
@patch('threading.Thread', lambda target: target())
@patch('time.sleep', MagicMock())
def test_train_distributed_master(run, tf_server, cluster_spec, distributed_training_env):
    training.train(distributed_training_env)

    cluster_spec.assert_called_with({'worker': ['host2:2222'],
                                     'master': ['host1:2222'],
                                     'ps': ['host1:2223', 'host2:2223']})

    tf_server.assert_called_with(
        cluster_spec(), job_name='ps', task_index=0, config=tf.ConfigProto(device_count={'GPU': 0})
    )
    tf_server().join.assert_called_with()

    tf_config = '{"cluster": {' \
                '"master": ["host1:2222"], ' \
                '"ps": ["host1:2223", "host2:2223"], ' \
                '"worker": ["host2:2222"]}, ' \
                '"environment": "cloud", ' \
                '"task": {"index": 0, "type": "master"}}'

    run.assert_called_with('s3://my/bucket', 'script_name',
                           distributed_training_env.to_cmd_args(),
                           {'TF_CONFIG': tf_config})


@pytest.mark.skipif(sys.version_info.major != 3,
                    reason="Skip this for python 2 because of dict key order mismatch")
@patch('tensorflow.train.ClusterSpec')
@patch('tensorflow.train.Server')
@patch('sagemaker_containers.beta.framework.entry_point.run')
@patch('time.sleep', MagicMock())
def test_train_distributed_worker(run, tf_server, cluster_spec, distributed_training_env):
    distributed_training_env.current_host = HOST2

    training.train(distributed_training_env)

    cluster_spec.assert_called_with({'worker': ['host2:2222'],
                                     'master': ['host1:2222'],
                                     'ps': ['host1:2223', 'host2:2223']})

    tf_server.assert_called_with(
        cluster_spec(), job_name='ps', task_index=1, config=tf.ConfigProto(device_count={'GPU': 0})
    )
    tf_server().join.assert_called_with()

    tf_config = '{"cluster": {' \
                '"master": ["host1:2222"], ' \
                '"ps": ["host1:2223", "host2:2223"], ' \
                '"worker": ["host2:2222"]}, ' \
                '"environment": "cloud", ' \
                '"task": {"index": 0, "type": "worker"}}'

    run.assert_called_with('s3://my/bucket', 'script_name',
                           distributed_training_env.to_cmd_args(),
                           {'TF_CONFIG': tf_config})


@patch('sagemaker_containers.beta.framework.entry_point.run')
def test_train_distributed_no_ps(run, distributed_training_env):
    distributed_training_env.additional_framework_parameters[
        training.SAGEMAKER_PARAMETER_SERVER_ENABLED] = False
    distributed_training_env.current_host = HOST2
    training.train(distributed_training_env)

    run.assert_called_with(MODULE_DIR, MODULE_NAME, distributed_training_env.to_cmd_args(),
                           distributed_training_env.to_env_vars(), runner=runner.ProcessRunnerType)


def test_build_tf_config():
    assert training._build_tf_config(HOST_LIST, HOST1) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': MASTER_TASK
    }
    assert training._build_tf_config(HOST_LIST, HOST1, ps_task=True) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': PS_TASK_1
    }
    assert training._build_tf_config(HOST_LIST, HOST2) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': WORKER_TASK
    }
    assert training._build_tf_config(HOST_LIST, HOST2, ps_task=True) == {
        'cluster': CLUSTER_WITH_PS,
        'environment': 'cloud',
        'task': PS_TASK_2}


def test_build_tf_config_error():
    with pytest.raises(ValueError) as error:
        training._build_tf_config([HOST1], HOST1, ps_task=True)
    assert 'Cannot have a ps task if there are no parameter servers in the cluster' in str(error)


@patch('sagemaker_tensorflow_container.training.train')
@patch('logging.Logger.setLevel')
@patch('sagemaker_containers.beta.framework.training_env')
@patch('sagemaker_containers.beta.framework.env.read_hyperparameters', return_value={})
@patch('sagemaker_tensorflow_container.s3_utils.configure')
def test_main(configure_s3_env, read_hyperparameters, training_env,
              set_level, train, single_machine_training_env):
    training_env.return_value = single_machine_training_env
    os.environ['SAGEMAKER_REGION'] = REGION
    training.main()
    read_hyperparameters.assert_called_once_with()
    training_env.assert_called_once_with(hyperparameters={})
    set_level.assert_called_once_with(LOG_LEVEL)
    train.assert_called_once_with(single_machine_training_env)
    configure_s3_env.assert_called_once()
