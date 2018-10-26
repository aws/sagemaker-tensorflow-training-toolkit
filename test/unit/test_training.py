# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import json

from mock import MagicMock, patch
import pytest

from sagemaker_tensorflow_container import training

MODULE_DIR = 's3://my/bucket'
MODULE_NAME = 'script_name'
LOG_LEVEL = 'Debug'
HOST1 = 'host1'
HOST2 = 'host2'
HOST_LIST = [HOST1, HOST2]
CURRENT_HOST = HOST1
PS_NUM = 2
CMD_ARGS = {'some_key': 'some_value'}
CLUSTER = {
    "master": ["{}:2222".format(HOST1)],
    "worker": ["{}:2222".format(HOST2)]
}
CLUSTER_WITH_PS = {
    "master": ["{}:2222".format(HOST1)],
    "worker": ["{}:2222".format(HOST2)],
    "ps": ["{}:2223".format(HOST1), "{}:2223".format(HOST2)]
}
CLUSTER_WITH_1_PS = {
    "master": ["{}:2222".format(HOST1)],
    "worker": ["{}:2222".format(HOST2)],
    "ps": ["{}:2223".format(HOST1)]
}
MASTER_TASK = {"index": 0, "type": "master"}
WORKER_TASK = {"index": 0, "type": "worker"}
PS_TASK_1 = {"index": 0, "type": "ps"}
PS_TASK_2 = {"index": 1, "type": "ps"}


@pytest.fixture
def distributed_training_env():
    env = MagicMock()

    env.module_dir = MODULE_DIR
    env.module_name = MODULE_NAME
    env.hyperparameters = {}
    env.log_level = LOG_LEVEL
    env.hosts = HOST_LIST
    env.current_host = CURRENT_HOST
    env.additional_framework_parameters = {
        training.SAGEMAKER_PARAMETER_SERVER_NUM: PS_NUM
    }

    return env


@pytest.fixture
def single_machine_training_env():
    env = MagicMock()

    env.module_dir = MODULE_DIR
    env.module_name = MODULE_NAME
    env.hyperparameters = {}
    env.log_level = LOG_LEVEL

    return env


def test_is_host_master():
    assert training._is_host_master(HOST_LIST, CURRENT_HOST) is True
    assert training._is_host_master(HOST_LIST, 'host2') is False
    assert training._is_host_master(HOST_LIST, 'somehost') is False


def test_should_run_parameter_server():
    env = MagicMock()
    env.additional_framework_parameters = {}
    assert training._should_run_parameter_server(env) is False
    env.additional_framework_parameters[training.SAGEMAKER_PARAMETER_SERVER_NUM] = 2
    assert training._should_run_parameter_server(env) is True


def test_get_parameter_server_num():
    env = MagicMock()
    env.additional_framework_parameters = {}
    assert training._get_parameter_server_num(env) is None
    env.additional_framework_parameters[training.SAGEMAKER_PARAMETER_SERVER_NUM] = 2
    assert training._get_parameter_server_num(env) is 2


def test_should_run_ps_on_this_host():
    assert training._should_run_ps_on_this_host(HOST_LIST, CURRENT_HOST, 1) is True
    assert training._should_run_ps_on_this_host(HOST_LIST, 'host2', 1) is False


@patch('sagemaker_containers.beta.framework.modules.run_module')
def test_single_machine(run_module, single_machine_training_env):
    training.train(single_machine_training_env)
    run_module.assert_called_with(MODULE_DIR, single_machine_training_env.to_cmd_args(),
                                  single_machine_training_env.to_env_vars(), MODULE_NAME)


@patch('sagemaker_tensorflow_container.training._build_tf_config')
def test_get_env_vars_with_tf_config(build_tf_config, distributed_training_env):
    distributed_training_env.to_env_vars.return_value = {}
    tf_config = {"some_key": "some_value"}
    build_tf_config.return_value = tf_config
    assert training._env_vars_with_tf_config(
        distributed_training_env, ps_task=True) == {"TF_CONFIG": json.dumps(tf_config)}
    build_tf_config.assert_called_once_with(hosts=HOST_LIST, current_host=CURRENT_HOST,
                                            ps_num=PS_NUM, ps_task=True)


@patch('sagemaker_containers.beta.framework.modules.run_module')
@patch('sagemaker_tensorflow_container.training._env_vars_with_tf_config')
def test_run_ps(get_env_vars_with_tf_config, run_module, distributed_training_env):
    get_env_vars_with_tf_config.return_value = {}
    distributed_training_env.to_cmd_args.return_value = CMD_ARGS
    training._run_ps(distributed_training_env)
    get_env_vars_with_tf_config.assert_called_once_with(distributed_training_env, ps_task=True)
    run_module.assert_called_once_with(distributed_training_env.module_dir,
                                       CMD_ARGS,
                                       {},
                                       distributed_training_env.module_name,
                                       wait=False)


@patch('sagemaker_containers.beta.framework.modules.write_env_vars')
@patch('sagemaker_containers.beta.framework.modules.run')
@patch('sagemaker_tensorflow_container.training._env_vars_with_tf_config')
def test_run_worker_no_install(get_env_vars_with_tf_config,
                               run,
                               write_env_vars,
                               distributed_training_env):
    get_env_vars_with_tf_config.return_value = {}
    distributed_training_env.to_cmd_args.return_value = CMD_ARGS
    training._run_worker(distributed_training_env, install_module=False)
    get_env_vars_with_tf_config.assert_called_once_with(distributed_training_env, ps_task=False)
    write_env_vars.assert_called_once_with({})
    run.assert_called_once_with(distributed_training_env.module_name,
                                CMD_ARGS,
                                {})


@patch('sagemaker_containers.beta.framework.modules.run_module')
@patch('sagemaker_tensorflow_container.training._env_vars_with_tf_config')
def test_run_worker_install(get_env_vars_with_tf_config,
                            run_module,
                            distributed_training_env):
    get_env_vars_with_tf_config.return_value = {}
    distributed_training_env.to_cmd_args.return_value = CMD_ARGS
    training._run_worker(distributed_training_env, install_module=True)
    get_env_vars_with_tf_config.assert_called_once_with(distributed_training_env, ps_task=False)
    run_module.assert_called_once_with(distributed_training_env.module_dir,
                                       CMD_ARGS,
                                       {},
                                       distributed_training_env.module_name)


def test_build_tf_config_no_ps():
    assert training._build_tf_config(HOST_LIST, HOST1, ps_num=0) == \
        {"cluster": CLUSTER, "environment": "cloud", "task": MASTER_TASK}
    assert training._build_tf_config(HOST_LIST, HOST2, ps_num=0) == \
        {"cluster": CLUSTER, "environment": "cloud", "task": WORKER_TASK}


def test_build_tf_config_all_ps():
    assert training._build_tf_config(HOST_LIST, HOST1, ps_num=2) ==\
        {"cluster": CLUSTER_WITH_PS, "environment": "cloud", "task": MASTER_TASK}
    assert training._build_tf_config(HOST_LIST, HOST1, ps_num=2, ps_task=True) == \
        {"cluster": CLUSTER_WITH_PS, "environment": "cloud", "task": PS_TASK_1}
    assert training._build_tf_config(HOST_LIST, HOST2, ps_num=2) ==\
        {"cluster": CLUSTER_WITH_PS, "environment": "cloud", "task": WORKER_TASK}
    assert training._build_tf_config(HOST_LIST, HOST2, ps_num=2, ps_task=True) == \
        {"cluster": CLUSTER_WITH_PS, "environment": "cloud", "task": PS_TASK_2}


def test_build_tf_config_1_ps():
    assert training._build_tf_config(HOST_LIST, HOST1, ps_num=1) == \
        {"cluster": CLUSTER_WITH_1_PS, "environment": "cloud", "task": MASTER_TASK}
    assert training._build_tf_config(HOST_LIST, HOST1, ps_num=1, ps_task=True) == \
        {"cluster": CLUSTER_WITH_1_PS, "environment": "cloud", "task": PS_TASK_1}
    assert training._build_tf_config(HOST_LIST, HOST2, ps_num=1) == \
        {"cluster": CLUSTER_WITH_1_PS, "environment": "cloud", "task": WORKER_TASK}


def test_build_tf_config_error():
    with pytest.raises(ValueError) as error:
        training._build_tf_config(HOST_LIST, HOST1, ps_num=0, ps_task=True)
    assert 'Can not have a ps task if there are no parameter servers in the cluster' in str(error)


@patch('sagemaker_tensorflow_container.training.train')
@patch('logging.Logger.setLevel')
@patch('sagemaker_containers.beta.framework.training_env')
@patch('sagemaker_containers.beta.framework.env.read_hyperparameters', return_value={})
def test_main(read_hyperparameters, training_env, set_level, train, single_machine_training_env):
    training_env.return_value = single_machine_training_env
    training.main()
    read_hyperparameters.assert_called_once_with()
    training_env.assert_called_once_with(hyperparameters={})
    set_level.assert_called_once_with(LOG_LEVEL)
    train.assert_called_once_with(single_machine_training_env)
