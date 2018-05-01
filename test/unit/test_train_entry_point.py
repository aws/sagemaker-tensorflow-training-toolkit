#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the 'License').
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the 'license' file accompanying this file. This file is distributed
#  on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.

import os

import pytest
from mock import Mock, patch

from test.unit.utils import mock_import_modules

CHECKPOINT_PATH = 'customer/checkpoint/path'
JOB_NAME = 'test-1234'
TUNING_METRIC = 'some-metric'


# Mock out tensorflow modules.
@pytest.fixture(scope='module')
def modules():
    modules_to_mock = [
        'numpy',
        'grpc.beta',
        'tensorflow.python.framework',
        'tensorflow.core.framework',
        'tensorflow.core.protobuf',
        'tensorflow_serving.apis',
        'tensorflow.python.saved_model.signature_constants',
        'google.protobuf.json_format',
        'tensorflow.core.example',
        'grpc.framework.interfaces.face.face'
    ]
    mock, modules = mock_import_modules(modules_to_mock)

    patcher = patch.dict('sys.modules', modules)
    patcher.start()
    yield mock
    patcher.stop()


@pytest.fixture
def train_entry_point_module(modules):
    import tf_container.train_entry_point
    yield tf_container.train_entry_point


def test_get_checkpoint_dir_without_checkpoint_path(train_entry_point_module):
    env = Mock(name='env', hyperparameters={}, model_dir='model/output')
    checkpoint_dir = train_entry_point_module._get_checkpoint_dir(env)

    assert checkpoint_dir == 'model/output'


def test_get_checkpoint_dir_without_tuning(train_entry_point_module):
    env = Mock(name='env', hyperparameters={'checkpoint_path': CHECKPOINT_PATH}, job_name=None)
    checkpoint_dir = train_entry_point_module._get_checkpoint_dir(env)

    assert checkpoint_dir == CHECKPOINT_PATH


def test_get_checkpoint_dir_with_job_name_in_path(train_entry_point_module):
    checkpoint_path_with_job_name = '{}/checkpoints'.format(JOB_NAME)
    hyperparameters = {
        'checkpoint_path': checkpoint_path_with_job_name,
        'algorithms_tuning_objective_metric': TUNING_METRIC,
    }
    env = Mock(name='env', hyperparameters=hyperparameters, job_name=JOB_NAME)
    checkpoint_dir = train_entry_point_module._get_checkpoint_dir(env)

    assert checkpoint_dir == checkpoint_path_with_job_name


def test_get_checkpoint_dir_without_job_name_env(train_entry_point_module):
    hyperparameters = {
        'checkpoint_path': CHECKPOINT_PATH,
        'algorithms_tuning_objective_metric': TUNING_METRIC,
    }
    env = Mock(name='env', hyperparameters=hyperparameters, job_name=None)
    checkpoint_dir = train_entry_point_module._get_checkpoint_dir(env)

    assert checkpoint_dir == CHECKPOINT_PATH


def test_get_checkpoint_dir_appending_job_name(train_entry_point_module):
    hyperparameters = {
        'checkpoint_path': CHECKPOINT_PATH,
        'algorithms_tuning_objective_metric': TUNING_METRIC,
    }
    env = Mock(name='env', hyperparameters=hyperparameters, job_name=JOB_NAME)
    checkpoint_dir = train_entry_point_module._get_checkpoint_dir(env)

    assert checkpoint_dir == os.path.join(CHECKPOINT_PATH, JOB_NAME, 'checkpoints')
