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

import pytest
from mock import MagicMock, patch

from tf_container import training

MODULE_DIR = 's3://my/bucket'
MODULE_NAME = 'script_name'

@pytest.fixture
def single_machine_training_env():
    env = MagicMock()

    env.module_dir = MODULE_DIR
    env.module_name = MODULE_NAME
    env.hyperparameters = {}

    return env

@patch('sagemaker_containers.beta.framework.modules.run_module')
def test_single_machine(run_module, single_machine_training_env):
    training.train(single_machine_training_env)
    run_module.assert_called_with(MODULE_DIR, single_machine_training_env.to_cmd_args(),
                                  single_machine_training_env.to_env_vars(), MODULE_NAME)
