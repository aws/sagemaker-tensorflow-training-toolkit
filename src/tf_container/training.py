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

import logging
import sagemaker_containers.beta.framework as framework

logger = logging.getLogger(__name__)


def train(env):
    framework.modules.run_module(env.module_dir, env.to_cmd_args(),
                                 env.to_env_vars(), env.module_name)


def main():
    hyperparameters = framework.env.read_hyperparameters()
    env = framework.training_env(hyperparameters=hyperparameters)
    logger.setLevel(env.log_level)
    train(env)
