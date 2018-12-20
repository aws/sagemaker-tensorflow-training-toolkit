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

import os
import tempfile
from urllib.parse import urlparse

import boto3
import pytest
from sagemaker.tensorflow import TensorFlow

RESOURCE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')


@pytest.mark.parametrize('instances, processes', [
    (2, 1)])
def test_distributed_training_cpu_horovod(instances,
                                          processes,
                                          sagemaker_local_session,
                                          docker_image,
                                          tmpdir):
    estimator = TensorFlow(
        entry_point=os.path.join(RESOURCE_PATH, 'mnist', 'horovod_mnist.py'),
        role='SageMakerRole',
        train_instance_type='ml.c5.xlarge',
        train_instance_count=instances,
        image_name=docker_image,
        framework_version='1.12',
        py_version='py3',
        script_mode=True,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_custom_mpi_options': '-verbose',
                         'sagemaker_mpi_num_of_processes_per_host': processes})

    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(RESOURCE_PATH, 'mnist', 'data-distributed'),
        key_prefix='scriptmode/mnist')

    estimator.fit(inputs)


@pytest.mark.parametrize('instances, processes', [
    (5, 1)])
def test_distributed_training_gpu_horovod(instances,
                                          processes,
                                          sagemaker_local_session,
                                          docker_image,
                                          tmpdir):

    estimator = TensorFlow(
        entry_point=os.path.join(RESOURCE_PATH, 'mnist', 'horovod_mnist.py'),
        role='SageMakerRole',
        train_instance_type='ml.p2.xlarge',
        train_instance_count=instances,
        image_name=docker_image,
        framework_version='1.12',
        py_version='py3',
        script_mode=True,
        hyperparameters={'sagemaker_mpi_enabled': True,
                         'sagemaker_mpi_custom_mpi_options': '-verbose',
                         'sagemaker_mpi_num_of_processes_per_host': processes})

    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(RESOURCE_PATH, 'mnist', 'data-distributed'),
        key_prefix='scriptmode/mnist')

    estimator.fit(inputs)
