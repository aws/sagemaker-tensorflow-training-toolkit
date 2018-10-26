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

import pytest
from sagemaker.tensorflow import TensorFlow

from sagemaker_tensorflow_container.training import SAGEMAKER_PARAMETER_SERVER_NUM


def test_mnist(sagemaker_session, ecr_image, instance_type):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           training_steps=1,
                           evaluation_steps=1,
                           train_instance_count=1,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           base_job_name='test-sagemaker-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data'),
        key_prefix='scriptmode/mnist')
    estimator.fit(inputs)


def test_distributed_mnist_no_ps(sagemaker_session, ecr_image, instance_type):
    resource_path = os.path.join(os.path.dirname(__file__), '../..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'distributed_mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           training_steps=1,
                           evaluation_steps=1,
                           train_instance_count=2,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           base_job_name='test-tf-sm-distributed-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data-distributed'),
        key_prefix='scriptmode/mnist-distributed')
    estimator.fit(inputs)


@pytest.mark.parametrize('ps_num', [1, 2])
def test_distributed_mnist_ps(sagemaker_session, ecr_image, instance_type, ps_num):
    resource_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources')
    script = os.path.join(resource_path, 'mnist', 'distributed_mnist.py')
    estimator = TensorFlow(entry_point=script,
                           role='SageMakerRole',
                           training_steps=1,
                           evaluation_steps=1,
                           hyperparameters={SAGEMAKER_PARAMETER_SERVER_NUM: ps_num},
                           train_instance_count=2,
                           train_instance_type=instance_type,
                           sagemaker_session=sagemaker_session,
                           image_name=ecr_image,
                           base_job_name='test-tf-sm-distributed-mnist')
    inputs = estimator.sagemaker_session.upload_data(
        path=os.path.join(resource_path, 'mnist', 'data-distributed'),
        key_prefix='scriptmode/mnist-distributed')
    estimator.fit(inputs)
